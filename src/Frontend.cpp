#include "Frontend.h"

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <Eigen/Geometry>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>


namespace ev {
int max_patch;
static int count_ = 0;
static double coefftime = 0;
static double simpleaddtime = 0;
double variance(const gsl_vector *vec, void *params){

    Eigen::Vector3d w, v, n;
    double p[36] = {};
    w << gsl_vector_get(vec, 0), gsl_vector_get(vec, 1), gsl_vector_get(vec, 2);
    v << gsl_vector_get(vec, 3), gsl_vector_get(vec, 4), gsl_vector_get(vec, 5);
    p[0] = gsl_vector_get(vec, 6);
    p[1] = gsl_vector_get(vec, 7);
    p[2] = 1.;
    for (int i = 3; i != 36; i++) {
        p[i] = gsl_vector_get(vec, 5+i);
    }

//    Eigen::Vector3d w, v, n;
//    w << gsl_vector_get(vec, 0), gsl_vector_get(vec, 1), gsl_vector_get(vec, 2);
//    v << gsl_vector_get(vec, 3), gsl_vector_get(vec, 4), gsl_vector_get(vec, 5);
//    double phi = gsl_vector_get(vec, 6);
//    double psi = gsl_vector_get(vec, 7);
//    n << std::cos(phi) * std::sin(psi), std::sin(phi) * std::sin(psi), std::cos(psi);
//    if (n(2) > 0)
//        n = -n;
    ComputeVarianceFunction *params_ = (ComputeVarianceFunction *) params;
    double cost = 0;
    params_->Intensity(params_->intensity, NULL, w, v, p);
    for (int r = 0; r < 180; ++r) {
        for (int c = 0; c < 240; ++c) {
            cost += std::pow(params_->intensity(r, c), 2);
        }
    }
    cost /= (240*180);
    cost = -cost;
    return cost;
}

inline void ComputeVarianceFunction::warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_v, Eigen::Vector3d& x,
                                   okvis::Duration& t, Eigen::Vector3d& w, Eigen::Vector3d& v, Eigen::Vector3d& n, double z) const {
    double t_ = t.toSec();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + ev::skew(-t_ * w);
    Eigen::Matrix3d H = R * (Eigen::Matrix3d::Identity() + v * z * t_ * n.transpose());
    x_v = H.inverse() * x;
    x_v /= x_v(2);

    x_v = param_.cameraMatrix_ * x_v;

//    if (dW != NULL) {
//        Eigen::MatrixXd dW_(3, 6);
//        dW_ <<     -z_*t_*x(1)*x_w(0), z_*t_*(x(0)*x_w(0)+1), -z_*t_*x(1), t_,  0, -t_*x_w(0),
//               -z_*t_*(x(1)*x_w(1)+1),     z_*t_*x(0)*x_w(1),  z_*t_*x(0),  0, t_, -t_*x_w(1),
//                                    0,                     0,           0,  0,  0,          0;

//        (*dW) = (param_.cameraMatrix_ / z * dW_).block(0, 0, 2, 6);
//    }

//    x_v = param_.cameraMatrix_ * x_v;

}


inline void ComputeVarianceFunction::fuse(Eigen::MatrixXd& image, Eigen::Vector2d& p, bool& polarity) const {
//    std::vector<std::pair<std::vector<int>, double>> pixels;
    auto valid = [](int x, int y)  -> bool {
        return (x >= 0 && x  < 240 && y >= 0 && y < 180);
    };
    int pol = int(polarity) * 2 - 1;

    int x1 = std::floor(p(0));
    int x2 = x1 + 1;
    int y1 = std::floor(p(1));
    int y2 = y1 + 1;
    if (valid(x1, y1)) {
        double a = (x2 - p(0)) * (y2 - p(1)) * pol;
        image(y1, x1) += a;
    }

    if (valid(x1, y2)) {
        double a = -(x2 - p(0)) * (y1 - p(1)) * pol;
        image(y2, x1) += a;
    }

    if (valid(x2, y1)) {
        double a = - (x1 - p(0)) * (y2 - p(1)) * pol;
        image(y1, x2) += a;
    }

    if (valid(x2, y2)) {
        double a = (x1 - p(0)) * (y1 - p(1)) * pol;
        image(y2, x2) += a;
    }

}

void ComputeVarianceFunction::Intensity(Eigen::MatrixXd& image, Eigen::MatrixXd* dIdw,
                                        Eigen::Vector3d& w, Eigen::Vector3d& v, double* normal) const {
    auto& param = param_;

    int patch_num_x = std::ceil(param.array_size_x / param.patch_width);
    int patch_num_y = std::ceil(param.array_size_y / param.patch_width);

    auto patch = [&param, patch_num_x, patch_num_y](Eigen::Vector3d p)  -> int {
        Eigen::Vector3d p_ = param.cameraMatrix_ * p;
        return truncate(std::floor(p_(0)/param.patch_width), 0, patch_num_x-1) * patch_num_y
                + truncate(std::floor(p_(1)/param.patch_width), 0, patch_num_y-1);
    };

    image = Eigen::MatrixXd::Zero(180, 240);
    okvis::Time t0 = em_->events.front().timeStamp;
    ev::count_ = 0;

    for(auto it = em_->events.begin(); it != em_->events.end(); it++) {
        Eigen::Vector3d p(it->measurement.x, it->measurement.y, it->measurement.z);
        Eigen::Vector3d point_warped;
        // project to last frame
        auto t = it->timeStamp - t0;
        Eigen::MatrixXd dWdwvz;
        double i1, i2, z;
        i1 = normal[3 * patch(p)];
        i2 = normal[3 * patch(p) + 1];
        z = std::abs(normal[3 * patch(p) + 2]);
        Eigen::Vector3d n;
        n << std::cos(i1) * std::sin(i2), std::sin(i1) * std::sin(i2), std::cos(i2);
        if (n(2) > 0)
            n = -n;
        if (dIdw != NULL) {
            warp(&dWdwvz, point_warped, p, t, w, v, n, z);
        } else {
            warp(NULL, point_warped, p, t, w, v, n, z);
        }

        Eigen::Vector2d point_camera(point_warped(0), point_warped(1));
        if (dIdw != NULL) {
            Eigen::MatrixXd dWdwv = dWdwvz.block(0, 0, 2, 6);
            std::vector<std::pair<std::vector<int>, double>> pixel_weight;
#if 0
            biInterp(pixel_weight, point_camera, it->measurement.p);

            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                std::vector<int> p_ = p_it->first;               
                double dc = point_warped(0) - p_[0];
                double dr = point_warped(1) - p_[1];
                int pol = int(it->measurement.p) * 2 - 1;
                double dW = 0.;
                Eigen::VectorXd dwv = Eigen::VectorXd::Zero(6);

                if (dr > 0) {
                    dW = (std::abs(dc) - 1) * pol;
                } else if (dr < 0) {
                    dW = (1 - std::abs(dc)) * pol;
                }
                dwv += dW * dWdwv.row(1);
                if (dc > 0) {
                    dW = (std::abs(dr) - 1) * pol;
                } else if (dc < 0) {
                    dW = (1 - std::abs(dr)) * pol;
                }
                dwv += dW * dWdwv.row(0);

                for (int i = 0; i != 6; i++) {
                    dIdw->coeffRef(p_[0] * 180 + p_[1], i) += dwv(i);
                }

                image.coeffRef(p_[1], p_[0]) += p_it->second;
            }

//#else
            // lacks adjustment for rotation!
            // delta_y

            Eigen::Vector2d h1(point_camera(1) + .5, point_camera(0));
            biInterp(pixel_weight, h1, it->measurement.p);
            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                Eigen::VectorXd dwv = p_it->second * dWdv.row(1);
                for (int i = 0; i != 3; i++) {
                    dIdw->coeffRef((p_it->first)[0] * 180 + (p_it->first)[1], i) += dwv(i);
                }
            }
            Eigen::Vector2d h2(point_camera(1) - .5, point_camera(0));
            biInterp(pixel_weight, h2, it->measurement.p);
            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                Eigen::VectorXd dwv = p_it->second * dWdv.row(1);
                for (int i = 0; i != 3; i++) {
                    dIdw->coeffRef((p_it->first)[0] * 180 + (p_it->first)[1], i) -= dwv(i);
                }
            }

            // delta_x
            Eigen::Vector2d h3(point_camera(1), point_camera(0) + .5);
            biInterp(pixel_weight, h3, it->measurement.p);
            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                Eigen::VectorXd dwv = p_it->second * dWdv.row(0);
                for (int i = 0; i != 3; i++) {
                    dIdw->coeffRef((p_it->first)[0] * 180 + (p_it->first)[1], i) += dwv(i);
                }

            }
            Eigen::Vector2d h4(point_camera(1), point_camera(0) - .5);
            biInterp(pixel_weight, h4, it->measurement.p);
            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                Eigen::VectorXd dwv = p_it->second * dWdv.row(0);
                for (int i = 0; i != 3; i++) {
                    dIdw->coeffRef((p_it->first)[0] * 180 + (p_it->first)[1], i) -= dwv(i);
                }

            }

            fuse(image, point_camera, it->measurement.p);
#endif

        } else {
            fuse(image, point_camera, it->measurement.p);
        }
    }

//    for (auto p_it = pixels.begin(); p_it != pixels.end(); p_it++) {
//        image((p_it->first)[1], (p_it->first)[0]) += p_it->second;
//    }

//    LOG(INFO) << "warp time for 50000 events is " << time/1000000 << " milliseconds.";
//    LOG(INFO) << "fuse time for 50000 events is " << time2/1000000 << " milliseconds.";
//    LOG(INFO) << "biinterp time for 50000 events is " << time2/1000000 << " milliseconds.";
//    LOG(INFO) << "coeff time for 50000 events is " << ev::coefftime/1000000 << " milliseconds.";
//    LOG(INFO) << "simple add time for 50000 events is " << ev::simpleaddtime/1000000 << " milliseconds.";
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    cv::GaussianBlur(src, dst, cv::Size(0, 0), sigma, 0);
    cv::cv2eigen(dst, image);
}

// Constructor.
Frontend::Frontend()
    : isInitialized_(false),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(
          std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2) {
    initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(
        std::shared_ptr<okvis::MultiFrame> frameOut,
        const okvis::kinematics::Transformation& T_WC,
        const std::vector<cv::KeyPoint> * keypoints) {
    return true;
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
        okvis::Estimator& estimator,
        okvis::kinematics::Transformation& /*T_WS_propagated*/, // TODO sleutenegger: why is this not used here?
        const ev::Parameters & params,
        const std::shared_ptr<okvis::MapPointVector> /*map*/, // TODO sleutenegger: why is this not used here?
        std::shared_ptr<okvis::MultiFrame> framesInOut,
        bool *asKeyframe) {
    return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const ev::Parameters& Params,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const {

    return false;
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::needANewKeyframe(
        const okvis::Estimator& estimator,
        std::shared_ptr<okvis::MultiFrame> currentFrame) {

    return true;
}

// Match a new multiframe to existing keyframes
template<class MATCHING_ALGORITHM>
int Frontend::matchToKeyframes(okvis::Estimator& estimator,
                               const okvis::VioParameters & params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers) {
    int retCtr = 0;

    return retCtr;
}

// Match a new multiframe to the last frame.
template<class MATCHING_ALGORITHM>
int Frontend::matchToLastFrame(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers) {

    int retCtr = 0;
    return retCtr;
}

// Perform 3D/2D RANSAC.
int Frontend::runRansac3d2d(okvis::Estimator& estimator,
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers) {
    int numInliers = 0;
    return numInliers;
}

// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2d(okvis::Estimator& estimator,
                            const okvis::VioParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly) {
    return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void Frontend::initialiseBriskFeatureDetectors() {
    featureDetectorMutex_.lock();

    featureDetector_.clear();
    descriptorExtractor_.clear();

    featureDetector_ = brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
                briskDetectionThreshold_, briskDetectionOctaves_,
                briskDetectionAbsoluteThreshold_,
                briskDetectionMaximumKeypoints_);

    descriptorExtractor_= brisk::BriskDescriptorExtractor(
                briskDescriptionRotationInvariance_,
                briskDescriptionScaleInvariance_);

    featureDetectorMutex_.unlock();
}

}
