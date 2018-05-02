#include "Frontend.h"

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
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


bool ComputeVarianceFunction::Evaluate(double const* const* parameters,
                                       double* residuals,
                                       double** jacobians) const {
    int area = 180 * 240;
    residuals[0] = 0;
    Eigen::Vector3d w;
    w << (*parameters)[0], (*parameters)[1], (*parameters)[2];
    //Eigen::MatrixXd intensity;
    Eigen::MatrixXd dIdw1;
    Eigen::MatrixXd dIdw2;
    Eigen::MatrixXd dIdw3;
    double dIm1;
    double dIm2;
    double dIm3;
    if (jacobians != NULL && jacobians[0] != NULL) {

        jacobians[0][0] = 0;
        jacobians[0][1] = 0;
        jacobians[0][2] = 0;

        Eigen::MatrixXd dIdw(area, 3);
        Intensity(intensity, dIdw, w);

        cv::Mat src, dst;

        dIdw1 = dIdw.col(0);
        dIdw1.resize(180, 240);
        cv::eigen2cv(dIdw1, src);
        cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
        cv::cv2eigen(dst, dIdw1);

        dIdw2 = dIdw.col(1);
        dIdw2.resize(180, 240);
        cv::eigen2cv(dIdw2, src);
        cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
        cv::cv2eigen(dst, dIdw2);

        dIdw3 = dIdw.col(2);
        dIdw3.resize(180, 240);
        cv::eigen2cv(dIdw3, src);
        cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
        cv::cv2eigen(dst, dIdw3);

        dIm1 = dIdw1.mean();
        dIm2 = dIdw2.mean();
        dIm3 = dIdw3.mean();

    } else {
        Intensity(intensity,w);
    }

    double Im = intensity.mean();

    for (int x_ = 0; x_ < 240; x_++) {
        for (int y_ = 0; y_ < 180; y_++) {
            double rho = intensity(y_, x_) - Im;
            residuals[0] += std::pow(rho, 2);

            if (jacobians != NULL && jacobians[0] != NULL) {
                jacobians[0][0] += rho * (dIdw1(y_, x_) - dIm1);
                jacobians[0][1] += rho * (dIdw2(y_, x_) - dIm2);
                jacobians[0][2] += rho * (dIdw3(y_, x_) - dIm3);
            }

        }
    }
    residuals[0] /= area;
    residuals[0] = 1./residuals[0];

    if (jacobians != NULL && jacobians[0] != NULL) {
        for (int i = 0; i != 3; i++) {
            // negative??
            jacobians[0][i] *= (-2 * std::pow(residuals[0], 2) / area);
            //jacobians[0][i] *= (-2 * std::pow(residuals[0], 2) / area);
            LOG(INFO) << "j: " << jacobians[0][i];
        }
    }

    return true;
}

void ComputeVarianceFunction::warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x,
                                   okvis::Duration& t, Eigen::Vector3d& w) const {
    Eigen::Vector3d x_ = x.homogeneous();
    if (w.norm() == 0) {
        x_w = x_;
    } else {
//        Eigen::AngleAxisd aa(w.norm()* t.toSec(), w.normalized());
//        x_w = aa.toRotationMatrix() * x_;
        x_w = x_ + (w * t.toSec()).cross(x_);
    }
}

void ComputeVarianceFunction::warp(Eigen::MatrixXd& dWdw, Eigen::Vector3d& x_w, Eigen::Vector2d& x,
                                   okvis::Duration& t, Eigen::Vector3d& w) const {
    Eigen::Vector3d x_ = x.homogeneous();
    double t_ = t.toSec();
    if (w.norm() == 0 || t_ == 0) {
        x_w = x_;
        dWdw = Eigen::MatrixXd::Zero(2, 3);
    } else {
        Eigen::Matrix3d cameraMatrix_;
        cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
//        Eigen::AngleAxisd aa(w.norm()* t.toSec(), w.normalized());
//        Eigen::Matrix3d R = aa.toRotationMatrix();
//        x_w =  R * x_;
//        Eigen::Matrix3d dWdw_ = -R*t_*ev::skew(x_)*
//                ((w*w.transpose()+(R.transpose()-Eigen::Matrix3d::Identity())*ev::skew(w)/t_)/w.squaredNorm());
//        dWdw = (cameraMatrix_ * (dWdw_/x_w(2)-x_w*dWdw_.row(2)/std::pow(x_w(2),2))).block(0, 0, 2, 3);
        Eigen::Vector3d w_ = w*t.toSec();
        x_w = x_ + (w * t.toSec()).cross(x_);
        dWdw = (cameraMatrix_ * (Eigen::Matrix3d::Identity() + ev::skew(w_))).block(0, 0, 2, 3);
    }

    //    x_w /= x_w(2);
}

void ComputeVarianceFunction::fuse(Eigen::MatrixXd& image, Eigen::Vector2d p, bool& polarity) const {
    std::vector<std::pair<std::vector<int>, double>> pixels;
    biInterp(pixels, p, polarity);
    for (auto p_it = pixels.begin(); p_it != pixels.end(); p_it++) {
        // ev::Contrast::events_number+=std::abs(p_it->second);
        image((p_it->first)[0], (p_it->first)[1]) += p_it->second;
    }
}

void ComputeVarianceFunction::biInterp(std::vector<std::pair<std::vector<int>, double>>& pixel_weight, Eigen::Vector2d point, bool& polarity) const {
    auto valid = [](int x, int y)  -> bool {
        return (x >= 0 && x  < 180 && y >= 0 && y < 240);
    };

    pixel_weight.clear();

    int pol = int(polarity) * 2 - 1;

    int x1 = std::floor(point(0));
    int x2 = x1 + 1;
    int y1 = std::floor(point(1));
    int y2 = y1 + 1;

    if (valid(x1, y1)) {
        double a = (x2 - point(0)) * (y2 - point(1)) * pol;
        std::vector<int> p = {x1, y1};
        pixel_weight.push_back(std::make_pair(p, a));
    }

    if (valid(x1, y2)) {
        double a = - (x2 - point(0)) * (y1 - point(1)) * pol;
        std::vector<int> p = {x1, y2};
        pixel_weight.push_back(std::make_pair(p, a));
    }

    if (valid(x2, y1)) {
        double a = - (x1 - point(0)) * (y2 - point(1)) * pol;
        std::vector<int> p = {x2, y1};
        pixel_weight.push_back(std::make_pair(p, a));
    }

    if (valid(x2, y2)) {
        double a = (x1 - point(0)) * (y1 - point(1)) * pol;
        std::vector<int> p = {x2, y2};
        pixel_weight.push_back(std::make_pair(p, a));
    }
}

void ComputeVarianceFunction::Intensity(Eigen::MatrixXd& image, Eigen::Vector3d w) const {
    image = Eigen::MatrixXd::Zero(param_.array_size_y, param_.array_size_x);
    okvis::Time t0 = em_->events.front().timeStamp;
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
//     ev::Contrast::events_number = .0;
    for(auto it = em_->events.begin(); it != em_->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = it->timeStamp - t0;
        warp(point_warped, p, t, w);

        // z'/z
        it->measurement.z = point_warped(2);
        point_warped /= point_warped(2);
        Eigen::Vector3d point_camera = cameraMatrix_ * point_warped;


        fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);

    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}

void ComputeVarianceFunction::Intensity(Eigen::MatrixXd& image, Eigen::MatrixXd& dIdw, Eigen::Vector3d w) const {
    image = Eigen::MatrixXd::Zero(param_.array_size_y, param_.array_size_x);
    dIdw = Eigen::MatrixXd::Zero(param_.array_size_y * param_.array_size_x, 3);
    okvis::Time t0 = em_->events.front().timeStamp;
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
    std::vector<std::pair<std::vector<int>, double>> pixel_weight;
//    ev::Contrast::events_number = .0;
    for(auto it = em_->events.begin(); it != em_->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = it->timeStamp - t0;
        Eigen::MatrixXd dWdw;
        warp(dWdw, point_warped, p, t, w);

        // z'/z
        it->measurement.z = point_warped(2);
        point_warped /= point_warped(2);
        Eigen::Vector3d point_camera = cameraMatrix_ * point_warped;

        // delta_x'
        biInterp(pixel_weight, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
        for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
            double dr = point_camera(0) - (p_it->first)[0];
            double dc = point_camera(1) - (p_it->first)[1];
            if (dr > 0)
                dIdw.row((p_it->first)[0] * 240 + (p_it->first)[1]) += ((std::abs(dc) - 1) * dWdw.row(0) * it->measurement.p);
            if (dr < 0)
                dIdw.row((p_it->first)[0] * 240 + (p_it->first)[1]) += ((1 - std::abs(dc)) * dWdw.row(0) * it->measurement.p);
            if (dc > 0)
                dIdw.row((p_it->first)[0] * 240 + (p_it->first)[1]) += ((1 - std::abs(dr)) * dWdw.row(1) * it->measurement.p);
            if (dc < 0)
                dIdw.row((p_it->first)[0] * 240 + (p_it->first)[1]) += ((std::abs(dr) - 1) * dWdw.row(1) * it->measurement.p);
            image((p_it->first)[0], (p_it->first)[1]) += p_it->second;
        }
        // fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);

}

void Contrast::warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration& t, Eigen::Vector3d& w) {
    Eigen::Vector3d x_ = x.homogeneous();
    if (w.norm() == 0) {
        x_w = x_;
    } else {
        Eigen::AngleAxisd aa(w.norm()* t.toSec(), w.normalized());
        x_w = aa.toRotationMatrix() * x_;
    }
    // x_w = x_ + (w * t.toSec()).cross(x_);
    x_w /= x_w(2);
}

void SE3::warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration &t, Eigen::Vector3d& w, Eigen::Vector3d& tr) {
    Eigen::Vector3d x_ = x.homogeneous();
    if (w.norm() == 0) {
        x_w = x_ + tr * t.toSec();
    } else {
        Eigen::AngleAxisd aa(w.norm()* t.toSec(), w.normalized());
        x_w = aa.toRotationMatrix() * x_ + tr * t.toSec();
    }
    // x_w(2) = z_w/z;
    x_w /= x_w(2);
}

void Contrast::fuse(Eigen::MatrixXd& image, Eigen::Vector2d p, bool& polarity) {
#if 1
    int pol = int(polarity) * 2 - 1;
#else
    int pol = 1;
#endif

    Eigen::Vector2d p1(std::floor(p(0)), std::floor(p(1)));
    Eigen::Vector2d p2(p1(0), p1(1) + 1);
    Eigen::Vector2d p3(p1(0) + 1, p1(1));
    Eigen::Vector2d p4(p1(0) + 1, p1(1) + 1);

    double a1 = (p4(0) - p(0)) * (p4(1) - p(1));
    double a2 = -(p3(0) - p(0)) * (p3(1) - p(1));
    double a3 = -(p2(0) - p(0)) * (p2(1) - p(1));
    double a4 =  (p1(0) - p(0)) * (p1(1) - p(1));

    image(p1(0), p1(1)) += a1 * pol;
    image(p2(0), p2(1)) += a2 * pol;
    image(p3(0), p3(1)) += a3 * pol;
    image(p4(0), p4(1)) += a4 * pol;
}

void Contrast::synthesizeEventFrame(Eigen::MatrixXd &frame, std::shared_ptr<eventFrameMeasurement>& em) {
    Intensity(frame, em, param, Eigen::Vector3d::Zero());
}

void Contrast::polarityEventFrame(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em, bool po) {
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param.cameraMatrix, cameraMatrix_);
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        if (it->measurement.p == po) {
            Eigen::Vector2d p(it->measurement.x, it->measurement.y);
            Eigen::Vector3d point_camera = cameraMatrix_ * p.homogeneous();

            // discard point outside frustum
            if (point_camera(0) > 0 && point_camera(0) < 179
                    && point_camera(1) > 0 && point_camera(1) < 239) {
                fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
            }
        }
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    int kernelSize = 3;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}

void Contrast::Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement> &em, Parameters& param, Eigen::Vector3d w) {
    okvis::Time t1 = em->events.back().timeStamp;
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param.cameraMatrix, cameraMatrix_);
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = t1 - it->timeStamp;
        warp(point_warped, p, t, w);

        Eigen::Vector3d point_camera = cameraMatrix_ * point_warped;
        if (point_camera(0) > 0 && point_camera(0) < 179
                && point_camera(1) > 0 && point_camera(1) < 239) {
            fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
        } else {
            // LOG(INFO) << "discard point outside frustum";
        }
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    int kernelSize = 3;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}


void SE3::Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em,
                    Parameters& param, Eigen::Vector3d& w, Eigen::Vector3d& tr) {
    okvis::Time t1 = em->events.back().timeStamp;
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = t1 - it->timeStamp;
        warp(point_warped, p, t, w, tr);
        Eigen::Matrix3d cameraMatrix_;
        cv::cv2eigen(param.cameraMatrix, cameraMatrix_);
        Eigen::Vector3d point_camera = cameraMatrix_ * point_warped;
        if (point_camera(0) > 0 && point_camera(0) < 179
                && point_camera(1) > 0 && point_camera(1) < 239) {
            fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
        } else {
            // LOG(INFO) << "discard point outside frustum";
        }
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    int kernelSize = 3;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}

double Contrast::getIntensity(int x, int y, Eigen::Vector3d& w) {
    if (x == 0 && y == 0) {
        intensity = Eigen::MatrixXd::Zero(180, 240);
        Intensity(intensity, em, param, w);
    }
    return intensity(x, y);
}

double SE3::getIntensity(int x, int y, Eigen::Vector3d& w, Eigen::Vector3d& tr) {
    if (x == 0 && y == 0) {
        intensity = Eigen::MatrixXd::Zero(180, 240);
        Intensity(intensity, em, param, w, tr);
    }
    return intensity(x, y);
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
