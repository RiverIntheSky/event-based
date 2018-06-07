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
int max_patch;
bool ComputeVarianceFunction::Evaluate(double const* const* parameters,
                                       double* residuals,
                                       double** jacobians) const {

    std::string files_path = param_.path + "/" + param_.experiment_name + "/" + std::to_string(param_.window_size) + "/";
    int area = 180 * 240;
    residuals[0] = 0;
    Eigen::Vector3d v;
    v << (*parameters)[0], (*parameters)[1], (*parameters)[2];
    const double* const* z = parameters + 1;

    // derivatives of intensity with respect to each parameters
    std::vector<Eigen::SparseMatrix<double>> dIdw_;
    // the mean of the derivatives
    std::vector<double> dIm;

    if (jacobians != NULL && jacobians[0] != NULL) {

        for (int i = 0; i != 3; i++) {
                jacobians[0][i] = 0;
        }

        for (int i = 0; i != param_.patch_num; i++) {
                jacobians[1][i] = 0;
        }

        Eigen::SparseMatrix<double> dIdw(param_.array_size_y * param_.array_size_x, 3 + param_.patch_num);
        Intensity(intensity, &dIdw, v, z);
        cv::Mat src, dst;

        for (int i = 0; i != 3 + param_.patch_num; i++){
            Eigen::MatrixXd d = dIdw.col(i);
            d.resize(180, 240);
            cv::eigen2cv(d, src);
            cv::GaussianBlur(src, dst, cv::Size(0, 0), sigma, 0);
            cv::cv2eigen(dst, d);
            Eigen::SparseMatrix<double> s = d.sparseView();
            dIdw_.push_back(s);
            dIm.push_back(cv::mean(dst)[0]);
        }

    } else {
        Intensity(intensity, NULL, v, z);
    }

    double Im = (intensity * Eigen::VectorXd::Ones(intensity.cols())).sum() / area;

    for (int s = 0; s < intensity.outerSize(); ++s) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(intensity, s); it; ++it) {
            double rho = it.value() - Im;
            residuals[0] += std::pow(rho, 2);

            int y_ = it.row();
            int x_ = it.col();

            if (jacobians != NULL && jacobians[0] != NULL) {
                for (int i = 0; i != 1; i++) {
                    jacobians[0][i] += rho * ((dIdw_[i]).coeffRef(y_, x_) - dIm[i]);
                }

                for (int i = 0; i != param_.patch_num; i++) {
                    jacobians[1][i] += rho * ((dIdw_[i + 3]).coeffRef(y_, x_) - dIm[i + 3]);
                }
            }
        }
    }
    residuals[0] += (area - intensity.nonZeros()) * std::pow(Im, 2);

    residuals[0] /= area;
    residuals[0] = 1./residuals[0];

    if (jacobians != NULL && jacobians[0] != NULL) {
        for (int i = 0; i != 1; i++) {
            jacobians[0][i] += (area - intensity.nonZeros()) * Im * dIm[i];
            jacobians[0][i] *= (-2 * std::pow(residuals[0], 2) / area);
            jacobians[0][i] = 0;
#ifndef NDEBUG
            LOG(INFO)<< "jv:" << jacobians[0][i];
#endif
        }

        for (int i = 0; i != param_.patch_num; i++) {
            jacobians[1][i] += (area - intensity.nonZeros()) * Im * dIm[i + 3];
            jacobians[1][i] *= (-2 * std::pow(residuals[0], 2) / area);
            LOG(INFO) << "jz:" << jacobians[1][i];
        }
        LOG(INFO) << "------------------";

        std::ofstream  c_file(files_path + "jacobian.txt", std::ios_base::app);
        if (c_file.is_open()) {
            c_file << (*z)[0] << " " << residuals[0] << ' ' << std::pow(residuals[0], 2)/2<< ' ' << jacobians[1][0] << '\n';
            c_file.close();
        } else
            std::cout << "怎么肥四"<<std::endl;
    }

    std::ofstream  c_file(files_path + "residual.txt", std::ios_base::app);
    if (c_file.is_open()) {
        c_file << (*z)[0] << " " << residuals[0] << ' ' << std::pow(residuals[0], 2)/2 << '\n';
        c_file.close();
    } else
        std::cout << "怎么肥四"<<std::endl;

//    if (jacobians != NULL && jacobians[0] != NULL) {
//        LOG(INFO) << "jacobian & residual " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
//                  << " nanoseconds";
//    } else {
//        LOG(INFO) << "residual " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
//                  << " nanoseconds";
//    }

    return true;
}

inline void ComputeVarianceFunction::warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_v, Eigen::Vector3d& x,
                                   okvis::Duration& t, Eigen::Vector3d& v, const double z) const {

    double t_ = t.toSec();
    x_v = (z - v(2) * t_) * x + v * t_;
    x_v /= x_v(2);

    Eigen::Matrix3d cameraMatrix_;

    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);


    if (dW != NULL) {
        Eigen::MatrixXd dW_(3, 4);
        dW_ << t_,  0, -x(0) * t_, x(0) - x_v(0),
                0, t_, -x(1) * t_, x(1) - x_v(1),
                0,  0,          0,             0;

        (*dW) = (cameraMatrix_ / z * dW_).block(0, 0, 2, 4);

    }

    x_v = cameraMatrix_ * x_v;
}

void ComputeVarianceFunction::fuse(Eigen::SparseMatrix<double>& image, Eigen::Vector2d& p, bool& polarity) const {
    std::vector<std::pair<std::vector<int>, double>> pixels;
    biInterp(pixels, p, polarity);
    for (auto p_it = pixels.begin(); p_it != pixels.end(); p_it++) {
        image.coeffRef((p_it->first)[1], (p_it->first)[0]) += p_it->second;
    }
}

inline void ComputeVarianceFunction::biInterp(std::vector<std::pair<std::vector<int>, double>>& pixel_weight, Eigen::Vector2d& point, bool& polarity) const {
    auto valid = [](int x, int y)  -> bool {
        return (x >= 0 && x  < 240 && y >= 0 && y < 180);
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

void ComputeVarianceFunction::Intensity(Eigen::SparseMatrix<double>& image, Eigen::SparseMatrix<double>* dIdw,
                                        Eigen::Vector3d& v, const double * const *z) const {
    image = Eigen::SparseMatrix<double>(180, 240);
    okvis::Time t0 = em_->events.front().timeStamp;
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
    auto& param = param_;

    int patch_num_x = std::ceil(param.array_size_x / param.patch_width);
    int patch_num_y = std::ceil(param.array_size_y / param.patch_width);

    auto patch = [&param, cameraMatrix_, patch_num_x, patch_num_y](Eigen::Vector3d p)  -> int {
        Eigen::Vector3d p_ = cameraMatrix_ * p;
        return truncate(std::floor(p_(0)/param.patch_width), 0, patch_num_x-1) * patch_num_y
                + truncate(std::floor(p_(1)/param.patch_width), 0, patch_num_y-1);
    };

    for(auto it = em_->events.begin(); it != em_->events.end(); it++) {
        Eigen::Vector3d p(it->measurement.x, it->measurement.y, it->measurement.z);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = it->timeStamp - t0;
        Eigen::MatrixXd dWdvz;
        //int patch_nr = patch(p);
        int patch_nr = 0;
        if (dIdw != NULL) {
            warp(&dWdvz, point_warped, p, t, v, (*z)[patch_nr]);
//            if (patch(p) == max_patch)
//                dWdvz.col(3) = Eigen::Vector2d::Zero();
        } else {
            warp(NULL, point_warped, p, t, v, (*z)[patch_nr]);
        }
        Eigen::Vector2d point_camera(point_warped(0), point_warped(1));

        if (dIdw != NULL) {
            Eigen::MatrixXd dWdv = dWdvz.block(0, 0, 2, 3);
            Eigen::Vector2d dWdz = dWdvz.col(3);
            std::vector<std::pair<std::vector<int>, double>> pixel_weight;
#if 1
            biInterp(pixel_weight, point_camera, it->measurement.p);
            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                std::vector<int> p_ = p_it->first;               
                double dc = point_warped(0) - p_[0];
                double dr = point_warped(1) - p_[1];
                int pol = int(it->measurement.p) * 2 - 1;
                double dW;
                double dz_ = 0.;
                Eigen::Vector3d dwv = Eigen::Vector3d::Zero();
                if (dr > 0) {
                    dW = (std::abs(dc) - 1) * pol;
                } else if (dr < 0) {
                    dW = (1 - std::abs(dc)) * pol;
                }
                dwv += dW * dWdv.row(1);
                dz_ += dW * dWdz(1);
                if (dc > 0) {
                    dW = (std::abs(dr) - 1) * pol;
                } else if (dc < 0) {
                    dW = (1 - std::abs(dr)) * pol;
                }
                dwv += dW * dWdv.row(0);
                dz_ += dW * dWdz(0);

                for (int i = 0; i != 3; i++) {
                    dIdw->coeffRef(p_[0] * 180 + p_[1], i) += dwv(i);
                }
                dIdw->coeffRef(p_[0] * 180 + p_[1], 3 + patch_nr) += dz_;

                image.coeffRef(p_[1], p_[0]) += p_it->second;
            }
#else
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
    Eigen::MatrixXd image_ = image.toDense();
    cv::Mat src, dst;
    cv::eigen2cv(image_, src);
    cv::GaussianBlur(src, dst, cv::Size(0, 0), sigma, 0);
    cv::cv2eigen(dst, image_);
    image = image_.sparseView();

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
