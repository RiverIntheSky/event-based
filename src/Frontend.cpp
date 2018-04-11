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

void Contrast::warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration& t, Eigen::Vector3d& w) {
    Eigen::Vector3d x_ = x.homogeneous();
    x_w = x_ + (w * t.toSec()).cross(x_);
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

void Contrast::Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement> &em, Parameters& param, Eigen::Vector3d w) {
    okvis::Time t1 = em->events.back().timeStamp;
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = t1 - it->timeStamp;
        warp(point_warped, p, t, w);
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

double Contrast::getIntensity(int x, int y, Eigen::Vector3d w) const {
    if (x == 0 && y == 0) {
        intensity = Eigen::MatrixXd::Zero(180, 240);
        Intensity(intensity, em, param, w);
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
