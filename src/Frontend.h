#pragma once

#include <mutex>
#include <limits>

#include <okvis/assert_macros.hpp>
#include <okvis/Estimator.hpp>
#include <okvis/VioFrontendInterface.hpp>
#include <okvis/timing/Timer.hpp>
#include <okvis/DenseMatcher.hpp>
#include <Eigen/StdVector>
#include "parameters.h"
#include "event.h"
#include "util/utils.h"

/// \brief okvis Main namespace of this package.
namespace ev {
extern int count;
class ComputeVarianceFunction : public ceres::SizedCostFunction<1, 3, 3, 12> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ComputeVarianceFunction(std::shared_ptr<eventFrameMeasurement> em, Parameters parameters):
        em_(em), param_(parameters) {}

    virtual ~ComputeVarianceFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    void biInterp(std::vector<std::pair<std::vector<int>, double>>& pixel_weight, Eigen::Vector2d& point, bool& polarity) const;

    void warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_v, Eigen::Vector3d &x,
              okvis::Duration& t, Eigen::Vector3d& w, Eigen::Vector3d& v, const double z) const;

    void fuse(Eigen::MatrixXd& image, Eigen::Vector2d &p, bool& polarity) const;

    void Intensity(Eigen::MatrixXd& image, Eigen::MatrixXd* dIdw, Eigen::Vector3d& w, Eigen::Vector3d& v, const double * const *z) const;

    std::shared_ptr<eventFrameMeasurement> em_;
    Parameters param_;
    int kernelSize = 3;
    static Eigen::MatrixXd intensity;
};


struct Contrast {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Contrast(){
        //        compute_variance.reset(
        //                    new ceres::CostFunctionToFunctor<1, 1>(new ComputeVarianceFunction));
    }

    // rotate vector x by w*t
    static void warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration &t, Eigen::Vector3d& w);

    // dealing with non-integer coordinates
    // bilinear interpolation
    static void fuse(Eigen::MatrixXd& image, Eigen::Vector2d p, bool& polarity);

    static void synthesizeEventFrame(Eigen::MatrixXd& frame, std::shared_ptr<eventFrameMeasurement>& em);

    static void polarityEventFrame(Eigen::MatrixXd& frame, std::shared_ptr<eventFrameMeasurement>& em, bool po);

    static void Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em, Parameters& param, Eigen::Vector3d w);

    static void Intensity(Eigen::MatrixXd& image, Eigen::Vector3d& w, Eigen::Vector3d& v, double* z);
    //    static void Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em,
    //                          Parameters& param, Eigen::Vector3d& w, Eigen::Vector3d& tr);

    static double getIntensity(int x, int y, Eigen::Vector3d& w);

    template <typename T>
    bool operator()(const T** w, T* residual) const {

        //        residual[0] = 0;
        //        double m = intensity.mean();
        //        for (int x_ = 0; x_ < 240; x_++) {
        //            for (int y_ = 0; y_ < 180; y_++) {
        //                residual[0] += std::pow(getIntensity(y_, x_, w_) - m, 2);
        //            }
        //        }
        //        residual[0] /= (240*180);
        //        residual[0] = std::sqrt(1./residual[0]);
        T f;
        (*compute_variance)(&w, &f);
        return true;
    }
    static std::shared_ptr<eventFrameMeasurement> em_;
    static double   events_number;
    static Eigen::MatrixXd intensity;
    static Parameters param_;
    std::unique_ptr<ceres::CostFunctionToFunctor<1, 1> > compute_variance;

};

struct SE3 : Contrast {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE3(){}

    // rotate vector x by w*t
    static void warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration &t, Eigen::Vector3d& w, Eigen::Vector3d& tr);

    static void Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em,
                          Parameters& param, Eigen::Vector3d& w, Eigen::Vector3d& tr);

    static double getIntensity(int x, int y, Eigen::Vector3d& w, Eigen::Vector3d& tr);

    template <typename T>
    bool operator()(const T* t1, const T* t2, const T* t3,
                    const T* w1, const T* w2, const T* w3,
                    T* residual) const {
        Eigen::Vector3d w_;
        w_ << *w1, *w2, *w3;
        Eigen::Vector3d t_;
        t_ << *t1, *t2, *t3;
        residual[0] = 0;
        double m = intensity.mean();
        for (int x_ = 0; x_ < 240; x_++) {
            for (int y_ = 0; y_ < 180; y_++) {
                residual[0] += std::pow(getIntensity(y_, x_, w_, t_) - m, 2);
            }
        }
        residual[0] /= (240*180);
        residual[0] = std::sqrt(1./residual[0]);

        return true;
    }
};

class imshowCallback: public ceres::IterationCallback {
public:
    imshowCallback(double* w, ComputeVarianceFunction* c): w_(w), c_(c) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        if (summary.step_is_successful || summary.iteration == 0) {
            std::string caption = "cost = " + std::to_string(summary.cost);
            ev::imshowRescaled(c_->intensity, msec_, s_, caption);
//            LOG(INFO) << "w:" << *w_ << " " << *(w_+1) << " " << *(w_+2);
        }        
        return ceres::SOLVER_CONTINUE;
    }

private:
    double* w_;
    ComputeVarianceFunction* c_;
    int msec_ = 1e5;
    std::string s_ = "optimizing";
};

/**
 * @brief A frontend using BRISK features
 */
class Frontend {
public:
    OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

#ifdef DEACTIVATE_TIMERS
    typedef okvis::timing::DummyTimer TimerSwitchable;
#else
    typedef okvis::timing::Timer TimerSwitchable;
#endif

    /**
   * @brief Constructor.
   * @param
   */
    Frontend();
    virtual ~Frontend() {
    }

    ///@{
    /**
   * @brief Detection and descriptor extraction on a per image basis.
   * @remark This method is threadsafe.
   * @param
   * @param frameOut    Multiframe containing the frames.????
   *                    Resulting keypoints and descriptors are saved in here.
   * @param T_WC        Pose of reference camera frame, T_{t_k^f}
   * @param[in] keypoints If the keypoints are already available from a different source, provide them here
   *                      in order to skip detection.
   * @warning Using keypoints from a different source is not yet implemented.
   * @return True if successful.
   */
    virtual bool detectAndDescribe(
            std::shared_ptr<okvis::MultiFrame> frameOut,
            const okvis::kinematics::Transformation& T_WC,
            const std::vector<cv::KeyPoint> * keypoints);

    /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in another thread.
   * @param estimator       Estimator.
   * @param T_WS_propagated Pose of sensor at image capture time.
   * @param params          Configuration parameters.
   * @param map             Unused.
   * @param framesInOut     Multiframe including the descriptors of all the keypoints.
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
    virtual bool dataAssociationAndInitialization(
            okvis::Estimator& estimator,
            okvis::kinematics::Transformation& T_WS_propagated,
            const ev::Parameters & params,
            const std::shared_ptr<okvis::MapPointVector> map,
            std::shared_ptr<okvis::MultiFrame> framesInOut, bool* asKeyframe);

    /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @see okvis::ceres::ImuError::propagation()
   * @remark This method is threadsafe.
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS_propagated Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return True on success.
   */
    virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                             const ev::Parameters& Params,
                             okvis::kinematics::Transformation& T_WS_propagated,
                             okvis::SpeedAndBias & speedAndBiases,
                             const okvis::Time& t_start, const okvis::Time& t_end,
                             Eigen::Matrix<double, 15, 15>* covariance,
                             Eigen::Matrix<double, 15, 15>* jacobian) const;

    ///@}
    /// @name Getters related to the BRISK detector
    /// @{

    /// @brief Get the number of octaves of the BRISK detector.
    size_t getBriskDetectionOctaves() const {
        return briskDetectionOctaves_;
    }

    /// @brief Get the detection threshold of the BRISK detector.
    double getBriskDetectionThreshold() const {
        return briskDetectionThreshold_;
    }

    /// @brief Get the absolute threshold of the BRISK detector.
    double getBriskDetectionAbsoluteThreshold() const {
        return briskDetectionAbsoluteThreshold_;
    }

    /// @brief Get the maximum amount of keypoints of the BRISK detector.
    size_t getBriskDetectionMaximumKeypoints() const {
        return briskDetectionMaximumKeypoints_;
    }

    ///@}
    /// @name Getters related to the BRISK descriptor
    /// @{

    /// @brief Get the rotation invariance setting of the BRISK descriptor.
    bool getBriskDescriptionRotationInvariance() const {
        return briskDescriptionRotationInvariance_;
    }

    /// @brief Get the scale invariance setting of the BRISK descriptor.
    bool getBriskDescriptionScaleInvariance() const {
        return briskDescriptionScaleInvariance_;
    }

    ///@}
    /// @name Other getters
    /// @{

    /// @brief Get the matching threshold.
    double getBriskMatchingThreshold() const {
        return briskMatchingThreshold_;
    }

    /// @brief Get the area overlap threshold under which a new keyframe is inserted.
    float getKeyframeInsertionOverlapThershold() const {
        return keyframeInsertionOverlapThreshold_;
    }

    /// @brief Get the matching ratio threshold under which a new keyframe is inserted.
    float getKeyframeInsertionMatchingRatioThreshold() const {
        return keyframeInsertionMatchingRatioThreshold_;
    }

    /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
    bool isInitialized() {
        return isInitialized_;
    }

    /// @}
    /// @name Setters related to the BRISK detector
    /// @{

    /// @brief Set the number of octaves of the BRISK detector.
    void setBriskDetectionOctaves(size_t octaves) {
        briskDetectionOctaves_ = octaves;
        initialiseBriskFeatureDetectors();
    }

    /// @brief Set the detection threshold of the BRISK detector.
    void setBriskDetectionThreshold(double threshold) {
        briskDetectionThreshold_ = threshold;
        initialiseBriskFeatureDetectors();
    }

    /// @brief Set the absolute threshold of the BRISK detector.
    void setBriskDetectionAbsoluteThreshold(double threshold) {
        briskDetectionAbsoluteThreshold_ = threshold;
        initialiseBriskFeatureDetectors();
    }

    /// @brief Set the maximum number of keypoints of the BRISK detector.
    void setBriskDetectionMaximumKeypoints(size_t maxKeypoints) {
        briskDetectionMaximumKeypoints_ = maxKeypoints;
        initialiseBriskFeatureDetectors();
    }

    /// @}
    /// @name Setters related to the BRISK descriptor
    /// @{

    /// @brief Set the rotation invariance setting of the BRISK descriptor.
    void setBriskDescriptionRotationInvariance(bool invariance) {
        briskDescriptionRotationInvariance_ = invariance;
        initialiseBriskFeatureDetectors();
    }

    /// @brief Set the scale invariance setting of the BRISK descriptor.
    void setBriskDescriptionScaleInvariance(bool invariance) {
        briskDescriptionScaleInvariance_ = invariance;
        initialiseBriskFeatureDetectors();
    }

    ///@}
    /// @name Other setters
    /// @{

    /// @brief Set the matching threshold.
    void setBriskMatchingThreshold(double threshold) {
        briskMatchingThreshold_ = threshold;
    }

    /// @brief Set the area overlap threshold under which a new keyframe is inserted.
    void setKeyframeInsertionOverlapThreshold(float threshold) {
        keyframeInsertionOverlapThreshold_ = threshold;
    }

    /// @brief Set the matching ratio threshold under which a new keyframe is inserted.
    void setKeyframeInsertionMatchingRatioThreshold(float threshold) {
        keyframeInsertionMatchingRatioThreshold_ = threshold;
    }

    /// @}

private:

    /**
   * @brief   feature detectors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the detector.
   */
    cv::FeatureDetector featureDetector_;
    /**
   * @brief   feature descriptors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the descriptor.
   */
    cv::DescriptorExtractor descriptorExtractor_;
    /// Mutex for feature detectors and descriptors.
    std::mutex featureDetectorMutex_;

    bool isInitialized_;        ///< Is the pose initialised?
    //const size_t numCameras_;   ///< Number of cameras in the configuration.

    /// @name BRISK detection parameters
    /// @{

    size_t briskDetectionOctaves_;            ///< The set number of brisk octaves.
    double briskDetectionThreshold_;          ///< The set BRISK detection threshold.
    double briskDetectionAbsoluteThreshold_;  ///< The set BRISK absolute detection threshold.
    size_t briskDetectionMaximumKeypoints_;   ///< The set maximum number of keypoints.

    /// @}
    /// @name BRISK descriptor extractor parameters
    /// @{

    bool briskDescriptionRotationInvariance_; ///< The set rotation invariance setting.
    bool briskDescriptionScaleInvariance_;    ///< The set scale invariance setting.

    ///@}
    /// @name BRISK matching parameters
    ///@{

    double briskMatchingThreshold_; ///< The set BRISK matching threshold.

    ///@}

    std::unique_ptr<okvis::DenseMatcher> matcher_; ///< Matcher object.

    /**
   * @brief If the hull-area around all matched keypoints of the current frame (with existing landmarks)
   *        divided by the hull-area around all keypoints in the current frame is lower than
   *        this threshold it should be a new keyframe.
   * @see   doWeNeedANewKeyframe()
   */
    float keyframeInsertionOverlapThreshold_;  //0.6
    /**
   * @brief If the number of matched keypoints of the current frame with an older frame
   *        divided by the amount of points inside the convex hull around all keypoints
   *        is lower than the threshold it should be a keyframe.
   * @see   doWeNeedANewKeyframe()
   */
    float keyframeInsertionMatchingRatioThreshold_;  //0.2

    /**
   * @brief Decision whether a new frame should be keyframe or not.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
    bool needANewKeyframe(const okvis::Estimator& estimator,
                          std::shared_ptr<okvis::MultiFrame> currentFrame);  // based on some overlap area heuristics

    /**
   * @brief Match a new multiframe to existing keyframes
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe
   * @param      estimator              Estimator.
   * @param[in]  params                 Parameter struct.
   * @param[in]  currentFrameId         ID of the current frame that should be matched against keyframes.
   * @param[out] rotationOnly           Was the rotation only RANSAC motion model good enough to
   *                                    explain the motion between the new frame and the keyframes?
   * @param[in]  usePoseUncertainty     Use the pose uncertainty for the matching.
   * @param[out] uncertainMatchFraction Return the fraction of uncertain matches. Set to nullptr if not interested.
   * @param[in]  removeOutliers         Remove outliers during RANSAC.
   * @return The number of matches in total.
   */
    template<class MATCHING_ALGORITHM>
    int matchToKeyframes(okvis::Estimator& estimator,
                         const okvis::VioParameters& params,
                         const uint64_t currentFrameId, bool& rotationOnly,
                         bool usePoseUncertainty = true,
                         double* uncertainMatchFraction = 0,
                         bool removeOutliers = true);  // for wide-baseline matches (good initial guess)

    /**
   * @brief Match a new multiframe to the last frame.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator           Estimator.
   * @param params              Parameter struct.
   * @param currentFrameId      ID of the current frame that should be matched against the last one.
   * @param usePoseUncertainty  Use the pose uncertainty for the matching.
   * @param removeOutliers      Remove outliers during RANSAC.
   * @return The number of matches in total.
   */
    template<class MATCHING_ALGORITHM>
    int matchToLastFrame(okvis::Estimator& estimator,
                         const okvis::VioParameters& params,
                         const uint64_t currentFrameId,
                         bool usePoseUncertainty = true,
                         bool removeOutliers = true);

    /**
   * @brief Perform 3D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @param removeOutliers  Remove observation of outliers in estimator.
   * @return Number of inliers.
   */
    int runRansac3d2d(okvis::Estimator& estimator,
                      const okvis::cameras::NCameraSystem &nCameraSystem,
                      std::shared_ptr<okvis::MultiFrame> currentFrame,
                      bool removeOutliers);

    /**
   * @brief Perform 2D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator         Estimator.
   * @param params            Parameter struct.
   * @param currentFrameId    ID of the new multiframe containing matches with the frame with ID olderFrameId.
   * @param olderFrameId      ID of the multiframe to which the current frame has been matched against.
   * @param initializePose    If the pose has not yet been initialised should the function try to initialise it.
   * @param removeOutliers    Remove observation of outliers in estimator.
   * @param[out] rotationOnly Was the rotation only RANSAC model enough to explain the matches.
   * @return Number of inliers.
   */
    int runRansac2d2d(okvis::Estimator& estimator,
                      const okvis::VioParameters& params, uint64_t currentFrameId,
                      uint64_t olderFrameId, bool initializePose,
                      bool removeOutliers, bool &rotationOnly);

    /// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
    void initialiseBriskFeatureDetectors();


};

}  // namespace okvis

