#ifndef INCLUDE_EV_THREADEDEVENTIMU_H
#define INCLUDE_EV_THREADEDEVENTIMU_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <okvis/ImuFrameSynchronizer.hpp>
#include <okvis/FrameSynchronizer.hpp>

#include "Frontend.h"
#include "ceres/ceres.h"

namespace ev {

class ThreadedEventIMU: public okvis::VioInterface
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef okvis::timing::Timer TimerSwitchable;

//    // DAVIS
//    typedef Eigen::Matrix<double, 2, 2> eventFrame;




//    bool correctEvent(cv::Mat& frame, eventFrameMeasurement* em);
    bool undistortEvents(std::shared_ptr<eventFrameMeasurement>& em);

    //void fuse(cv::Mat& image, cv::Vec2d point, bool polarity);

    ThreadedEventIMU(Parameters& parameters);

    // ??
    virtual ~ThreadedEventIMU();

    /// \name Add measurements to the algorithm
    /// \{
    /**
     * \brief              Add a new image.
     * \warning Not implemented.
     * \param stamp        The image timestamp.
     * \param cameraIndex  The index of the camera that the image originates from.
     * \param image        The image.
     * \param keypoints    Optionally aready pass keypoints. This will skip the detection part.
     * \param asKeyframe   Use the new image as keyframe. Not implemented.
     * \warning The frame consumer loop does not support using existing keypoints yet.
     * \warning Already specifying whether this frame should be a keyframe is not implemented yet.
     * \return             Returns true normally. False, if the previous one has not been processed yet.
     */
    virtual bool addImage(const okvis::Time & stamp, size_t cameraIndex,
                          const cv::Mat & image,
                          const std::vector<cv::KeyPoint> * keypoints = 0,
                          bool* asKeyframe = 0){


        cv::imshow("image", image/256);
        cv::waitKey(1);
        return true;
    }

    /**
     * \brief             Add an abstracted image observation.
     * \warning Not implemented.
     * \param stamp       The timestamp for the start of integration time for the image.
     * \param cameraIndex The index of the camera.
     * \param keypoints   A vector where each entry represents a [u,v] keypoint measurement. Also set the size field.
     * \param landmarkIds A vector of landmark ids for each keypoint measurement.
     * \param descriptors A matrix containing the descriptors for each keypoint.
     * \param asKeyframe  Optionally force keyframe or not.
     * \return            Returns true normally. False, if the previous one has not been processed yet.
     */
    virtual bool addKeypoints(const okvis::Time & stamp, size_t cameraIndex,
                              const std::vector<cv::KeyPoint> & keypoints,
                              const std::vector<uint64_t> & landmarkIds,
                              const cv::Mat& descriptors = cv::Mat(),
                              bool* asKeyframe = 0) {
        return false;
    }

    /**
     * \brief                      Add a position measurement.
     * \warning Not implemented.
     * \param stamp                The measurement timestamp.
     * \param position             The position in world frame.
     * \param positionOffset       Body frame antenna position offset [m].
     * \param positionCovariance   The position measurement covariance matrix.
     */
    virtual void addPositionMeasurement(
        const okvis::Time & stamp, const Eigen::Vector3d & position,
        const Eigen::Vector3d & positionOffset,
            const Eigen::Matrix3d & positionCovariance) {}

    /**
     * \brief                       Add a GPS measurement.
     * \warning Not implemented.
     * \param stamp                 The measurement timestamp.
     * \param lat_wgs84_deg         WGS84 latitude [deg].
     * \param lon_wgs84_deg         WGS84 longitude [deg].
     * \param alt_wgs84_deg         WGS84 altitude [m].
     * \param positionOffset        Body frame antenna position offset [m].
     * \param positionCovarianceENU The position measurement covariance matrix.
     */
    virtual void addGpsMeasurement(const okvis::Time & stamp,
                                   double lat_wgs84_deg, double lon_wgs84_deg,
                                   double alt_wgs84_deg,
                                   const Eigen::Vector3d & positionOffset,
                                   const Eigen::Matrix3d & positionCovarianceENU) {}

    /**
     * \brief                      Add a magnetometer measurement.
     * \warning Not implemented.
     * \param stamp                The measurement timestamp.
     * \param fluxDensityMeas      Measured magnetic flux density (sensor frame) [uT].
     * \param stdev                Measurement std deviation [uT].
     */
    virtual void addMagnetometerMeasurement(
        const okvis::Time & stamp, const Eigen::Vector3d & fluxDensityMeas,
            double stdev) {}

    /**
     * \brief                      Add a static pressure measurement.
     * \warning Not implemented.
     * \param stamp                The measurement timestamp.
     * \param staticPressure       Measured static pressure [Pa].
     * \param stdev                Measurement std deviation [Pa].
     */
    virtual void addBarometerMeasurement(const okvis::Time & stamp,
                                         double staticPressure, double stdev) {}

    /**
     * \brief                      Add a differential pressure measurement.
     * \warning Not implemented.
     * \param stamp                The measurement timestamp.
     * \param differentialPressure Measured differential pressure [Pa].
     * \param stdev                Measurement std deviation [Pa].
     */
    virtual void addDifferentialPressureMeasurement(const okvis::Time & stamp,
                                                    double differentialPressure,
                                                    double stdev) {}

    /**
     * \brief          Add an IMU measurement.
     * \param stamp    The measurement timestamp.
     * \param alpha    The acceleration measured at this time.
     * \param omega    The angular velocity measured at this time.
     * \return Returns true normally. False if the previous one has not been processed yet.
     */
    virtual bool addImuMeasurement(const okvis::Time& stamp,
                            const Eigen::Vector3d& alpha,
                            const Eigen::Vector3d& omega);

    virtual bool addEventMeasurement(okvis::Time& t, unsigned int x, unsigned int y, bool p);

    bool addGroundtruth(const okvis::Time& t,
                        const Eigen::Vector3d& position,
                        const Eigen::Quaterniond& orientation);


private:
    /// \}
    /// \name Setters
    /// \{

    /**
     * \brief Set the blocking variable that indicates whether the addMeasurement() functions
     *        should return immediately (blocking=false), or only when the processing is complete.
     */
    virtual void setBlocking(bool blocking);

    /// \brief Start all threads.
    virtual void startThreads();
    /// \brief Initialises settings and calls startThreads().
    void init();

    /// \brief Loop to process IMU measurements.
    void imuConsumerLoop();
    /// \brief Loop to process event measurements.
    void eventConsumerLoop();

    /// \brief Loop that performs the optimization and marginalisation.
    void optimizationLoop();

    /**
     * @brief Get a subset of the recorded IMU measurements.
     * @param start The first IMU measurement in the return value will be older than this timestamp.
     * @param end The last IMU measurement in the return value will be newer than this timestamp.
     * @remark This function is threadsafe.
     * @return The IMU Measurement spanning at least the time between start and end.
     */
    okvis::ImuMeasurementDeque getImuMeasurments(okvis::Time& start,
                                                 okvis::Time& end);

    /**
     * @brief Remove IMU measurements from the internal buffer.
     * @param eraseUntil Remove all measurements that are strictly older than this time.
     * @return The number of IMU measurements that have been removed
     */
    int deleteImuMeasurements(const okvis::Time& eraseUntil);

    /**
     * @brief Get a subset of the recorded event measurements.
     * @param
     * @remark This function is threadsafe.
     * @return The events spanning at least the time between start and end.
     */
    ev::EventMeasurementDeque getEventMeasurments(okvis::Time& start,
                                                 okvis::Time& end);

    /**
     * @brief Remove event measurements from the internal buffer.
     * @param eraseUntil Remove all measurements that are strictly older than this time.
     * @return The number of events that have been removed
     */
    int deleteEventMeasurements(const okvis::Time& eraseUntil);

     /// @name Measurement input queues
     /// @{

     /// IMU measurement input queue.
     okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement> imuMeasurementsReceived_;

     /// Events input queue.
     okvis::threadsafe::ThreadSafeQueue<ev::EventMeasurement> eventMeasurementsReceived_;

     /// ground truth queue. added first, no need to be thread-safe
     std::vector<ev::MaconMeasurement> maconMeasurements_;
     std::vector<ev::MaconMeasurement>::iterator it_gt;


     /// @brief This struct contains the results of the optimization for ease of publication.
     ///        It is also used for publishing poses that have been propagated with the IMU
     ///        measurements.
     struct OptimizationResults {
           EIGEN_MAKE_ALIGNED_OPERATOR_NEW
       okvis::Time stamp;                          ///< Timestamp of the optimized/propagated pose.
       okvis::kinematics::Transformation T_WS;     ///< The pose.
       okvis::SpeedAndBias speedAndBiases;         ///< The speeds and biases.
       Eigen::Matrix<double, 3, 1> omega_S;        ///< The rotational speed of the sensor.
       /// The relative transformation of the cameras to the sensor (IMU) frame
       okvis::kinematics::Transformation T_SC;
       okvis::MapPointVector landmarksVector;      ///< Vector containing the current landmarks.
       okvis::MapPointVector transferredLandmarks; ///< Vector of the landmarks that have been marginalized out.
       bool onlyPublishLandmarks;                  ///< Boolean to signalise the publisherLoop() that only the landmarks should be published
     };

     /// @}
     /// @name State variables
     /// @{

     okvis::SpeedAndBias speedAndBiases_propagated_;     ///< The speeds and IMU biases propagated by the IMU measurements.

     /// \brief The IMU parameters.
     ev::Parameters parameters_;

     okvis::kinematics::Transformation T_WS_propagated_; ///< The pose propagated by the IMU measurements
     std::shared_ptr<okvis::MapPointVector> map_;        ///< The map. Unused.

     // lock lastState_mutex_ when accessing these
     /// \brief Resulting pose of the last optimization
     /// \warning Lock lastState_mutex_.
     okvis::kinematics::Transformation lastOptimized_T_WS_;
     /// \brief Resulting speeds and IMU biases after last optimization.
     /// \warning Lock lastState_mutex_.
     okvis::SpeedAndBias lastOptimizedSpeedAndBiases_;
     /// \brief Timestamp of newest frame used in the last optimization.
     /// \warning Lock lastState_mutex_.
     okvis::Time lastOptimizedStateTimestamp_;
     /// This is set to true after optimization to signal the IMU consumer loop to repropagate
     /// the state from the lastOptimizedStateTimestamp_.
     std::atomic_bool repropagationNeeded_;

     okvis::ImuFrameSynchronizer imuFrameSynchronizer_;  ///< The IMU frame synchronizer.
     /// \brief The frame synchronizer responsible for merging frames into multiframes
     /// \warning Lock with frameSynchronizer_mutex_
//     okvis::FrameSynchronizer frameSynchronizer_;

     okvis::Time lastAddedStateTimestamp_; ///< Timestamp of the newest state in the Estimator.
     okvis::Time lastAddedImageTimestamp_; ///< Timestamp of the newest image added to the image input queue.

     /// @}
     /// @name Algorithm objects.
     /// @{
     okvis::Estimator estimator_;    ///< The backend estimator.
     ev::Frontend frontend_;      ///< The frontend.

     /// @}
     /// @name Measurement operation queues.
     /// @{
     /// \brief The IMU measurements.
     /// \warning Lock with imuMeasurements_mutex_.
     okvis::ImuMeasurementDeque imuMeasurements_;
     /// \brief The event measurements.
     /// \warning Lock with eventMeasurements_mutex_.
     ev::EventMeasurementDeque eventMeasurements_;
     /// The queue containing the results of the optimization or IMU propagation ready for publishing.
     okvis::threadsafe::ThreadSafeQueue<OptimizationResults> optimizationResults_;
     /// counter for step size
     unsigned counter_s_{0};

     /// @}
     /// @name Mutexes
     /// @{

     std::mutex imuMeasurements_mutex_;      ///< Lock when accessing imuMeasurements_
     std::mutex eventMeasurements_mutex_;      ///< Lock when accessing eventMeasurements_
     std::mutex estimator_mutex_;            ///< Lock when accessing the estimator_.
     /// Boolean flag for whether optimization is done for the last state that has been added to the estimator.
     std::atomic_bool optimizationDone_;
     std::mutex lastState_mutex_;            ///< Lock when accessing any of the 'lastOptimized*' variables.

     /// @}
     /// @name Consumer threads
     /// @{
     std::thread imuConsumerThread_;           ///< Thread running imuConsumerLoop().
     std::thread eventConsumerThread_;           ///< Thread running eventConsumerLoop().

     /// @}

     /// The maximum input queue size before IMU measurements are dropped.
     const size_t maxImuInputQueueSize_;

     /// The maximum input queue size before events are dropped.
     const size_t maxEventInputQueueSize_;


};
}

#endif // THREADEDEVENTIMU_H
