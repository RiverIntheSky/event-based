#ifndef INCLUDE_EV_THREADEDEVENTIMU_H
#define INCLUDE_EV_THREADEDEVENTIMU_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iomanip>
#include <omp.h>

#include <okvis/Measurements.h>
#include <okvis/ThreadsafeQueue.h>

#include "Tracking.h"

namespace ev {
class ThreadedEventIMU
{
    /**
     *  \brief
     *  This class manages the complete data flow in and out of the algorithm, as well as between the
     *  processing threads. It starts processing as soon as a certain amount of data is received. This
     *  class was designed for sensor fusion, however, for this project only the event measurements
     *  are used.
     */
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ThreadedEventIMU(Parameters& parameters);

    ~ThreadedEventIMU();

    /**
     * \brief          Add an event measurement.
     * \param t        The measurement timestamp.
     * \param x        The x coordinate.
     * \param y        The y coordinate.
     * \param p        The event polarity.
     * \return Returns true normally. This function had a test to drop events at the upper left corner,
     *         where the undistortion is not reliable, but it reduces information in the scene and was
     *         not working really good.
     */
    bool addEventMeasurement(okvis::Time& t, unsigned int x, unsigned int y, bool p);

    /**
     * \brief                Read groundtruth pose.
     * \param t              The groundtruth timestamp.
     * \param positon        The groundtruth position of the camera.
     * \param orientation    The groundtruth orientation of the camera.
     * \return Returns true normally.
     */
    bool addGroundtruth(const okvis::Time& t,
                        const Eigen::Vector3d& position,
                        const Eigen::Quaterniond& orientation);

    /**
     * \brief   Interpolate the groundtruth poses at the queried timestamp. Is used for
     *          comparison between groundtruth and estimation.
     * \param timeStamp      The timestamp at which the groundtruth pose is queried.
     * \param pose           The interpolated pose.
     * \return Returns true when timeStamp is in the valid range.
     */
    bool interpolateGroundtruth(Pose &pose, const okvis::Time& timeStamp);

private:

    /// \brief Start all threads.
    void startThreads();

    /// \brief Initialises settings and calls startThreads().
    void init();

    /// \brief Loop to process event measurements.
    void eventConsumerLoop();

    /// \brief Helper function for outputing motion parameters to file
    bool writeToFile(okvis::Time& t, Eigen::Vector3d& vec, std::string file_name);
    bool writeToFile(okvis::Time& t, cv::Mat& vec, std::string file_name);

     /// Events input queue.
     okvis::threadsafe::ThreadSafeQueue<ev::EventMeasurement> eventMeasurementsReceived_;

     /// ground truth queue. added first, no need to be thread-safe
     std::vector<ev::MaconMeasurement, Eigen::aligned_allocator<ev::MaconMeasurement>> maconMeasurements_;

     /// \brief The parameters.
     ev::Parameters parameters_;

     /// \brief The event measurements.
     /// \warning Lock with eventMeasurements_mutex_.
     ev::EventMeasurementDeque eventMeasurements_;

     std::mutex eventMeasurements_mutex_;      ///< Lock when accessing eventMeasurements_

     std::thread eventConsumerThread_;           ///< Thread running eventConsumerLoop().

     /// The maximum input queue size before events are dropped.
     const size_t maxEventInputQueueSize_;

     /// Tracker. It receives a frame and computes the associated camera pose.
     /// It also decides when to insert a new keyframe, create some new MapPoints
     Tracking* mpTracker;

     /// world map
     Map* mpMap;

     /// current frame
     shared_ptr<Frame> mCurrentFrame;
public:
     /// If all the ground truth data is read
     bool allGroundtruthAdded_ = false;
};
}

#endif // THREADEDEVENTIMU_H
