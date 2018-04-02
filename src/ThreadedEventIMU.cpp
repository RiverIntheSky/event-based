#include "ThreadedEventIMU.h"

namespace ev {
ThreadedEventIMU::ThreadedEventIMU(Parameters &parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      parameters_(parameters),
      repropagationNeeded_(false),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(false),
      maxImuInputQueueSize_(10000),
      maxEventInputQueueSize_(3000000) {
    setBlocking(true);
    init();
}

ThreadedEventIMU::~ThreadedEventIMU() {
    imuMeasurementsReceived_.Shutdown();
    eventMeasurementsReceived_.Shutdown();

    // consumer threads
    imuConsumerThread_.join();
    eventConsumerThread_.join();

}

void ThreadedEventIMU::init() {
    startThreads();
}

bool ThreadedEventIMU::addEventMeasurement(okvis::Time& t, unsigned int x, unsigned int y, bool p) {

    ev::EventMeasurement event_measurement;
    event_measurement.measurement.x = x;
    event_measurement.measurement.y = y;
    event_measurement.measurement.p = p;
    event_measurement.timeStamp = t;

    // std::cout << event.t << " " << event.x << " " << event.y << " " << event.p << std::endl;

    if (blocking_) {
        eventMeasurementsReceived_.PushBlockingIfFull(event_measurement, 1);
        return true;
    } else {
        eventMeasurementsReceived_.PushNonBlockingDroppingIfFull(
                    event_measurement, maxEventInputQueueSize_);
        return eventMeasurementsReceived_.Size() == 1;
    }

    return true;
}

bool ThreadedEventIMU::addImuMeasurement(const okvis::Time& stamp,
                                         const Eigen::Vector3d& alpha,
                                         const Eigen::Vector3d& omega) {

    okvis::ImuMeasurement imu_measurement;
    imu_measurement.measurement.accelerometers = alpha;
    imu_measurement.measurement.gyroscopes = omega;
    imu_measurement.timeStamp = stamp;

    if (blocking_) {
        imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
        return true;
    } else {
        imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
                    imu_measurement, maxImuInputQueueSize_);
        return imuMeasurementsReceived_.Size() == 1;
    }
    return true;
}

// Loop to process IMU measurements.
void ThreadedEventIMU::imuConsumerLoop() {
    LOG(INFO) << "I am imu consumer loop";
    LOG(INFO) << imuMeasurementsReceived_.Size();

    okvis::ImuMeasurement data;
    //    TimerSwitchable processImuTimer("0 processImuMeasurements",true);
    for (;;) {
        // get data and check for termination request
        if (imuMeasurementsReceived_.PopBlocking(&data) == false)
            return;
        //        processImuTimer.start();
        okvis::Time start;
        const okvis::Time* end;  // do not need to copy end timestamp
        {
            std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
            OKVIS_ASSERT_TRUE(Exception,
                              imuMeasurements_.empty()
                              || imuMeasurements_.back().timeStamp < data.timeStamp,
                              "IMU measurement from the past received");

            if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
                start = imuMeasurements_.back().timeStamp;
            } else if (repropagationNeeded_) {
                std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
                start = lastOptimizedStateTimestamp_;
                T_WS_propagated_ = lastOptimized_T_WS_;
                speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
                repropagationNeeded_ = false;
            } else
                start = okvis::Time(0, 0);
            end = &data.timeStamp;
        }
        imuMeasurements_.push_back(data);

        // notify other threads that imu data with timeStamp is here.
        imuFrameSynchronizer_.gotImuData(data.timeStamp);

        Eigen::Matrix<double, 15, 15> covariance;
        Eigen::Matrix<double, 15, 15> jacobian;

        frontend_.propagation(imuMeasurements_, parameters_, T_WS_propagated_,
                              speedAndBiases_propagated_, start, *end, &covariance,
                              &jacobian);
        OptimizationResults result;
        result.stamp = *end;
        result.T_WS = T_WS_propagated_;
        result.speedAndBiases = speedAndBiases_propagated_;
        result.omega_S = imuMeasurements_.back().measurement.gyroscopes
                - speedAndBiases_propagated_.segment<3>(3);

        //result.T_SC = *parameters_.T_SC;

        result.onlyPublishLandmarks = false;
        optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
    }
    //    processImuTimer.stop();

}

// Loop to process event measurements.
void ThreadedEventIMU::eventConsumerLoop() {
    LOG(INFO) << "I am event consumer loop";

    ev::EventMeasurement data;
    // ??
    TimerSwitchable processEventTimer("0 processEventMeasurements",true);

    std::deque<std::shared_ptr<eventFrameMeasurement>> eventFrames;

    double w1 = 0.;
    double w2 = 0.;
    double w3 = 0.;

    for (;;) {
        // get data and check for termination request
        if (eventMeasurementsReceived_.PopBlocking(&data) == false) {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);
            LOG(INFO) << "size " << eventMeasurementsReceived_.Size();
            return;
        }

        processEventTimer.start();
        //        okvis::Time start;
        //        const okvis::Time* end;  // do not need to copy end timestamp
        {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);

            if (counter_s_ == 0) {
                eventFrames.push_back(std::make_shared<eventFrameMeasurement>());
            }
            for (auto it = eventFrames.begin(); it != eventFrames.end(); it++) {
                // auto singleEvent = correctEvent(data);
                // (*it)->eventframe.timeStamp not set
                // (*it)->eventframe.measurement += singleEvent;
                (*it)->events.push_back(data);
                (*it)->counter_w++;
            }
            auto em = eventFrames.front();
            if (em->counter_w == parameters_.window_size) {
                //cv::meanStdDev(InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray())
                //                addImage(em->eventframe.timeStamp, em->eventframe.sensorId, em->eventframe.measurement);
                //cv::Mat synthesizedFrame(parameters_.array_size_x, parameters_.array_size_y, CV_64F, cv::Scalar(0.0));
                Eigen::MatrixXd synthesizedFrame = Eigen::MatrixXd::Zero(parameters_.array_size_x, parameters_.array_size_y);
                undistortEvents(em);
                Contrast::em = em;
                Contrast::param = parameters_;
                Contrast::synthesizeEventFrame(synthesizedFrame, em);

                Contrast::I_mu = synthesizedFrame.mean();
                LOG(INFO) << synthesizedFrame.mean();
                ceres::Problem problem;

                        ceres::CostFunction* cost_function =
                          new ceres::NumericDiffCostFunction<Contrast, ceres::CENTRAL, 1, 1, 1, 1>(
                            new Contrast());
//                        ceres::CostFunction* cost_function =
//                             new ceres::AutoDiffCostFunction<Contrast, 1, 1, 1>(
//                                 new Contrast(col, row));
                        problem.AddResidualBlock(cost_function, NULL , &w1, &w2, &w3);


                  ceres::Solver::Options options;
                  options.minimizer_progress_to_stdout = true;
                  ceres::Solver::Summary summary;
                  ceres::Solve(options, &problem, &summary);
                  std::cout << summary.BriefReport() << "\n";
                  std::cout << "w : " << w1
                            << " " << w2
                            << " " << w3
                            << "\n";
//                cv::Mat ef(parameters_.array_size_x, parameters_.array_size_y, CV_64F, cv::Scalar(0.0));
//                cv::eigen2cv(synthesizedFrame, ef);
//                std::ofstream file("/home/weizhen/ceres.txt");
//                if (file.is_open()) {
//                    for (int c = 0; c != 240; c++) {
//                        for (int l = 0; l != 180; l++) {
//                            file << synthesizedFrame(c, l) << '\n';
//                        }
//                    }
//                }
//                file.close();
                // addImage(em->events.front().timeStamp, 0, ef);
                eventFrames.pop_front();

            }
            counter_s_ = (counter_s_ + 1) % parameters_.step_size;
        }
        processEventTimer.stop();
        // LOG(INFO) << okvis::timing::Timing::print();
    }

    //LOG(INFO) << counter_s_ << " at time " << data.timeStamp;


}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedEventIMU::setBlocking(bool blocking) {
    blocking_ = blocking;
    // disable time limit for optimization
    if(blocking_) {
        std::lock_guard<std::mutex> lock(estimator_mutex_);
        // estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
    }
}

//bool ThreadedEventIMU::correctEvent(cv::Mat& frame, eventFrameMeasurement *em) {


//}

// bool ThreadedEventIMU::correctEvent(Eigen::MatrixXd& frame, std::shared_ptr<eventFrameMeasurement> em) {
bool ThreadedEventIMU::undistortEvents(std::shared_ptr<eventFrameMeasurement>& em) {
    auto it = em->events.begin();
#if 0
    for (; it != em->events.end(); it++) {
        frame(it->measurement.x, it->measurement.y) += (int(it->measurement.p) * 2 - 1);
    }
#else
    for (; it != em->events.end(); it++) {
//        LOG(INFO) << "Before (" << it->measurement.x << ", " << it->measurement.y << ")";
        cv::Vec2d p((it->measurement.x-parameters_.cameraMatrix.at<double>(0,2))/parameters_.cameraMatrix.at<double>(0,0),
                    (it->measurement.y-parameters_.cameraMatrix.at<double>(1,2))/parameters_.cameraMatrix.at<double>(1,1));
//        cv::Vec2d p(it->measurement.x, it->measurement.y);
        cv::Vec2d p_undistorted;
#if 1
        cv::undistort(p, p_undistorted, cv::Mat::eye(3, 3, CV_64F), parameters_.distCoeffs);
#else
        p_undistorted = p;
#endif
        it->measurement.x = p_undistorted[0];
        it->measurement.y = p_undistorted[1];
//        cv::Vec3d p_undistorted_(p_undistorted[0], p_undistorted[1], 1);
//        cv::Mat p_camera_ = parameters_.cameraMatrix * cv::Mat(p_undistorted_);



//        LOG(INFO) << "After (" << p_camera_.at<double>(0,0) << ", " <<p_camera_.at<double>(1,0) << ")";
//        LOG(INFO);
    }
#endif
    return true;
}



//cv::Mat ThreadedEventIMU::correctEvent(const EventMeasurement& event_measurement) {
//    // cv::Matx<double, x, y>;
//    cv::Mat ef(parameters_.array_size_x, parameters_.array_size_y, CV_64F, cv::Scalar(0.0));
//    // do not distinguish between positive and negative events
//    ef.at<double>(event_measurement.measurement.x, event_measurement.measurement.y) += 1;
//    return ef;
//}


// Start all threads.
void ThreadedEventIMU::startThreads() {
    imuConsumerThread_ = std::thread(&ThreadedEventIMU::imuConsumerLoop, this);
    eventConsumerThread_ = std::thread(&ThreadedEventIMU::eventConsumerLoop, this);
}


}
