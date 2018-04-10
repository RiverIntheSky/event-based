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

bool ThreadedEventIMU::addGroundtruth(const okvis::Time& t,
                                      const Eigen::Vector3d& position,
                                      const Eigen::Quaterniond& orientation) {
    ev::MaconMeasurement groundtruth;
    groundtruth.measurement.p = position;
    groundtruth.measurement.q = orientation;
    groundtruth.timeStamp = t;

        maconMeasurements_.push_back(groundtruth);
        it_gt = maconMeasurements_.begin();

    return true;
}

// Loop to process IMU measurements.
void ThreadedEventIMU::imuConsumerLoop() {
    LOG(INFO) << "I am imu consumer loop";

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

//    double w1 =  0.04111337760464654;
//    double w2 =  2.133983923557489;
//    double w3 =  -2.272747100334764;
    double w1 =  0;
    double w2 =  1;
    double w3 =  0;

    Contrast::param = parameters_;

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

            // every synthesized event frame starts with new groundtruth data,
            // ends with newly available groundtruth data

            if (data.timeStamp > it_gt->timeStamp){
                if (!eventFrames.empty()) {
                    auto em = eventFrames.front();
                    undistortEvents(em);
                    Contrast::em = em;
                    Eigen::MatrixXd synthesizedFrame = Eigen::MatrixXd::Zero(parameters_.array_size_x, parameters_.array_size_y);

                    // synthesized event frame with zero motion
                    Contrast::synthesizeEventFrame(synthesizedFrame, em);

                    ceres::Problem problem;

                    ceres::CostFunction* cost_function =
                            new ceres::NumericDiffCostFunction<Contrast, ceres::CENTRAL, 1, 1, 1, 1>(
                                new Contrast());
                    problem.AddResidualBlock(cost_function, NULL , &w1, &w2, &w3);


                    ceres::Solver::Options options;
                    ev::imshowCallback callback(w1, w2, w3);
                    options.callbacks.push_back(&callback);
    //                options.minimizer_progress_to_stdout = true;
                    options.update_state_every_iteration = true;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    std::cout << summary.BriefReport() << "\n";
                    std::cout << "w : " << w1
                              << " " << w2
                              << " " << w3
                              << "\n";

                    eventFrames.pop_front();

                    // ground truth rotation

                    Eigen::Quaterniond p1 = it_gt->measurement.q;
                    Eigen::Quaterniond p2 = (it_gt+1)->measurement.q;

                    okvis::Time begin = em->events.front().timeStamp;
                    okvis::Time end = em->events.back().timeStamp;

                    // world transition
                    Eigen::Quaterniond transition = p1 * p2.inverse();
                    Eigen::AngleAxisd angleAxis = Eigen::AngleAxisd(transition);
                    Eigen::Vector3d angularVelocity = angleAxis.axis() * angleAxis.angle()  / (end.toSec() - begin.toSec());

                    LOG(INFO) << begin;
                    LOG(INFO) << end;

                    Eigen::MatrixXd groundTruth = Eigen::MatrixXd::Zero(parameters_.array_size_x, parameters_.array_size_y);
                    Contrast::Intensity(groundTruth, em, Contrast::param, angularVelocity);

                    double cost = 0;
                    double mu = groundTruth.mean();
                    for (int x_ = 0; x_ < 240; x_++) {
                        for (int y_ = 0; y_ < 180; y_++) {
                            cost += std::pow(groundTruth(x_, y_) - mu, 2);
                        }
                    }
                    cost /= (240*180);
                    cost = std::sqrt(1./cost);

                    std::string caption = "cost = " + std::to_string(cost);
                    ev::imshowRescaled(groundTruth, 1, "ground truth", caption);


                }
                eventFrames.push_back(std::make_shared<eventFrameMeasurement>());
                it_gt++;
            }

            for (auto it = eventFrames.begin(); it != eventFrames.end(); it++) {
                (*it)->events.push_back(data);
            }
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

bool ThreadedEventIMU::undistortEvents(std::shared_ptr<eventFrameMeasurement>& em) {

    cv::Mat src(240, 180, CV_64F, cv::Scalar(0.0));
    cv::Mat dst(240, 180, CV_64F, cv::Scalar(0.0));
#if 0
    auto it = em->events.begin();
    for (; it != em->events.end(); it++) {
        src.at<double>(it->measurement.x, it->measurement.y) +=  (int(it->measurement.p) * 2 - 1);
    }
    cv::undistort(src, dst, parameters_.cameraMatrix, parameters_.distCoeffs);
    cv::Mat src_;
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    cv::subtract(src, cv::Mat(src.rows, src.cols, CV_64F, cv::Scalar(min)), src_);
    src_ /= (max - min);
    cv::imshow("distorted", src_);
    cv::waitKey(0);
    cv::Mat dst_;
    cv::minMaxLoc(dst, &min, &max);
    cv::subtract(dst, cv::Mat(src.rows, src.cols, CV_64F, cv::Scalar(min)), dst_);
    dst_ /= (max - min);
    cv::imshow("undistorted", dst_);
    cv::waitKey(0);
#else
    cv::Mat map_x, map_y;
    map_x.create(src.size(), CV_64F);
    map_y.create(src.size(), CV_64F);
    std::vector<cv::Point2d> inputDistortedPoints;
    std::vector<cv::Point2d> outputUndistortedPoints;
    for (auto it = em->events.begin(); it != em->events.end(); it++) {
        cv::Point2d point(it->measurement.x, it->measurement.y);
        inputDistortedPoints.push_back(point);
    }
    cv::undistortPoints(inputDistortedPoints, outputUndistortedPoints,
                        parameters_.cameraMatrix, parameters_.distCoeffs);
    auto it = em->events.begin();
    auto p_it = outputUndistortedPoints.begin();
    for (; it != em->events.end(); it++, p_it++) {
        it->measurement.x = p_it->x;
        it->measurement.y = p_it->y;
    }
    //    cv::Mat_<double> I = cv::Mat_<double>::eye(3,3);
    //    cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    //    cv::initUndistortRectifyMap(parameters_.cameraMatrix, parameters_.distCoeffs,
    //                                I, parameters_.cameraMatrix, cv::Size(src.cols, src.rows),
    //                                map_x.type(), map_x, map_y);
#endif
    return true;
}


// Start all threads.
void ThreadedEventIMU::startThreads() {
    imuConsumerThread_ = std::thread(&ThreadedEventIMU::imuConsumerLoop, this);
    eventConsumerThread_ = std::thread(&ThreadedEventIMU::eventConsumerLoop, this);
}


}
