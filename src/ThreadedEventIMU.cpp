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
    event_measurement.measurement.z = 1;
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
     std::default_random_engine gen;
std::uniform_real_distribution<double> dis(-0.1, 0.1);
    double w[] = {0, 2, 0};
    double t[] = {0, 1, 0};

    while (!allGroundtruthAdded_) {LOG(INFO) << "waiting";}
    ev::Pose estimatedPose;
    std::vector<std::vector<std::pair<double, double>>> estimatedPoses(3);

    Gnuplot gp;
    std::vector<std::vector<std::pair<double, double>>> groudtruth(3);

    for (auto it = maconMeasurements_.begin(); it != maconMeasurements_.end(); it++) {
        Eigen::Quaterniond q = it->measurement.q;
        double euler[3];
        ev::quat2eul(q, euler);
        for (int i = 0; i < 3; i++) {
            groudtruth[i].push_back(std::make_pair(it->timeStamp.toSec(), euler[i]));
        }
    }

    gp << "set xrange [15:16]\n";

//    gp << "plot" << gp.file1d(groudtruth[0]) << "with lines title 'roll',"
//                 << gp.file1d(groudtruth[1]) << "with lines title 'pitch',"
//                 << gp.file1d(groudtruth[2]) << "with lines title 'yaw'"
//                                    << std::endl;
    Contrast::param = parameters_;
    for (;;) {
        // get data and check for termination request
        if (eventMeasurementsReceived_.PopBlocking(&data) == false) {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);
            LOG(INFO) << "size " << eventMeasurementsReceived_.Size();
            return;
        }

        //        okvis::Time start;
        //        const okvis::Time* end;  // do not need to copy end timestamp
        {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);

            if (counter_s_ == 0) {
                eventFrames.push_back(std::make_shared<eventFrameMeasurement>());
            }

            for (auto it = eventFrames.begin(); it != eventFrames.end(); it++) {
                (*it)->events.push_back(data);
                (*it)->counter_w++;
            }

            auto em = eventFrames.front();

            if (em->counter_w == parameters_.window_size) {


//                int size = em->events.size();
//                for (int i = size / 10; i != size; i++) {
//                    em->events.pop_back();
//                }
                if (estimatedPose.q.norm() == 0) {
                    interpolateGroundtruth(estimatedPose, em->events.begin()->timeStamp);
                }
                double euler[3];
                ev::quat2eul(estimatedPose.q, euler);

                for (int i = 0; i < 3; i++) {
                    estimatedPoses[i].push_back(std::make_pair(em->events.begin()->timeStamp.toSec(), euler[i]));
                }

                gp << "plot '-' binary" << gp.binFmt1d(groudtruth[0], "record") << "with lines title 'roll',"
                   << "'-' binary" << gp.binFmt1d(groudtruth[1], "record") << "with lines title 'pitch',"
                   << "'-' binary" << gp.binFmt1d(groudtruth[2], "record") << "with lines title 'yaw',"
                   << "'-' binary" << gp.binFmt1d(estimatedPoses[0], "record") << "with lines title 'roll_',"
                   << "'-' binary" << gp.binFmt1d(estimatedPoses[1], "record") << "with lines title 'pitch_',"
                   << "'-' binary" << gp.binFmt1d(estimatedPoses[2], "record") << "with lines title 'yaw_'\n";

                gp.sendBinary1d(groudtruth[0]);
                gp.sendBinary1d(groudtruth[1]);
                gp.sendBinary1d(groudtruth[2]);
                gp.sendBinary1d(estimatedPoses[0]);
                gp.sendBinary1d(estimatedPoses[1]);
                gp.sendBinary1d(estimatedPoses[2]);
                gp.flush();

                undistortEvents(em);
                Contrast::em = em;

                Eigen::MatrixXd synthesizedFrame = Eigen::MatrixXd::Zero(parameters_.array_size_y, parameters_.array_size_x);
                Contrast::synthesizeEventFrame(synthesizedFrame, em);
                ev::imshowRescaled(synthesizedFrame, 1, "zero motion");

//                Eigen::MatrixXd image = Eigen::MatrixXd::Constant(parameters_.array_size_y, parameters_.array_size_x, 0.5);
//                Eigen::Matrix3d cameraMatrix_;
//                cv::cv2eigen(parameters_.cameraMatrix, cameraMatrix_);
//                for(auto it = em->events.begin(); it != em->events.end(); it++) {
//                    Eigen::Vector2d p(it->measurement.x, it->measurement.y);
//                    Eigen::Vector3d point_camera = cameraMatrix_ * p.homogeneous();

//                    if (point_camera(0) > 0 && point_camera(0) < 179
//                            && point_camera(1) > 0 && point_camera(1) < 239) {
//                        Contrast::fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
//                    }
//                    cv::Mat dst;
//                    cv::eigen2cv(image, dst);
//                    cv::imshow("animation", dst);
//                    cv::waitKey(3);
//                }

                bool positive = true;

//                synthesizedFrame = Eigen::MatrixXd::Zero(parameters_.array_size_y, parameters_.array_size_x);
//                Contrast::polarityEventFrame(synthesizedFrame, em, positive);
//                ev::imshowRescaled(synthesizedFrame, 1, "positive events");

//                synthesizedFrame = Eigen::MatrixXd::Zero(parameters_.array_size_y, parameters_.array_size_x);
//                Contrast::polarityEventFrame(synthesizedFrame, em, !positive);
//                ev::imshowRescaled(synthesizedFrame, 1, "negative events");

                // ground truth
                okvis::Time begin = em->events.front().timeStamp;
                okvis::Time end = em->events.back().timeStamp;

                // ???
                Eigen::Vector3d velocity = ((it_gt-1)->measurement.p - it_gt->measurement.p) / (end.toSec() - begin.toSec());

                Eigen::Quaterniond p1 = (it_gt-1)->measurement.q;
                Eigen::Quaterniond p2 = it_gt->measurement.q;


                // world transition
                Eigen::Quaterniond transition = p1 * p2.inverse();
                Eigen::AngleAxisd angleAxis = Eigen::AngleAxisd(transition);
                Eigen::Vector3d angularVelocity = angleAxis.axis() * angleAxis.angle()  / (end.toSec() - begin.toSec());

                LOG(INFO) << begin;
                LOG(INFO) << "events: " << em->events.size() << '\n';
                LOG(INFO) << end << '\n';

                LOG(INFO) << "ground truth:\n" << angularVelocity;

                Eigen::MatrixXd groundTruth = Eigen::MatrixXd::Zero(parameters_.array_size_y, parameters_.array_size_x);
                //Contrast::Intensity(groundTruth, em, Contrast::param, angularVelocity, velocity);
                Contrast::Intensity(groundTruth, em, Contrast::param, angularVelocity);

                double cost = 0;
                double mu = groundTruth.mean();
                for (int x_ = 0; x_ < 240; x_++) {
                    for (int y_ = 0; y_ < 180; y_++) {
                        cost += std::pow(groundTruth(y_, x_) - mu, 2);
                    }
                }
                cost /= (240*180);

                // adjust to ceres format
                cost = 1./std::pow(cost, 2);
                cost /= 2;

                std::string caption = "cost = " + std::to_string(cost);
                ev::imshowRescaled(groundTruth, 1, "ground truth", caption);

                processEventTimer.start();

                ceres::Solver::Options options;
                options.update_state_every_iteration = true;
                options.num_threads = 6;
                ceres::Solver::Summary summary;
                ev::imshowCallback callback(w);
                options.callbacks.push_back(&callback);
                options.minimizer_progress_to_stdout = true;

                ceres::Problem problem;
                ceres::CostFunction* cost_function = new ComputeVarianceFunction(em, parameters_);
                problem.AddResidualBlock(cost_function, NULL, w);

                ceres::Solve(options, &problem, &summary);

                LOG(INFO) << "Translation";
                //                    {
                //                        ceres::Problem problem;
                //                        ceres::CostFunction* cost_function =
                //                                new ceres::NumericDiffCostFunction<SE3, ceres::CENTRAL, 1, 1, 1, 1, 1, 1, 1>(
                //                                    new SE3());
                //                        problem.AddResidualBlock(cost_function, NULL, &t1, &t2, &t3, &w1, &w2, &w3);

                //                        ceres::Solve(options, &problem, &summary);
                //                    }


                //                    ceres::Problem problem_t;
                //                    ceres::CostFunction* cost_function_t =
                //                            new ceres::NumericDiffCostFunction<SE3, ceres::CENTRAL, 1, 1, 1, 1, 1, 1, 1>(
                //                                new SE3());
                //                    problem_t.AddResidualBlock(cost_function_t, NULL, &w1, &w2, &w3, &t1, &t2, &t3);
                //                    ceres::Solver::Options options_t;
                //                    options_t.update_state_every_iteration = true;
                //                    options_t.num_threads = 6;
                //                    ceres::Solver::Summary summary_t;
                //                    ceres::Solve(options_t, &problem_t, &summary_t);

                processEventTimer.stop();

                LOG(INFO) << okvis::timing::Timing::print();
                Eigen::Vector3d rotation_(w[0], w[1], w[2]);
                rotation_ *= ((end.toSec() - begin.toSec()));
                Eigen::AngleAxisd angleAxis_;
                if (rotation_.norm() == 0) {
                    angleAxis_ = Eigen::AngleAxisd(0, (Eigen::Vector3d() << 0, 0, 1).finished());
                } else {
                    angleAxis_ = Eigen::AngleAxisd(rotation_.norm(), rotation_.normalized());
                }
                estimatedPose.q = Eigen::Quaterniond(angleAxis_) * estimatedPose.q;
//                Eigen::AngleAxisd difference = Eigen::AngleAxisd(angleAxis_ * angleAxis.inverse());
//                double error = difference.angle() / (end.toSec() - begin.toSec());

                LOG(INFO) << "w :\n" << w[0]
                          << "\n" << w[1]
                          << "\n" << w[2];
//                LOG(INFO) << "error: " << error << " rad/s";
                LOG(INFO) << summary.BriefReport();

                eventFrames.pop_front();
            }
            counter_s_ = (counter_s_ + 1) % parameters_.step_size;
        }
        // LOG(INFO) << okvis::timing::Timing::print();
    }
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


// Start all threads.
void ThreadedEventIMU::startThreads() {
    imuConsumerThread_ = std::thread(&ThreadedEventIMU::imuConsumerLoop, this);
    eventConsumerThread_ = std::thread(&ThreadedEventIMU::eventConsumerLoop, this);
}

bool ThreadedEventIMU::undistortEvents(std::shared_ptr<eventFrameMeasurement>& em) {

    std::vector<cv::Point2d> inputDistortedPoints;
    std::vector<cv::Point2d> outputUndistortedPoints;
    for (auto it = em->events.begin(); it != em->events.end(); it++) {
        cv::Point2d point(it->measurement.y, it->measurement.x);
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
    return true;
}

bool ThreadedEventIMU::allGroundtruthAdded() {
    return allGroundtruthAdded_;
}

bool ThreadedEventIMU::interpolateGroundtruth(ev::Pose& pose, const okvis::Time& timeStamp) {
    int stepSize = maconMeasurements_.size();
    auto lo = maconMeasurements_.begin();
    auto hi = lo + stepSize - 1;
    if (timeStamp > hi->timeStamp || timeStamp < lo->timeStamp)
        return false;
    while (stepSize != 1) {
        stepSize /= 2;
        auto mid = lo + stepSize;
        if (timeStamp < mid->timeStamp) hi = mid;
        else if (timeStamp > mid->timeStamp) lo = mid;
        else {
            pose = mid->measurement;
            return true;
        }
    }
    if ((lo+1)->timeStamp < timeStamp) {
        lo++;
    } else if ((hi-1)->timeStamp > timeStamp) {
        hi--;
    }
    double dt = (timeStamp - lo->timeStamp).toSec() / (hi->timeStamp - lo->timeStamp).toSec();
    Eigen::Vector3d p = (hi->measurement.p - lo->measurement.p) * dt + lo->measurement.p;
    Eigen::Quaterniond q = lo->measurement.q.slerp(dt, hi->measurement.q);
    pose(p, q);
    return true;
}

}


