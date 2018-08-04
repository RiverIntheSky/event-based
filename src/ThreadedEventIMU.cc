#include "MapDrawer.h"
#include "ThreadedEventIMU.h"

namespace ev {
int count = 0;
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
    drawThread_.join();
}

void ThreadedEventIMU::init() {
    mpMap = new Map();
    mpTracker = new Tracking(parameters_.K, parameters_.distCoeffs, mpMap);
    mpMapDrawer = new MapDrawer(&parameters_, mpMap, mpTracker);
    Optimizer::mCameraProjectionMat(0, 0) = 200;
    Optimizer::mCameraProjectionMat(1, 1) = 200;
    Optimizer::mCameraProjectionMat(0, 2) = 150;
    Optimizer::mCameraProjectionMat(1, 2) = 150;
    startThreads();
}

bool ThreadedEventIMU::addEventMeasurement(okvis::Time& t, unsigned int x, unsigned int y, bool p) {

    if (x < 30 && y < 30) // distortion is wrong
        return false;

    ev::EventMeasurement event_measurement;
    event_measurement.measurement.x = x;
    event_measurement.measurement.y = y;
    event_measurement.measurement.z = 1;
    event_measurement.measurement.p = p;
    event_measurement.timeStamp = t;

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
    LOG(INFO) << "Event consumer loop joined";

    while (!allGroundtruthAdded_) {system("sleep 1");}

    count = 0;

    std::string files_path = parameters_.path + "/" + parameters_.experiment_name + "/" + std::to_string(parameters_.window_size) + "/";

    Eigen::Quaterniond R0 = maconMeasurements_.front().measurement.q.inverse();
    Eigen::Vector3d t0 = maconMeasurements_.front().measurement.p;

    for (;;) {

        // get data and check for termination request
        ev::EventMeasurement* data = new ev::EventMeasurement();
        if (eventMeasurementsReceived_.PopBlocking(data) == false) {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);
            LOG(INFO) << "size " << eventMeasurementsReceived_.Size();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);
            mCurrentFrame = mpTracker->getCurrentFrame();
            mCurrentFrame->vEvents.insert(data);

            if (mCurrentFrame->events() == parameters_.window_size) {
                double time = (*(mCurrentFrame->vEvents.rbegin()))->timeStamp.toSec();
                LOG(INFO) << "time stamp: "<< time;


                // feed in groundtruth
                okvis::Time begin =(*(mCurrentFrame->vEvents.begin()))->timeStamp;
                okvis::Time end = (*(mCurrentFrame->vEvents.rbegin()))->timeStamp;

                ev::Pose p1, p2;
                // of camera
                Eigen::Vector3d linear_velocity;
                Eigen::Quaterniond angular_velocity;
                if (interpolateGroundtruth(p1, begin) && interpolateGroundtruth(p2, end)) {
                    linear_velocity = (p1.q).inverse().toRotationMatrix() * (p2.p - p1.p) / (end.toSec() - begin.toSec());
                    angular_velocity =  (p1.q).inverse() * p2.q ;

                } else {
                    linear_velocity = Eigen::Vector3d(0, 0, 0);
                    angular_velocity = Eigen::Quaterniond(1, 0, 0, 0);
                }
                Eigen::AngleAxisd angleAxis = Eigen::AngleAxisd(angular_velocity);
                Eigen::Vector3d angularVelocity = angleAxis.axis() * angleAxis.angle() / (end.toSec() - begin.toSec());

                Eigen::Matrix3d R_ = (R0 * p1.q).toRotationMatrix();
                cv::Mat R = Converter::toCvMat(R_);
                cv::Mat R_w = rotm2axang(R);

                Eigen::Vector3d t = R0.toRotationMatrix() * (p1.p - t0);

//                mpTracker->Track(R, Converter::toCvMat(t), Converter::toCvMat(angularVelocity), Converter::toCvMat(linear_velocity));
                mpTracker->Track();

//                auto pMP = mpMap->getAllMapPoints().front();
//                imwriteRescaled(pMP->mFront, files_path + "map_" + std::to_string(count) + ".jpg", NULL);
//                cv::Mat blurred_map;
//                cv::GaussianBlur(pMP->mFront, blurred_map, cv::Size(0, 0), 1, 0);
//                imshowRescaled(blurred_map, 1, "map");

//                LOG(INFO) << "keyframes " << mpMap->getAllKeyFrames().size();
//                LOG(INFO) << "frames " << ++count;
//                LOG(INFO) << "normal " << pMP->getNormal();

                if (parameters_.write_to_file) {
                    std::ofstream  myfile_pr(files_path + "groundtruth_pose_rotation.txt", std::ios_base::app);
                    if (myfile_pr.is_open()) {
                        myfile_pr << begin.toSec() << " "
                               << R_w.at<float>(0) << " "
                               << R_w.at<float>(1) << " "
                               << R_w.at<float>(2) << "\n";
                        myfile_pr.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_pt(files_path + "groundtruth_pose_translation.txt", std::ios_base::app);
                    if (myfile_pt.is_open()) {
                        myfile_pt << begin.toSec() << " "
                               << t(0) << " "
                               << t(1) << " "
                               << t(2) << "\n";
                        myfile_pt.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile(files_path + "groundtruth_rotation.txt", std::ios_base::app);
                    if (myfile.is_open()) {
                        myfile << begin.toSec() << " "
                               << angularVelocity(0) << " "
                               << angularVelocity(1) << " "
                               << angularVelocity(2) << "\n";
                        myfile.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::string files_path = parameters_.path + "/" + parameters_.experiment_name + "/" + std::to_string(parameters_.window_size) + "/";

                    std::ofstream  myfilet(files_path + "groundtruth_translation.txt", std::ios_base::app);
                    if (myfilet.is_open()) {
                        myfilet << begin.toSec() << " "
                                << linear_velocity(0) << " "
                                << linear_velocity(1) << " "
                                << linear_velocity(2) << '\n';
                        myfilet.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_(files_path + "estimated_rotation.txt", std::ios_base::app);
                    if (myfile_.is_open()) {
                        myfile_ << begin.toSec() << " "
                                << (mpTracker->w).at<float>(0) << " "
                                << (mpTracker->w).at<float>(1) << " "
                                << (mpTracker->w).at<float>(2) << "\n";
                        myfile_.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_t(files_path + "estimated_translation.txt", std::ios_base::app);
                    if (myfile_t.is_open()) {
                        myfile_t<< begin.toSec() << " "
                                << (mpTracker->v).at<float>(0) << " "
                                << (mpTracker->v).at<float>(1) << " "
                                << (mpTracker->v).at<float>(2) << " ";
                        myfile_t << '\n';
                        myfile_t.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_r(files_path + "estimated_pose_rotation.txt", std::ios_base::app);
                    if (myfile_r.is_open()) {
                        myfile_r << begin.toSec() << " "
                                << (mpTracker->r).at<float>(0) << " "
                                << (mpTracker->r).at<float>(1) << " "
                                << (mpTracker->r).at<float>(2) << "\n";
                        myfile_r.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_tt(files_path + "estimated_pose_translation.txt", std::ios_base::app);
                    if (myfile_tt.is_open()) {
                        myfile_tt<< begin.toSec() << " "
                                << (mpTracker->t).at<float>(0) << " "
                                << (mpTracker->t).at<float>(1) << " "
                                << (mpTracker->t).at<float>(2) << " ";
                        myfile_tt << '\n';
                        myfile_tt.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;
                }
            }
        }
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
    drawThread_ = std::thread(&MapDrawer::drawMapPoints, mpMapDrawer);
}

// to be deleted !!
bool ThreadedEventIMU::undistortEvents(std::shared_ptr<eventFrameMeasurement>& em) {

    std::vector<cv::Point2d> inputDistortedPoints;
    std::vector<cv::Point2d> outputUndistortedPoints;
    for (auto it = em->events.begin(); it != em->events.end(); it++) {
        cv::Point2d point(it->measurement.x, it->measurement.y);
        inputDistortedPoints.push_back(point);
    }
    cv::undistortPoints(inputDistortedPoints, outputUndistortedPoints,
                        parameters_.K, parameters_.distCoeffs);
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
    if (timeStamp > hi->timeStamp) {
        Eigen::Vector3d p = hi->measurement.p;
        Eigen::Quaterniond q = hi->measurement.q;
        pose(p, q);
        return false;
    } else if (timeStamp < lo->timeStamp) {
        Eigen::Vector3d p = lo->measurement.p;
        Eigen::Quaterniond q = lo->measurement.q;
        pose(p, q);
        return false;
    }

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

double ThreadedEventIMU::contrastCost(Eigen::MatrixXd &image) {

    double cost = 0;
    for (int r = 0; r < 180; r++) {
        for (int c = 0; c < 240; c++) {
            cost += std::pow(image(r, c), 2);
        }
    }

    cost /= (240*180);

    return cost;

}

}
