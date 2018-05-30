
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
    std::cout.setf(std::ios::left);
    std::cout.fill(' ');
    std::cerr<< std::setw(31) << "e";
    LOG(INFO) << "I am event consumer loop";

    ev::EventMeasurement data;
    // ??
    TimerSwitchable processEventTimer("0 processEventMeasurements",true);
    std::deque<std::shared_ptr<eventFrameMeasurement>> eventFrames;
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dis(-0.1, 0.1);
    double w[] = {0.0, 0.0, 0.0};
    double v[] = {0, 0, 0};
    double z[parameters_.patch_num] = {};
    LOG(INFO)<<parameters_.patch_num;
    std::vector<double*> params;
    params.push_back(w);
    params.push_back(v);
    params.push_back(z);

    while (!allGroundtruthAdded_) {}
    ev::Pose estimatedPose;
    std::vector<std::vector<std::pair<double, double>>> estimatedPoses(3);

    // Gnuplot gp;
    std::vector<std::vector<std::pair<double, double>>> groundtruth(3);

    //    for (auto it = maconMeasurements_.begin(); it != maconMeasurements_.end(); it++) {
    //        Eigen::Quaterniond q = it->measurement.q;
    //        double euler[3];
    //        ev::quat2eul(q, euler);
    //        for (int i = 0; i < 3; i++) {
    //            groundtruth[i].push_back(std::make_pair(it->timeStamp.toSec(), euler[i]));
    //        }
    //    }

    // gp << "set xrange [0:20]\n";

    //    gp << "plot" << gp.file1d(groundtruth[0]) << "with lines title 'roll',"
    //                 << gp.file1d(groundtruth[1]) << "with lines title 'pitch',"
    //                 << gp.file1d(groundtruth[2]) << "with lines title 'yaw'"
    //                                    << std::endl;
    Contrast::param_ = parameters_;
    count = 0;
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
                for (int i = 0; i < parameters_.patch_num; i++) {
                    z[i] = 1;
                }



                //                if (estimatedPose.q.norm() == 0) {
                //                    interpolateGroundtruth(estimatedPose, em->events.begin()->timeStamp);
                //                }
                //                double euler[3];
                //                ev::quat2eul(estimatedPose.q, euler);

                //                for (int i = 0; i < 3; i++) {
                //                    estimatedPoses[i].push_back(std::make_pair(em->events.begin()->timeStamp.toSec(), euler[i]));
                //                }


                undistortEvents(em);
                int patch_num[12] = {};
                // helper function, truncates value outside a range
                auto truncate = [](int value, int min_value, int max_value) -> int {
                    return std::min(std::max(value, min_value), max_value);
                };

                // divide scene into patches
                auto& param = parameters_;
                Eigen::Matrix3d cameraMatrix_;
                cv::cv2eigen(param.cameraMatrix, cameraMatrix_);
                auto patch = [&param, cameraMatrix_, truncate](Eigen::Vector3d p)  -> int {
                    Eigen::Vector3d p_ = cameraMatrix_ * p;
                    int patch_num_x = std::ceil(param.array_size_x / param.patch_width);
                    int patch_num_y = std::ceil(param.array_size_y / param.patch_width);
                    return truncate(std::floor(p_(0)/param.patch_width), 0, patch_num_y-1) * patch_num_y
                            + truncate(std::floor(p_(1)/param.patch_width), 0, patch_num_x-1);
                };
                for (auto it = em->events.begin(); it != em->events.end(); it++) {
                    Eigen::Vector3d p(it->measurement.x, it->measurement.y, it->measurement.z);
                    patch_num[patch(p)]++;
                }
                max_patch = 0;
                for  (int i = 0; i != 12; i++) {
                    if (patch_num[i] > patch_num[max_patch])
                        max_patch = i;
                }
                LOG(INFO) << "max " << max_patch;

                ev::ComputeVarianceFunction varianceVisualizer(em, parameters_);

                Eigen::SparseMatrix<double> zero_motion;
                double* initial_depth = new double[parameters_.patch_num];
                for (unsigned i = 0; i != parameters_.patch_num; i++) {
                    initial_depth[i] = 2.;
                }
                Eigen::Vector3d zero_vec3 = Eigen::Vector3d::Zero();
                varianceVisualizer.Intensity(zero_motion, NULL, zero_vec3, zero_vec3, &initial_depth);
                double cost = contrastCost(zero_motion);
                //std::string caption =  "cost = " + std::to_string(cost);
                //#if show_optimizing_process
                //                ev::imshowRescaled(zero_motion, 20, "zero motion", NULL);
                //#endif
                delete initial_depth;
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

                //                bool positive = true;

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

                ev::Pose p1, p2;
                // of camera
                Eigen::Vector3d linear_velocity;
                Eigen::Quaterniond angular_velocity;
                ;

                if (interpolateGroundtruth(p1, begin) && interpolateGroundtruth(p2, end)) {
                    linear_velocity = (p1.q).inverse().toRotationMatrix() * (p2.p - p1.p) / (end.toSec() - begin.toSec());
                    // blender coordinate
                    // linear_velocity(1) = -linear_velocity(1)
                    // (p1.q).inverse() * (p2.q * (p1.q).inverse()) * p1.q
                    angular_velocity =  (p1.q).inverse() * p2.q ;

                } else {
                    linear_velocity = Eigen::Vector3d(0, 0, 0);
                    angular_velocity = Eigen::Quaterniond(1, 0, 0, 0);
                }
                Eigen::AngleAxisd angleAxis = Eigen::AngleAxisd(angular_velocity);
                Eigen::Vector3d angularVelocity = angleAxis.axis() * angleAxis.angle()  / (end.toSec() - begin.toSec());

                //                LOG(INFO) << begin;
                //                LOG(INFO) << "events: " << em->events.size() << '\n';
                //                LOG(INFO) << end << '\n';

                std::stringstream ss;
                ss << "\nground truth:\n\n"
                   << std::left << std::setw(15) << "angular :" << std::setw(15) << "linear :" << '\n'
                   << std::setw(15) << angularVelocity(0)  << std::setw(15) << linear_velocity(0) << '\n'
                   << std::setw(15) << angularVelocity(1)  << std::setw(15) << linear_velocity(1) << '\n'
                   << std::setw(15) << angularVelocity(2)  << std::setw(15) << linear_velocity(2) << '\n';

                LOG(INFO) << ss.str();
                //                v[0] =  linear_velocity(0);
                //                v[1] =  velocity(1);
                //                v[2] =  velocity(2);

                //                w[0] = angularVelocity(0);
                //                w[1] = angularVelocity(1);
                //                w[2] = angularVelocity(2);
#if !optimize
                Eigen::SparseMatrix<double> ground_truth;
                double* groundtruth_depth = new double[parameters_.patch_num];

                for (unsigned i = 0; i != parameters_.patch_num; i++) {
                    groundtruth_depth[i] = 2;
                }

                varianceVisualizer.Intensity(ground_truth, NULL, angularVelocity, linear_velocity, &groundtruth_depth);
                cost = contrastCost(ground_truth);
                std::string caption =  "cost = " + std::to_string(cost);
                ev::imshowRescaled(ground_truth, 10, caption, NULL);

//                for (int i = 0; i != 3; i++) {
//                     linear_velocity(2) = linear_velocity(0);
//                }

                varianceVisualizer.Intensity(ground_truth, NULL, zero_vec3, zero_vec3, &groundtruth_depth);
                cost = contrastCost(ground_truth);
                caption =  "cost = " + std::to_string(cost);
                ev::imshowRescaled(ground_truth, 10, caption, NULL);
                count++;

                delete groundtruth_depth;
#else
                processEventTimer.start();

                ceres::Solver::Options options;
                options.minimizer_progress_to_stdout = false;

                ceres::Solver::Summary summary;
                ceres::Problem problem;

                ceres::CostFunction* cost_function = new ComputeVarianceFunction(em, parameters_);

                // identity initialization
                double w_[] = {.0, .0, .0};
                double v_[] = {.0, .0, .0};
                double z_[parameters_.patch_num] = {};
                for (int i = 0; i < parameters_.patch_num; i++) {
                    z_[i] = 2.;
                }
                std::vector<double*> params_;
                params_.push_back(w_);
                params_.push_back(v_);
                params_.push_back(z_);

                ceres::Solver::Options options_;
                options_.minimizer_progress_to_stdout = false;

                ceres::Solver::Summary summary_;
                ceres::Problem problem_;

                ceres::CostFunction* cost_function_ = new ComputeVarianceFunction(em, parameters_);

//#pragma omp parallel
                {
//#pragma omp sections
                    {
//#pragma omp section
                        {
                            problem.AddResidualBlock(cost_function, NULL, params);
                            ceres::Solve(options, &problem, &summary);
                        }
//#pragma omp section
                        if (summary.final_cost > cost) {
                            problem_.AddResidualBlock(cost_function_, NULL, params_);
                            ceres::Solve(options_, &problem_, &summary_);
                        }
                    }
                }
                if (summary.final_cost > cost && summary_.final_cost < summary.final_cost) {
                    for (int i = 0; i != 3; i++) {
                        w[i] = w_[i];
                    }
                    for (int i = 0; i != 3; i++) {
                        v[i] = v_[i];
                    }
                    for (int i = 0; i != parameters_.patch_num; i++) {
                        z[i] = z_[i];
                    }
                    cost_function = cost_function_;
                    LOG(ERROR) << "change of direction";
                }

                // ev::imshowCallback callback(w, static_cast<ComputeVarianceFunction*>(cost_function));
                // options.callbacks.push_back(&callback);


#if !show_optimizing_process

                ev::imshowRescaled(zero_motion, 1, "zero motion", NULL);
                ev::imshowRescaled(static_cast<ComputeVarianceFunction*>(cost_function)->intensity,
                                   1, "optimized", z);
                count++;
#endif

                for (int i = 0; i < 3; i++) {
                    groundtruth[i].push_back(std::make_pair(end.toSec(), angularVelocity(i)));
                    estimatedPoses[i].push_back(std::make_pair(end.toSec(), w[i]));
                }

                //                gp << "plot '-' binary" << gp.binFmt1d(groundtruth[0], "record") << "with lines title 'w^1',"
                //                   << "'-' binary" << gp.binFmt1d(groundtruth[1], "record") << "with lines title 'w^2',"
                //                   << "'-' binary" << gp.binFmt1d(groundtruth[2], "record") << "with lines title 'w^3',"
                //                   << "'-' binary" << gp.binFmt1d(estimatedPoses[0], "record") << "with lines title 'w^1_{estimated}',"
                //                   << "'-' binary" << gp.binFmt1d(estimatedPoses[1], "record") << "with lines title 'w^2_{estimated}',"
                //                   << "'-' binary" << gp.binFmt1d(estimatedPoses[2], "record") << "with lines title 'w^3_{estimated}'\n";

                //                gp.sendBinary1d(groundtruth[0]);
                //                gp.sendBinary1d(groundtruth[1]);
                //                gp.sendBinary1d(groundtruth[2]);
                //                gp.sendBinary1d(estimatedPoses[0]);
                //                gp.sendBinary1d(estimatedPoses[1]);
                //                gp.sendBinary1d(estimatedPoses[2]);
                //                gp.flush();

                processEventTimer.stop();

                LOG(INFO) << okvis::timing::Timing::print();
                //Eigen::Vector3d rotation_(w[0], w[1], w[2]);
                //rotation_ *= ((end.toSec() - begin.toSec()));
                //Eigen::AngleAxisd angleAxis_;
                //if (rotation_.norm() == 0) {
                //    angleAxis_ = Eigen::AngleAxisd(0, (Eigen::Vector3d() << 0, 0, 1).finished());
                //} else {
                //    angleAxis_ = Eigen::AngleAxisd(rotation_.norm(), rotation_.normalized());
                //}
                //estimatedPose.q = Eigen::Quaterniond(angleAxis_) * estimatedPose.q;
                //                Eigen::AngleAxisd difference = Eigen::AngleAxisd(angleAxis_ * angleAxis.inverse());
                //                double error = difference.angle() / (end.toSec() - begin.toSec());

                ss.str(std::string());
                ss << '\n' << std::left << std::setw(15) << "angular :" << std::setw(15) << "linear :" << '\n'
                   << std::setw(15) << w[0]  << std::setw(15) << v[0] << '\n'
                   << std::setw(15) << w[1]  << std::setw(15) << v[1] << '\n'
                   << std::setw(15) << w[2]  << std::setw(15) << v[2] << '\n';

                LOG(INFO) << ss.str();
                std::string s("\ndepth :");
                for (unsigned i = 0; i != parameters_.patch_num; i++) {
                    s = s + " " + std::to_string(z[i]);
                }
                s += '\n';
                ss.str(s);
                LOG(INFO) << ss.str();

                std::ofstream  myfile("./shapes_translation_fix1depth/groundtruth_rotation.txt", std::ios_base::app);
                if (myfile.is_open()) {
                    myfile << angularVelocity(0) << " "
                           << angularVelocity(1) << " "
                           << angularVelocity(2) << "\n";
                    myfile.close();
                } else
                    std::cout << "怎么肥四"<<std::endl;

                std::ofstream  myfile_("./shapes_translation_fix1depth/estimated_rotation.txt", std::ios_base::app);
                if (myfile_.is_open()) {
                    myfile_ << w[0] << " "
                                    << w[1] << " "
                                    << w[2] << "\n";
                    myfile_.close();
                } else
                    std::cout << "怎么肥四"<<std::endl;

                std::ofstream  myfilet("./shapes_translation_fix1depth/groundtruth_translation.txt", std::ios_base::app);
                if (myfilet.is_open()) {
                    myfilet << linear_velocity(0) << " "
                            << linear_velocity(1) << " "
                            << linear_velocity(2) << '\n';
                    myfilet.close();
                } else
                    std::cout << "怎么肥四"<<std::endl;

                std::ofstream  myfile_t("./shapes_translation_fix1depth/estimated_translation.txt", std::ios_base::app);
                if (myfile_t.is_open()) {
                    myfile_t << v[0] << " "
                                     << v[1] << " "
                                     << v[2] << " ";
                    for (int i = 0; i!= 12; i++) {
                        myfile_t << z[i] << " ";
                    }
                    myfile_t << '\n';
                    myfile_t.close();
                } else
                    std::cout << "怎么肥四"<<std::endl;

                Eigen::Vector3d estimated_normalized = (Eigen::Vector3d()<<v[0],v[1],v[2]).finished().normalized();
                double error = std::acos(linear_velocity.normalized().dot(estimated_normalized));

                std::ofstream  error_file("./shapes_translation_fix1depth/error.txt", std::ios_base::app);
                if (error_file.is_open()) {
                    error_file << error << '\n';
                    error_file.close();
                } else
                    std::cout << "怎么肥四"<<std::endl;

                LOG(INFO) << "error: " << error << " rad/s";
                LOG(INFO) << summary.BriefReport();
#endif
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

double ThreadedEventIMU::contrastCost(Eigen::SparseMatrix<double>& image) {
    // average intensity
    double mu = image.sum() / (240*180);

    // cost
    double cost = 0;
    for (int s = 0; s < image.outerSize(); ++s) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(image, s); it; ++it) {
            double rho = it.value() - mu;
            cost += std::pow(rho, 2);
        }
    }

    cost += ((240 * 180) - image.nonZeros()) * std::pow(mu, 2);
    cost /= (240*180);

    // adjust to ceres format
    cost = 1./std::pow(cost, 2);
    cost /= 2;
    return cost;
}

}
