
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
    std::uniform_real_distribution<double> dis(-0.02, 0.02);
    double w[] = {0, 0, 0};
    double v[] = {0, 0, 0};
//    double p[] = {0, M_PI}; // normal (0, 0, -1)
    double p[3*parameters_.patch_num] = {};
    for (int i = 0; i < 3*parameters_.patch_num; i+=3) {
                        p[i] = 0;
    }
    for (int i = 1; i < 3*parameters_.patch_num; i+=3) {
                        p[i] = M_PI;
    }
    for (int i = 2; i < 3*parameters_.patch_num; i+=3) {
                        p[i] = 1.;
    }
    double p_[3*parameters_.patch_num] = {};
    for (int i = 0; i < 3*parameters_.patch_num; i+=3) {
                        p_[i] = 0;
    }

    for (int i = 1; i < 3*parameters_.patch_num; i+=3) {
                        p_[i] = M_PI;
    }
    for (int i = 2; i < 3*parameters_.patch_num; i+=3) {
                        p_[i] = 1.;
    }

    while (!allGroundtruthAdded_) {LOG(INFO) << "LOADING GROUNDTRUTH";}

    count = 0;

    // helper function, divide scene into patches

    for (;;) {
        // get data and check for termination request
        if (eventMeasurementsReceived_.PopBlocking(&data) == false) {
            std::lock_guard<std::mutex> lock(eventMeasurements_mutex_);
            LOG(INFO) << "size " << eventMeasurementsReceived_.Size();
            return;
        }


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

                undistortEvents(em);

                ev::ComputeVarianceFunction varianceVisualizer(em, parameters_);

                Eigen::MatrixXd zero_motion;

                Eigen::Vector3d zero_vec3 = Eigen::Vector3d::Zero();
                Eigen::Vector3d vertical;
                vertical << 0, 0, -1;
//                varianceVisualizer.Intensity(zero_motion, NULL, zero_vec3, zero_vec3, vertical);
//                double cost = contrastCost(zero_motion);

                // ground truth
                okvis::Time begin = em->events.front().timeStamp;
                okvis::Time end = em->events.back().timeStamp;

                // ???

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
                Eigen::Vector3d angularVelocity = angleAxis.axis() * angleAxis.angle()  / (end.toSec() - begin.toSec());

                std::stringstream ss;
                ss << "\nground truth:\n\n"
                   << std::left << std::setw(15) << "angular :" << std::setw(15) << "linear :" << '\n'
                   << std::setw(15) << angularVelocity(0)  << std::setw(15) << linear_velocity(0) << '\n'
                   << std::setw(15) << angularVelocity(1)  << std::setw(15) << linear_velocity(1) << '\n'
                   << std::setw(15) << angularVelocity(2)  << std::setw(15) << linear_velocity(2) << '\n';

                LOG(INFO) << ss.str();

//                w[0] = angularVelocity(0) + dis(gen);
//                w[1] = angularVelocity(1) + dis(gen);
//                w[2] = angularVelocity(2) + dis(gen);

//                v[0] = linear_velocity(0) + dis(gen);
//                v[1] = linear_velocity(1) + dis(gen);
//                v[2] = linear_velocity(2) + dis(gen);

//                                w[0] = angularVelocity(0);
//                                w[1] = angularVelocity(1);
//                                w[2] = angularVelocity(2);

//                                v[0] = linear_velocity(0);
//                                v[1] = linear_velocity(1);
//                                v[2] = linear_velocity(2);

                Eigen::MatrixXd ground_truth;
                std::string files_path = parameters_.path + "/" + parameters_.experiment_name + "/" + std::to_string(parameters_.window_size) + "/";
//#if !optimize
#if 0
                std::ofstream  myfile(files_path + "z.txt", std::ios_base::app);
                if (myfile.is_open() && linear_velocity(0) > 0.3) {
                    for (double i=-10.; i<10.;i++){
                        groundtruth_depth[2] = i/10+0.01;
                        groundtruth_depth[5] = i/10+0.01;
                        groundtruth_depth[8] = i/10+0.01;
                        groundtruth_depth[11] = i/10+0.01;
                        varianceVisualizer.Intensity(ground_truth, NULL, angularVelocity, linear_velocity, &groundtruth_depth);

                        double cost_zero = contrastCost(ground_truth);
                        ev::imshowRescaled(ground_truth, 10, files_path+std::to_string(cost_zero), groundtruth_depth);
                    }

                    LOG(INFO) << "sleep";
                    system("sleep 10");
                    myfile.close();
                }

#else

                Eigen::Vector3d l = linear_velocity/0.231;
                varianceVisualizer.Intensity(ground_truth, NULL, zero_vec3, zero_vec3, p_);
                double cost = contrastCost(ground_truth);
//                std::string caption =  "cost = " + std::to_string(cost);
                LOG(INFO)<<"cost = " << cost;
                ev::imshowRescaled(ground_truth, 10, files_path + "zero_motion", NULL);

                processEventTimer.start();

                ev::ComputeVarianceFunction param(em, parameters_);

                const gsl_multimin_fminimizer_type *T =
                  gsl_multimin_fminimizer_nmsimplex2;
                gsl_multimin_fminimizer *s = NULL;
                gsl_vector *step_size, *x;
                gsl_multimin_function minex_func;

                size_t iter = 0;
                int status;
                double size;

                /* Starting point */
                x = gsl_vector_alloc(41);

                for (int i = 0; i < 3; i ++)
                    gsl_vector_set(x, i, w[i]);

                for (int i = 0; i < 3; i ++)
                    gsl_vector_set(x, i+3, v[i]);

                for (int i = 0; i < 2; i ++)
                    gsl_vector_set(x, i+6, p[i]);

                for (int i = 3; i < 36; i ++)
                    gsl_vector_set(x, i+5, p[i]);

                /* Set initial step sizes to 1 */
                step_size = gsl_vector_alloc(41);
                gsl_vector_set_all(step_size, 1);

                /* Initialize method and iterate */
                minex_func.n = 41;
                minex_func.f = variance;
                minex_func.params = &param;

                s = gsl_multimin_fminimizer_alloc(T, 41);
                gsl_multimin_fminimizer_set(s, &minex_func, x, step_size);

                do
                {
                    iter++;
                    status = gsl_multimin_fminimizer_iterate(s);

                    if (status)
                        break;

                    size = gsl_multimin_fminimizer_size(s);
                    status = gsl_multimin_test_size(size, 1e-2);

//                    if (status == GSL_SUCCESS)
//                    {
//                        printf ("converged to minimum at\n");
//                    }

//                    LOG(INFO) << size;

                }
                while (status == GSL_CONTINUE && iter < 100);

                for (int i = 0; i < 3; i++) {
                    w[i] = gsl_vector_get(s->x, i);

                }
                for (int i = 0; i < 3; i++) {
                    v[i] = gsl_vector_get(s->x, i+3);
                }
                p[0] = gsl_vector_get(s->x, 6);
                p[1] = gsl_vector_get(s->x, 7);
                for (int i = 3; i < 3*parameters_.patch_num-1; i++) {
                    p[i] = gsl_vector_get(s->x, i+5);
                }

                gsl_vector_free(x);
                gsl_vector_free(step_size);
                gsl_multimin_fminimizer_free (s);


#if !show_optimizing_process

                ev::imshowRescaled(param.intensity,
                                   1, files_path + "optimized", NULL);

                count++;



#endif

                processEventTimer.stop();

                LOG(INFO) << okvis::timing::Timing::print();

                ss.str(std::string());
                ss << '\n' << std::left << std::setw(15) << "angular :" << std::setw(15) << "linear :" << '\n'
                   << std::setw(15) << w[0] << std::setw(15) << v[0] << '\n'
                   << std::setw(15) << w[1] << std::setw(15) << v[1] << '\n'
                   << std::setw(15) << w[2] << std::setw(15) << v[2] << '\n';

                LOG(INFO) << ss.str();
//                LOG(INFO) << "normal " << std::cos(p[0]) * std::sin(p[1]) << ' ' << std::sin(p[0]) * std::sin(p[1]) << ' ' << std::cos(p[1]);
                LOG(INFO)<<"cost = " << contrastCost(param.intensity);
                Eigen::Vector3d rotation_(w[0], w[1], w[2]);
                Eigen::AngleAxisd angleAxis_;
                rotation_ *= ((end.toSec() - begin.toSec()));
                if (rotation_.norm() == 0) {
                    angleAxis_ = Eigen::AngleAxisd(0, (Eigen::Vector3d() << 0, 0, 1).finished());
                } else {
                    angleAxis_ = Eigen::AngleAxisd(rotation_.norm(), rotation_.normalized());
                }
                Eigen::AngleAxisd difference = Eigen::AngleAxisd(angleAxis_ * angleAxis.inverse());
                double error = difference.angle() / (end.toSec() - begin.toSec());

                LOG(ERROR) << "error: " << error << " rad/s";

                if (parameters_.write_to_file) {

                    std::string files_path = parameters_.path + "/" + parameters_.experiment_name + "/" + std::to_string(parameters_.window_size) + "/";


                    std::ofstream  myfile(files_path + "groundtruth_rotation.txt", std::ios_base::app);
                    if (myfile.is_open()) {
                        myfile << begin.toSec() << " "
                               << angularVelocity(0) << " "
                               << angularVelocity(1) << " "
                               << angularVelocity(2) << "\n";
                        myfile.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_(files_path + "estimated_rotation.txt", std::ios_base::app);
                    if (myfile_.is_open()) {
                        myfile_ << begin.toSec() << " "
                                << w[0] << " "
                                << w[1] << " "
                                << w[2] << "\n";
                        myfile_.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfilet(files_path + "groundtruth_translation.txt", std::ios_base::app);
                    if (myfilet.is_open()) {
                        myfilet << begin.toSec() << " "
                                << linear_velocity(0) << " "
                                << linear_velocity(1) << " "
                                << linear_velocity(2) << '\n';
                        myfilet.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;

                    std::ofstream  myfile_t(files_path + "estimated_translation.txt", std::ios_base::app);
                    if (myfile_t.is_open()) {
                        myfile_t<< begin.toSec() << " "
                                << v[0] << " "
                                << v[1] << " "
                                << v[2] << " ";
                        myfile_t << '\n';
                        myfile_t.close();
                    } else
                        std::cout << "怎么肥四"<<std::endl;
                }
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
