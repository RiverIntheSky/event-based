
#include "ThreadedEventIMU.h"

namespace ev {
// counter for writing images to file
int count = 0;

ThreadedEventIMU::ThreadedEventIMU(Parameters &parameters)
    : parameters_(parameters),
      maxEventInputQueueSize_(3000000) {
    init();
}

ThreadedEventIMU::~ThreadedEventIMU() {   
    eventMeasurementsReceived_.Shutdown();
    eventConsumerThread_.join();
}

void ThreadedEventIMU::init() {
    mpMap = new Map();
    mpTracker = new Tracking(parameters_.cameraMatrix, parameters_.distCoeffs, mpMap);

    // the projection matrix to the map
    Optimizer::mCameraProjectionMat(0, 0) = 200;
    Optimizer::mCameraProjectionMat(1, 1) = 200;
    Optimizer::mCameraProjectionMat(0, 2) = 300;
    Optimizer::mCameraProjectionMat(1, 2) = 300;
    startThreads();
}

bool ThreadedEventIMU::addEventMeasurement(okvis::Time& t, unsigned int x, unsigned int y, bool p) {

//    if (x < 30 && y < 30)
//        return false;

    ev::EventMeasurement event_measurement;
    event_measurement.measurement.x = x;
    event_measurement.measurement.y = y;
    event_measurement.measurement.z = 1;
    event_measurement.measurement.p = p;
    event_measurement.timeStamp = t;

    eventMeasurementsReceived_.PushBlockingIfFull(event_measurement, 1);
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
    return true;
}

// Loop to process event measurements.
void ThreadedEventIMU::eventConsumerLoop() {
    LOG(INFO) << "Event consumer loop joined";

    while (!allGroundtruthAdded_) {system("sleep 1");}

    count = 0;

    std::string files_path = parameters_.path + "/" + parameters_.experiment_name + "/" + std::to_string(parameters_.window_size) + "/";

    // ?
    okvis::Time start(1.0);
    ev::Pose p;
    interpolateGroundtruth(p, start);

    // the starting pose
    Eigen::Quaterniond R0 = p.q.inverse();
    Eigen::Vector3d t0 = p.p;

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

                okvis::Time begin =(*(mCurrentFrame->vEvents.begin()))->timeStamp;
                okvis::Time end = (*(mCurrentFrame->vEvents.rbegin()))->timeStamp;

                LOG(INFO) << "time stamp: " << begin.toSec();

                ev::Pose p1, p2; /* start pose and end pose of the current frame */
                // camera velocity
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

                // planar scene
                auto pMP = mpMap->getAllMapPoints().front();
                imwriteRescaled(pMP->mFront, files_path + "map_" + std::to_string(count) + ".jpg", NULL);
                cv::Mat blurred_map;
                cv::GaussianBlur(pMP->mFront, blurred_map, cv::Size(0, 0), 1, 0);
                imshowRescaled(blurred_map, 1, "map");

                LOG(INFO) << "keyframes " << mpMap->getAllKeyFrames().size();
                LOG(INFO) << "frames " << ++count;
                LOG(INFO) << "normal " << pMP->getNormal();

                if (parameters_.write_to_file) {
                    writeToFile(begin, R_w, files_path + "groundtruth_pose_rotation.txt");
                    writeToFile(begin, t, files_path + "groundtruth_pose_translation.txt");
                    writeToFile(begin, angularVelocity, files_path + "groundtruth_rotation.txt");
                    writeToFile(begin, linear_velocity, files_path + "groundtruth_translation.txt");

                    writeToFile(begin, mpTracker->r, files_path + "estimated_pose_rotation.txt");
                    writeToFile(begin, mpTracker->t, files_path + "estimated_pose_translation.txt");
                    writeToFile(begin, mpTracker->w, files_path + "estimated_rotation.txt");
                    writeToFile(begin, mpTracker->v, files_path + "estimated_translation.txt");
                }
            }
        }
    }
}

// Start all threads. Currently only eventConsumerThread implemented
void ThreadedEventIMU::startThreads() {
    eventConsumerThread_ = std::thread(&ThreadedEventIMU::eventConsumerLoop, this);
}

bool ThreadedEventIMU::writeToFile(okvis::Time& t, Eigen::Vector3d& vec, std::string file_name) {
    std::ofstream  file(file_name, std::ios_base::app);
    if (file.is_open()) {
        file << t.toSec() << " "
             << vec(0) << " "
             << vec(1) << " "
             << vec(2) << '\n';
        file.close();
        return true;
    } else {
        LOG(ERROR) << strerror(errno);
        return false;
    }
}

bool ThreadedEventIMU::writeToFile(okvis::Time& t, cv::Mat& vec, std::string file_name) {
    std::ofstream  file(file_name, std::ios_base::app);
    if (file.is_open()) {
        file << t.toSec() << " "
             << vec.at<double>(0) << " "
             << vec.at<double>(1) << " "
             << vec.at<double>(2) << '\n';
        file.close();
        return true;
    } else {
        LOG(ERROR) << strerror(errno);
        return false;
    }
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

}
