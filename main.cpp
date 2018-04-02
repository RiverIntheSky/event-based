#include <iostream>
#include <fstream>
#include "ThreadedEventIMU.h"
#include "util/utils.h"
#include <chrono>

std::shared_ptr<ev::eventFrameMeasurement> ev::Contrast::em = NULL;
double ev::Contrast::I_mu;
Eigen::MatrixXd ev::Contrast::intensity = Eigen::Matrix3d::Zero();
ev::Parameters ev::Contrast::param = ev::Parameters();


int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
#ifndef NDEBUG
    //FLAGS_alsologtostderr = 1;  // output LOG for debugging
    FLAGS_logtostderr = 1;  //Logs are written to standard error instead of to files.
#endif
    FLAGS_colorlogtostderr = 1;

    // Measurement data path
    std::string path = "/home/weizhen/Documents/dataset/boxes_6dof/boxes_6dof";

    // open the events file
    std::string events_line;
    std::string events_path = path + "/events.txt";
    std::ifstream events_file(events_path);
    if (!events_file.good()) {
        LOG(ERROR)<< "no events file found at " << events_path;
        return -1;
    }
//    unsigned int number_of_lines;
//    while (std::getline(events_file, events_line))
//        ++number_of_lines;
//    LOG(INFO)<< "No. events measurements: " << number_of_lines;
//    if (number_of_lines <= 0) {
//        LOG(ERROR)<< "no events messages present in " << events_path;
//        return -1;
//    }
    // set reading position to first line
    events_file.clear();
    events_file.seekg(0);

    // open the IMU file
    std::string imu_line;
    std::string imu_path = path + "/imu.txt";
    std::ifstream imu_file(imu_path);
    if (!imu_file.good()) {
        LOG(ERROR)<< "no imu file found at " << imu_path;
        return -1;
    }
//    number_of_lines = 0;
//    while (std::getline(imu_file, imu_line))
//        ++number_of_lines;
//    LOG(INFO)<< "No. IMU measurements: " << number_of_lines;

//    if (number_of_lines <= 0) {
//        LOG(ERROR)<< "no imu messages present in " << imu_path;
//        return -1;
//    }
    // set reading position to first line
    imu_file.clear();
    imu_file.seekg(0);

    okvis::Time start(0.0);
    okvis::Time t_imu = start;
    okvis::Time t_ev = start;
    okvis::Duration deltaT(0.0);

    std::string configFilename = path + "/calib.txt";
    ev::parameterReader pr(configFilename);
    ev::Parameters parameters;
    pr.getParameter(parameters);

    ev::ThreadedEventIMU ev_estimator(parameters);

    while (std::getline(imu_file, imu_line) && t_imu < okvis::Time(1)) {

        std::stringstream stream(imu_line);
        std::string s;
        std::getline(stream, s, ' ');
        std::string nanoseconds = s.substr(s.size() - 9, 9);
        std::string seconds = s.substr(0, s.size() - 9);

        Eigen::Vector3d gyr;
        for (int j = 0; j < 3; ++j) {
          std::getline(stream, s, ' ');
          gyr[j] = std::stof(s);
        }

        Eigen::Vector3d acc;
        for (int j = 0; j < 3; ++j) {
          std::getline(stream, s, ' ');
          acc[j] = std::stof(s);
        }

        t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));


        do {

            if (!std::getline(events_file, events_line)) {
              std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
              cv::waitKey();
              return 0;
            }

            std::stringstream stream_ev(events_line);

            std::getline(stream_ev, s, ' ');
            nanoseconds = s.substr(s.size() - 9, 9);
            seconds = s.substr(0, s.size() - 9);
            t_ev = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

            std::getline(stream_ev, s, ' ');
            unsigned int x = std::stoi(s);

            std::getline(stream_ev, s, ' ');
            unsigned int y = std::stoi(s);

            std::getline(stream_ev, s, ' ');
            bool p = std::stoi(s);

            // ???
            if (t_ev - start > deltaT) {
                ev_estimator.addEventMeasurement(t_ev, x, y, p);

            }

        } while (t_ev <= t_imu);

        // add the IMU measurement for (blocking) processing
        // ???
        if (t_imu - start > deltaT) {
            ev_estimator.addImuMeasurement(t_imu, acc, gyr);
        }
    }



    return 0;
}
