#include <iostream>
#include <fstream>
#include "ThreadedEventIMU.h"
#include <chrono>

Eigen::MatrixXd ev::ComputeVarianceFunction::intensity = Eigen::MatrixXd::Zero(180, 240);
ev::Parameters* ev::Optimizer::param = NULL;

int main(int argc, char *argv[])
{

    google::InitGoogleLogging(argv[0]);
#ifndef NDEBUG
    //FLAGS_alsologtostderr = 1;  // output LOG for debugging
    FLAGS_logtostderr = 1;  //Logs are written to standard error instead of to files.
#endif
    FLAGS_colorlogtostderr = 1;
    FLAGS_logtostderr = 1;

    if (argc <= 1) {
        LOG(ERROR) << "\nusage: " << "./event_based dataset_name window_size overlap_rate experiment_name\n"
                  << "example: " << "./event_based "
                  << "/home/weizhen/Documents/dataset/slider_hdr_close"
                  << " 10000 0.5 x_only\n";
        return 1;
    }

    // Measurement data path
    std::string path = argv[1];
    LOG(INFO) << "dataset: " + path;

    // open the events file
    std::string events_line;
    std::string events_path = path + "/events.txt";
    std::ifstream events_file(events_path);
    if (!events_file.good()) {
        LOG(ERROR)<< "no events file found at " << events_path;
        return -1;
    }

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

    imu_file.clear();
    imu_file.seekg(0);

    okvis::Time start(0.0);
    okvis::Time t_imu = start;
    okvis::Time t_ev = start;

    okvis::Duration deltaT(1);

    std::string configFilename = path + "/calib.txt";
    ev::parameterReader pr(configFilename);
    ev::Parameters parameters;
    pr.getParameter(parameters);
    ev::Optimizer::param = &parameters;
    ev::Optimizer::mCameraProjectionMat = parameters.K_;

//    std::string im = argv[5];
//    std::string imPath = "/home/weizhen/Documents/dataset/calibration/images/frame_0000" + im + ".png";
//    cv::Mat src = cv::imread(imPath, CV_LOAD_IMAGE_GRAYSCALE);
//    cv::Mat dst;
//    cv::imshow("src", src);
//    cvWaitKey(0);
//    cv::undistort(src, dst, parameters.K, parameters.distCoeffs);
//    cv::imshow("dst", dst);
//    cvWaitKey(0);
//    cv::undistort(src, dst, parameters.K, parameters.distCoeffs, parameters.K_n);
//    cv::imshow("dst_with_k", dst);
//    cvWaitKey(0);


    parameters.path = path;
    if (argc > 2) {
        parameters.window_size = atoi(argv[2]);
        if (argc > 3) {
            parameters.step_size = std::floor(parameters.window_size * (1 - atof(argv[3])));
            if (argc > 4) {
                parameters.experiment_name = argv[4];
                parameters.write_to_file = true;
                std::string files_path;
                files_path = parameters.path + "/" + parameters.experiment_name + "/" + std::to_string(parameters.window_size);
                system(("rm -rf " + files_path).c_str());
                system(("mkdir -p " + files_path).c_str());
                LOG(INFO) << "write to " << files_path;
            }
        }
    }


    LOG(INFO) << "window_size: " << parameters.window_size;

    ev::ThreadedEventIMU ev_estimator(parameters);

    // open the groundtruth file
    std::string groundtruth_line;
    std::string groundtruth_path = path + "/groundtruth.txt";
    std::ifstream groundtruth_file(groundtruth_path);
    if (!groundtruth_file.good()) {
        LOG(ERROR)<< "no groundtruth file found at " << groundtruth_path;
        return -1;
    }

    while (std::getline(groundtruth_file, groundtruth_line)) {

        std::stringstream stream(groundtruth_line);
        std::string s;
        std::getline(stream, s, ' ');
        std::string nanoseconds = s.substr(s.find("."));
        std::string seconds = s.substr(0, s.find("."));

        okvis::Time t_gt = okvis::Time(std::stoi(seconds), std::stod(nanoseconds)*1e9);

        Eigen::Vector3d position;
        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ' ');
            position[j] = std::stof(s);
        }

        double w, x, y, z;
        stream >> x;
        stream >> y;
        stream >> z;
        stream >> w;

        if (w < 0) {
            w = -w; x = -x; y = -y; z = -z;
        }

        // q = -q not strictly recognized
        Eigen::Quaterniond orientation(w, x, y, z);

        if (t_gt - start > deltaT) {
            ev_estimator.addGroundtruth(t_gt, position, orientation);
        }
    }

ev_estimator.allGroundtruthAdded_ = true;

    while (std::getline(imu_file, imu_line)) {

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
            nanoseconds = s.substr(s.find("."));
            seconds = s.substr(0, s.find("."));
            t_ev = okvis::Time(std::stoi(seconds), std::stod(nanoseconds)*1e9);

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
