#include <iostream>
#include <fstream>
#include "ThreadedEventIMU.h"
#include <chrono>

ev::Parameters* ev::Optimizer::param = NULL;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
#ifndef NDEBUG
    FLAGS_logtostderr = 1;  //Logs are written to standard error instead of to files.
#endif
    FLAGS_colorlogtostderr = 1;
    FLAGS_logtostderr = 1;

    if (argc <= 1) {
        LOG(ERROR) << "\nusage: " << "./event_based dataset_name window_size experiment_name\n"
                  << "example: " << "./event_based "
                  << "/home/weizhen/Documents/dataset/slider_hdr_close"
                  << " 50000 test\n";
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

    okvis::Time start(0.0);
    okvis::Time t_ev = start;

    okvis::Duration deltaT(1.0);

    std::string configFilename = path + "/calib.txt";
    ev::parameterReader pr(configFilename);
    ev::Parameters parameters;
    pr.getParameter(parameters);
    ev::Optimizer::param = &parameters;

    parameters.path = path;
    if (argc > 2) {
        parameters.window_size = atoi(argv[2]);
        if (argc > 3) {
            parameters.experiment_name = argv[3];
            parameters.write_to_file = true;
            std::string files_path;
            files_path = parameters.path + "/" + parameters.experiment_name + "/" + std::to_string(parameters.window_size);
            // warning: this will rewrite the original file
            system(("rm -rf " + files_path).c_str());
            system(("mkdir -p " + files_path).c_str());
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

        ev_estimator.addGroundtruth(t_gt, position, orientation);
    }

    ev_estimator.allGroundtruthAdded_ = true;

    do {
        if (!std::getline(events_file, events_line)) {
            std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
            cv::waitKey();
            return 0;
        }

        std::stringstream stream_ev(events_line);
        std::string s;
        std::getline(stream_ev, s, ' ');
        std::string nanoseconds = s.substr(s.find("."));
        std::string seconds = s.substr(0, s.find("."));
        t_ev = okvis::Time(std::stoi(seconds), std::stod(nanoseconds)*1e9);

        std::getline(stream_ev, s, ' ');
        unsigned int x = std::stoi(s);

        std::getline(stream_ev, s, ' ');
        unsigned int y = std::stoi(s);

        std::getline(stream_ev, s, ' ');
        bool p = std::stoi(s);

        if (t_ev - start > deltaT) {
            ev_estimator.addEventMeasurement(t_ev, x, y, p);
        }

    } while (true);

    return 0;
}
