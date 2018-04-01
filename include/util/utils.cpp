#include "utils.h"
#include <glog/logging.h>

namespace ev{
parameterReader::parameterReader(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        LOG(ERROR)<< "no config file found at " << filename;
    }
    std::string s;

    std::getline(file, s, ' ');
    double fx = std::stod(s);
    parameters.cameraMatrix.at<double>(0, 0) = fx;

    std::getline(file, s, ' ');
    double fy = std::stod(s);
//    parameters.cameraMatrix.at<double>(1, 1) = fy;

    std::getline(file, s, ' ');
    double cx = std::stod(s);
//    parameters.cameraMatrix.at<double>(0, 2) = cx;

    std::getline(file, s, ' ');
    double cy = std::stod(s);
    //parameters.cameraMatrix.at<double>(1, 2) = cy;

    std::getline(file, s, ' ');
    double k1 = std::stod(s);

    std::getline(file, s, ' ');
    double k2 = std::stod(s);

    std::getline(file, s, ' ');
    double p1 = std::stod(s);

    std::getline(file, s, ' ');
    double p2 = std::stod(s);

    std::getline(file, s, ' ');
    double k3 = std::stod(s);

    parameters.distCoeffs = cv::Vec<double, 5>{k1, k2, p1, p2, k3};

}

bool parameterReader::getParameter(ev::Parameters& parameter){
    parameter = parameters;
    return true;
}
}
