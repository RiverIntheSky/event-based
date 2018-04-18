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
    parameters.cameraMatrix.at<double>(1, 1) = fy;

    std::getline(file, s, ' ');
    double cx = std::stod(s);
    parameters.cameraMatrix.at<double>(0, 2) = cx;

    std::getline(file, s, ' ');
    double cy = std::stod(s);
    parameters.cameraMatrix.at<double>(1, 2) = cy;
    
    LOG(INFO) << parameters.cameraMatrix;

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

void imshowRescaled(const cv::Mat &src, int msec, std::string title, std::string text) {
    cv::Mat dst;
#if 1
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    cv::subtract(src, cv::Mat(src.rows, src.cols, CV_64F, cv::Scalar(min)), dst);

    dst /= (max - min);
#else
    dst = src;
#endif
    cv::putText(dst, text, cvPoint(30,30),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
    cv::imshow(title, dst);
    cv::waitKey(msec);
}

void imshowRescaled(Eigen::MatrixXd &src_, int msec, std::string title, std::string text) {
    cv::Mat src;
    cv::eigen2cv(src_, src);
    imshowRescaled(src, msec, title, text);
}

void quat2eul(Eigen::Quaterniond& q, double* euler) {
    double y2 = q.y() * q.y();
    euler[0] = std::atan2(2*(q.w()*q.z() + q.x()*q.y()), (1 - 2*(y2 + q.z()*q.z())));
    euler[1] = std::asin( 2*(q.w()*q.y() - q.z()*q.x()));
    euler[2] = std::atan2(2*(q.w()*q.x() + q.y()*q.z()), (1 - 2*(q.x()*q.x() + y2)));
}
}
