#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#define OPTIMIZE true
namespace ev {
struct Parameters
{
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // camera parameters
    unsigned window_size{10000};
    unsigned width{240};
    unsigned height{180};
    cv::Mat cameraMatrix{cv::Mat::eye(3, 3, CV_64F)};
    Eigen::Matrix3d cameraMatrix_;
    cv::Vec<double, 5> distCoeffs;
    bool write_to_file = false;
    std::string experiment_name;
    std::string path;
    Parameters() {}
};
}

