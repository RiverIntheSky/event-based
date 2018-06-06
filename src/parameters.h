#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#define optimize true
namespace ev {
struct Parameters
{
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // camera parameters
    unsigned window_size{10000};
    unsigned step_size{4999};
    unsigned array_size_x{240};
    unsigned array_size_y{180};
    cv::Mat cameraMatrix{cv::Mat::eye(3, 3, CV_64F)};
    cv::Vec<double, 5> distCoeffs;
    unsigned patch_width{60};
    unsigned patch_num{std::ceil(array_size_x/patch_width) * std::ceil(array_size_y/patch_width)};
    bool write_to_file = false;
    std::string experiment_name;
    std::string path;
    Parameters() {}
};
}

#endif // PARAMETERS_H
