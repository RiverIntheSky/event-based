#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace ev {
struct Parameters
{
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // camera parameters
    unsigned window_size{300};
    unsigned step_size{300};
    unsigned array_size_x{240};
    unsigned array_size_y{180};
    cv::Mat cameraMatrix{cv::Mat::eye(3, 3, CV_64F)};
    cv::Vec<double, 5> distCoeffs;
};
}

#endif // PARAMETERS_H
