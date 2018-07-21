#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#define OPTIMIZE true
namespace ev {
struct Parameters
{
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    unsigned window_size{10000};
    unsigned step_size{9999};

    // camera parameters
    int width{240};
    int height{180};
    float fx;
    float fy;
    float cx;
    float cy;
    cv::Mat distCoeffs;
    cv::Mat K{cv::Mat::eye(3, 3, CV_32F)};
    Eigen::Matrix3d K_;

//    unsigned patch_width{60};
//    int patch_num{std::ceil(array_size_x/patch_width) * std::ceil(array_size_y/patch_width)};
    bool write_to_file{false};
    bool use_polarity{false};

    // map parameters
    // ??
    float znear{0.1f};
    float zfar{100.f};

    std::string experiment_name;
    std::string path;
    Parameters() {}
};
}

