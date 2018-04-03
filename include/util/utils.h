#ifndef UTILS_H
#define UTILS_H

#include "parameters.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

namespace ev {
class parameterReader
{
public:
    parameterReader(const std::string& filename);
    bool getParameter(ev::Parameters& parameter);

    ev::Parameters parameters;
};

// map entries in src to [0, 1]
void imshowRescaled(const cv::Mat& src, int msec = 0, std::string s = "image");
void imshowRescaled(Eigen::MatrixXd &src_, int msec = 0, std::string s = "image");
}

#endif // UTILS_H
