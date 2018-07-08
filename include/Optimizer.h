#pragma once

#include "MapPoint.h"
#include "parameters.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

namespace ev {
class Optimizer
{
public:
    double static variance(const gsl_vector *vec, void *params);
    void static optimize(MapPoint* pMP);

    void static warp(cv::Mat& x_w, cv::Mat& x,okvis::Duration& t, const cv::Mat& w, const cv::Mat& v,const cv::Mat& n);
    // inline??
    // add event to frame via bilinear interpolation
    void static fuse(cv::Mat& image, cv::Mat& p_, bool& polarity);
    void static intensity(cv::Mat& intensity, const gsl_vector *vec, MapPoint* pMP);
public:
    static Parameters* param;


};
}
