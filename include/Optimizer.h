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
    void static optimize(shared_ptr<MapPoint>& pMP);

    void warp(cv::Mat& x_w, cv::Mat& x,okvis::Duration& t, const cv::Mat& w, const cv::Mat& v,const cv::Mat& n) const;
    // inline??
    // add event to frame via bilinear interpolation
    void fuse(cv::Mat& image, cv::Point2d& p, bool& polarity) const;
    void intensity(shared_ptr<MapPoint>& pMP) const;
public:
    static Parameters param;


};
}
