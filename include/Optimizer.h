#pragma once

#include "MapPoint.h"
#include "parameters.h"
#include "Converter.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

namespace ev {
class Optimizer
{
public:
    double static variance(const gsl_vector *vec, void *params);
    void static optimize(MapPoint* pMP);

    void inline static warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, const Eigen::Vector3d& w, const Eigen::Vector3d& v,const Eigen::Vector3d& n);
    void inline static warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                            const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& Rn);
    // inline??
    // add event to frame via bilinear interpolation
    void static fuse(cv::Mat& image, Eigen::Vector3d& p_, bool& polarity);
    void static intensity(cv::Mat& intensity, const gsl_vector *vec, MapPoint* pMP);
public:
    static Parameters* param;
    static Eigen::Matrix3d mPatchProjectionMat;

};
}
