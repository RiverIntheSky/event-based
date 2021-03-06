#pragma once

#include "MapPoint.h"
#include "MapDrawer.h"
#include "parameters.h"
#include "Converter.h"
#include "Tracking.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

namespace ev {
class Frame;
class KeyFrame;
class MapPoint;
struct idxOrder;

class Optimizer
{
public:
    // don't quite like this!!
    struct mapPointAndFrame {
        MapPoint* mP;
        Frame* frame;
    };

    struct mapPointAndKeyFrames {
        MapPoint* mP;
        std::set<std::shared_ptr<KeyFrame>, idxOrder>* kfs;
    };

    struct pixel{
        pixel(int x_, int y_, double p_): x(x_), y(y_), p(p_) {}
        int x;
        int y;
        double p;
    };

    struct pixel_value{
        bool operator()(const pixel lhs, const pixel rhs) const {
            return lhs.p > rhs.p;
        }
    };

    double static variance_map(const gsl_vector *vec, void *params);
    double static variance_track(const gsl_vector *vec, void *params);
    double static variance_frame(const gsl_vector *vec, void *params);
    double static variance_ba(const gsl_vector *vec, void *params);
    double static variance_relocalization(const gsl_vector *vec, void *params);
    void static optimize(Frame* frame);
    void static optimize(MapPoint* pMP);
    void static optimize(MapPoint* pMP, Frame* frame);
    bool static optimize(MapPoint* pMP, std::shared_ptr<KeyFrame>& pKF);
    bool static optimize(MapPoint* pMP, Frame* frame, cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v);
    void static optimize_gsl(double ss, int nv, double (*f)(const gsl_vector*, void*), void *params,
                             gsl_multimin_fminimizer* s, gsl_vector* x, double* res, size_t iter);
    void inline static warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, const Eigen::Vector3d& w, const Eigen::Vector3d& v,const Eigen::Vector3d& n);
    void inline static warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                            const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& Rn, const Eigen::Matrix3d& H_);
    void inline static warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                            const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& H_);
    // inline??
    // add event to frame via bilinear interpolation
    void static fuse(cv::Mat& image, Eigen::Vector3d& p_, bool polarity);
    void static intensity(cv::Mat& image, const gsl_vector *vec, Frame* pF);
    void static intensity(cv::Mat& image, const gsl_vector *vec, MapPoint* pMP);
    void static intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndFrame* mf);
    void static intensity(cv::Mat& image, const double *vec, mapPointAndFrame* mf);
    void static intensity_relocalization(cv::Mat& image, const gsl_vector *vec, mapPointAndFrame* mf);
    void static intensity_relocalization(cv::Mat& image, const double *vec, mapPointAndFrame* mf);
    void static intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndKeyFrames* mkf);
    void static intensity(cv::Mat& image, const double *vec, KeyFrame* kF);

    bool static inFrame_(cv::Mat Xw, cv::Mat& Rwc, cv::Mat& twc, float &x, float &y);
public:
    static Parameters* param;
    static Eigen::Matrix3d mPatchProjectionMat;
    static Eigen::Matrix3d mCameraProjectionMat;
    static bool inFrame;
    static bool toMap;
    static int sigma;
    static int count_frame;
    static int count_map;
};
}
