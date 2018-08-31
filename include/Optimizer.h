#pragma once

#include "MapPoint.h"
#include "parameters.h"
#include "Converter.h"
#include "Tracking.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

namespace ev {
class Optimizer
{
public:
    // parameters used in optimization
    struct mapPointAndFrame {
        MapPoint* mP;
        Frame* frame;
    };
    struct mapPointAndKeyFrames {
        MapPoint* mP;
        std::set<shared_ptr<KeyFrame>, idxOrder>* kfs;
    };

    // compute variance of the frame for each step in the pipeline
    // numerical differentiation
    double static variance_map(const gsl_vector *vec, void *params);
    double static variance_track(const gsl_vector *vec, void *params);
    double static variance_ba(const gsl_vector *vec, void *params);
    double static variance_relocalization(const gsl_vector *vec, void *params);

    // variance as well as it's jacobians for tracking each frame
    double static f_frame(const gsl_vector *vec, void *params);
    void static df_frame(const gsl_vector *vec, void *params, gsl_vector* df);
    void static fdf_frame(const gsl_vector *vec, void *params, double *f, gsl_vector* df);

    double static f_track(const gsl_vector *vec, void *params);
    void static df_track(const gsl_vector *vec, void *params, gsl_vector* df);
    void static fdf_track(const gsl_vector *vec, void *params, double *f, gsl_vector* df);

    // nonlinear optimization
    void static optimize(MapPoint* pMP);
    void static optimize(MapPoint* pMP, Frame* frame);
    bool static optimize(MapPoint* pMP, shared_ptr<KeyFrame>& pKF);
    bool static optimize(MapPoint* pMP, Frame* frame, cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v);
    /**
     * \brief          Optimization without derivatives
     * \param ss       Initial step size
     * \param nv       Number of parameters to be optimized
     * \param res      The optimized results as a vector
     * \param iter     Maximal number of iterations
     * Other parameters please refer to the GNU-GSL manual
     * https://www.gnu.org/software/gsl/doc/html/multimin.html
     */
    void static gsl_f(double ss, int nv, double (*f)(const gsl_vector*, void*), void *params,
                             gsl_multimin_fminimizer* s, gsl_vector* x, double* res, size_t iter);
    /**
     * \brief          Optimization with derivatives
     * \param nv       Number of parameters to be optimized
     * \param res      The optimized results as a vector
     * Other parameters please refer to the GNU-GSL manual
     * https://www.gnu.org/software/gsl/doc/html/multimin.html
     */
    void static gsl_fdf(double (*f)(const gsl_vector*, void*), void (*df)(const gsl_vector*, void*, gsl_vector*),
                                     void (*fdf)(const gsl_vector*, void*, double *, gsl_vector *), int nv, void *params,
                                     gsl_multimin_fdfminimizer* s, gsl_vector* x, double* res);

    // warp events
    void static warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                            const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& H_);

    // aggregate event to frame via bilinear interpolation
    void static fuse(Eigen::MatrixXd* dIdW, Eigen::MatrixXd* dW, cv::Mat& image, Eigen::Vector3d& p_, bool polarity);

    // synthesize event frame
    void static intensity(cv::Mat& image, const gsl_vector *vec, Eigen::MatrixXd* dIdW, Frame* pF);
    void static intensity(cv::Mat& image, const gsl_vector *vec, Frame* pF);
    void static intensity(cv::Mat& image, const gsl_vector *vec, MapPoint* pMP);
    void static intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndKeyFrames* mkf);
    void static intensity(cv::Mat& image, const gsl_vector *vec, Eigen::MatrixXd* dIdW, mapPointAndFrame* mf);
    void static intensity_relocalization(cv::Mat& image, const gsl_vector *vec, mapPointAndFrame* mf);

    // draw using the right parameter set
    void static intensity(cv::Mat& image, const double *vec, mapPointAndFrame* mf);
    void static intensity(cv::Mat& image, const double *vec, KeyFrame* kF);
    void static intensity_relocalization(cv::Mat& image, const double *vec, mapPointAndFrame* mf);

public:
    static Parameters* param;
    static Eigen::Matrix3d mCameraProjectionMat;

    // blur radius
    static int sigma;

    // size of the map
    static int width;
    static int height;

    // the two steps in tracking
    static bool inFrame;
    static bool toMap;

    // whether to use numerical differentiation
    static bool num_diff;
};
}
