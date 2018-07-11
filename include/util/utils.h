#ifndef UTILS_H
#define UTILS_H
#include "parameters.h"
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <okvis/Measurements.hpp>
#include <Eigen/Sparse>
#include "glog/logging.h"
#include <chrono>

namespace ev {
extern int count;
class parameterReader
{
public:
    parameterReader(const std::string& filename);
    bool getParameter(ev::Parameters& parameter);

    ev::Parameters parameters;
};

struct Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /// \brief Default constructor.
    Pose(): p(), q() {}

    /// \brief Constructor.
    Pose(Eigen::Vector3d& p_, Eigen::Quaterniond& q_)
        : p(p_), q(q_) {}

//    /// \brief Copy constructor.
//    Pose(ev::Pose& po)
//        : p(po.p), q(po.q) {}

    void operator() (Eigen::Vector3d& p_, Eigen::Quaterniond& q_) {
        p = p_;
        q = q_;
    }

// private:
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
};

typedef okvis::Measurement<Pose> MaconMeasurement;

// map entries in src to [0, 1]
void imshowRescaled(const cv::Mat& src, int msec = 0, std::string title = "image", double* text = NULL);
void imshowRescaled(Eigen::MatrixXd &src_, int msec = 0, std::string title = "image", double* text = NULL);
void imshowRescaled(Eigen::SparseMatrix<double> &src_, int msec = 0, std::string title = "image", double *text = NULL);
void imwriteRescaled(const cv::Mat &src, std::string title = "image", double* text = NULL);
void quat2eul(Eigen::Quaterniond& q, double* euler);
Eigen::Matrix3d skew(Eigen::Vector3d v);

inline cv::Mat skew(cv::Mat v){
    return  (cv::Mat_<double>(3, 3) << 0, -v.at<double>(2), v.at<double>(1),
                                      v.at<double>(2), 0, -v.at<double>(0),
                                     -v.at<double>(1), v.at<double>(0), 0);
}

inline int truncate(int value, int min_value, int max_value) {
    return std::min(std::max(value, min_value), max_value);
}

cv::Mat axang2rotm(const cv::Mat& w);
Eigen::Matrix3d axang2rotm(const Eigen::Vector3d& w);

void rotateAngleByQuaternion(double* p, Eigen::Quaterniond q, double* p_);
}

#endif // UTILS_H
