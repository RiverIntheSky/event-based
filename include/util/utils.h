#ifndef UTILS_H
#define UTILS_H
#include "parameters.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <okvis/Measurements.hpp>

namespace ev {
class parameterReader
{
public:
    parameterReader(const std::string& filename);
    bool getParameter(ev::Parameters& parameter);

    ev::Parameters parameters;
};

struct Pose {
    /// \brief Default constructor.
    Pose(): p(), q() {}

    /// \brief Constructor.
    Pose(Eigen::Vector3d p_, Eigen::Quaterniond q_)
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
void imshowRescaled(const cv::Mat& src, int msec = 0, std::string title = "image", std::string text = "");
void imshowRescaled(Eigen::MatrixXd &src_, int msec = 0, std::string title = "image", std::string text = "");
void quat2eul(Eigen::Quaterniond& q, double* euler);
Eigen::Matrix3d skew(Eigen::Vector3d& v);
}

#endif // UTILS_H
