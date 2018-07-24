#include "utils.h"
#include <glog/logging.h>

namespace ev{
parameterReader::parameterReader(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        LOG(ERROR)<< "no config file found at " << filename;
    }
    std::string s;

    std::getline(file, s, ' ');
    parameters.fx = std::stof(s);
    parameters.K.at<float>(0, 0) = parameters.fx;

    std::getline(file, s, ' ');
    parameters.fy = std::stof(s);
    parameters.K.at<float>(1, 1) = parameters.fy;

    std::getline(file, s, ' ');
    parameters.cx = std::stof(s);
    parameters.K.at<float>(0, 2) = parameters.cx;

    std::getline(file, s, ' ');
    parameters.cy = std::stof(s);
    parameters.K.at<float>(1, 2) = parameters.cy;
    
    cv::cv2eigen(parameters.K, parameters.K_);

    std::getline(file, s, ' ');
    float k1 = std::stof(s);

    std::getline(file, s, ' ');
    float k2 = std::stof(s);

    std::getline(file, s, ' ');
    float p1 = std::stof(s);

    std::getline(file, s, ' ');
    float p2 = std::stof(s);

    std::getline(file, s, ' ');
    float k3 = std::stof(s);

    parameters.distCoeffs = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);

}

bool parameterReader::getParameter(ev::Parameters& parameter){
    parameter = parameters;
    return true;
}

void imshowRescaled(const cv::Mat &src, int msec, std::string title, double* text) {

    cv::Mat dst;
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    cv::subtract(src, cv::Mat(src.rows, src.cols, CV_64F, cv::Scalar(min)), dst);

    dst /= (max - min);

//    if (text != NULL) {
//           for (int i = 0; i != 12; i++) {
//               auto divresult = div(i, 3);
//               std::string depth = std::to_string(text[i]);
//               auto pos = depth.find(".");
//               cv::putText(dst, depth.substr(0, pos + 2), cvPoint(divresult.quot * 60 + 20, divresult.rem * 60 + 30),
//                           cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
//           }
//               cv::line(dst, cvPoint(60, 0), cvPoint(60, 180), cvScalar(200, 200, 250), 1);
//               cv::line(dst, cvPoint(120, 0), cvPoint(120, 180), cvScalar(200, 200, 250), 1);
//               cv::line(dst, cvPoint(180, 0), cvPoint(180, 180), cvScalar(200, 200, 250), 1);
//               cv::line(dst, cvPoint(0, 60), cvPoint(240, 60), cvScalar(200, 200, 250), 1);
//               cv::line(dst, cvPoint(0, 120), cvPoint(240, 120), cvScalar(200, 200, 250), 1);
//               cv::line(dst, cvPoint(0, 179), cvPoint(240, 180), cvScalar(200, 200, 250), 1);
//       }

   // std::string file_name = title + "_" + std::to_string(count) + ".jpg";
    std::string file_name = title;
    cv::imshow(file_name, dst);
    cv::waitKey(msec);

}

void imwriteRescaled(const cv::Mat &src, std::string title, double* text) {
    cv::Mat dst;
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    cv::subtract(src, cv::Mat(src.rows, src.cols, CV_64F, cv::Scalar(min)), dst);
    dst /= (max - min);
    dst *= 255;
    std::string file_name = title;
    cv::imwrite(file_name, dst);
}


void imshowRescaled(Eigen::MatrixXd &src_, int msec, std::string title, double *text) {
    cv::Mat src;
    cv::eigen2cv(src_, src);
    imshowRescaled(src, msec, title, text);
}

void imshowRescaled(Eigen::SparseMatrix<double> &src_, int msec, std::string title, double* text) {
    Eigen::MatrixXd src = src_;
    imshowRescaled(src, msec, title, text);
}

void quat2eul(Eigen::Quaterniond& q, double* euler) {
    double y2 = q.y() * q.y();
    euler[0] = std::atan2(2*(q.w()*q.z() + q.x()*q.y()), (1 - 2*(y2 + q.z()*q.z())));
    euler[1] = std::asin( 2*(q.w()*q.y() - q.z()*q.x()));
    euler[2] = std::atan2(2*(q.w()*q.x() + q.y()*q.z()), (1 - 2*(q.x()*q.x() + y2)));
}

Eigen::Matrix3d skew(Eigen::Vector3d v){
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return m;
}

cv::Mat axang2rotm(const cv::Mat& w) {
    double angle = cv::norm(w);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);;
    if (std::abs(angle) > EPS) {
        cv::Mat axis = w / angle;
        cv::Mat K = skew(axis);
        R += std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
    }
    return R;
}

Eigen::Matrix3d axang2rotm(const Eigen::Vector3d& w) {
    double angle = w.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (std::abs(angle) > EPS) {
        Eigen::Vector3d axis = w / angle;
        Eigen::Matrix3d K = skew(axis);
        R += std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
    }
    return R;
}

cv::Mat rotm2axang(const cv::Mat& R) {
    double cos_angle = (cv::trace(R).val[0] - 1)/2;
    double angle;
    cv::Mat axis = cv::Mat::zeros(3, 1, CV_64F);
    if (std::abs(cos_angle - 1) < EPS)
        return axis;
    if (std::abs(cos_angle + 1) < EPS) {
        angle = M_PI;
        double xx = (R.at<double>(0, 0) + 1) / 2;
        double yy = (R.at<double>(1, 1) + 1) / 2;
        double zz = (R.at<double>(2, 2) + 1) / 2;
        double xy = (R.at<double>(0, 1) + R.at<double>(1, 0)) / 4;
        double xz = (R.at<double>(0, 2) + R.at<double>(2, 0)) / 4;
        double yz = (R.at<double>(1, 2) + R.at<double>(2, 1)) / 4;
        double x, y, z;
        if ((xx > yy) && (xx > zz)) {
            if (xx< EPS) {
                x = 0;
                y = std::sqrt(.5);
                z = std::sqrt(.5);
            } else {
                x = std::sqrt(xx);
                y = xy / x;
                z = xz / x;
            }
        } else if (yy > zz) {
            if (yy < EPS) {
                x = std::sqrt(.5);
                y = 0;
                z = std::sqrt(.5);
            } else {
                y = std::sqrt(yy);
                x = xy / y;
                z = yz / y;
            }
        } else {
            if (zz < EPS) {
                x = std::sqrt(.5);
                y = std::sqrt(.5);
                z = 0;
            } else {
                z = std::sqrt(zz);
                x = xz / z;
                y = yz / z;
            }
        }
        axis.at<double>(0) = x;
        axis.at<double>(1) = y;
        axis.at<double>(2) = z;
        axis *= angle;
    } else {
       angle = std::acos(cos_angle);
       axis.at<double>(0) = R.at<double>(2, 1) - R.at<double>(1, 2);
       axis.at<double>(1) = R.at<double>(0, 2) - R.at<double>(2, 0);
       axis.at<double>(2) = R.at<double>(1, 0) - R.at<double>(0, 1);
       axis /= (2 * std::sin(angle));
       axis *= angle;
    }
    return axis;
}

void rotateAngleByQuaternion(double* p, Eigen::Quaterniond q, double* p_) {
    Eigen::Vector3d direction(std::cos(p[0]) * std::sin(p[1]), std::sin(p[0]) * std::sin(p[1]), std::cos(p[1]));
    Eigen::Vector3d direction_after = q.toRotationMatrix() * direction;
    p_[1] = std::acos(direction_after(2));
    p_[0] = std::atan(direction_after(1)/direction(0));
    if (std::cos(p_[0]) * std::sin(p_[1]) * direction(0) < 0)
        p_[0] += M_PI;

//    LOG(INFO) << std::cos(p_[0]) * std::sin(p_[1]) << " " << direction_after(0);
//    LOG(INFO) << std::sin(p_[0]) * std::sin(p_[1]) << " " << direction_after(1);
//    LOG(INFO) << std::cos(p_[1]) << " " << direction_after(2);
}
}