#include "Frontend.h"

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <Eigen/Geometry>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>



namespace ev {

bool ComputeVarianceFunction::Evaluate(double const* const* parameters,
                                       double* residuals,
                                       double** jacobians) const {


    int area = 180 * 240;
    residuals[0] = 0;
    Eigen::Vector3d w;
    w << (*parameters)[0], (*parameters)[1], (*parameters)[2];
    Eigen::Vector3d v;
    v << (*(parameters + 1))[0], (*(parameters + 1))[1], (*(parameters + 1))[2];
    const double* const* z = parameters + 2;
    int patch_num = 12;

    // derivatives of intensity with respect to each parameters
    std::vector<Eigen::SparseMatrix<double>> dIdw_;
    // the mean of the derivatives
    std::vector<double> dIm;

    // auto start = Clock::now();

    if (jacobians != NULL && jacobians[0] != NULL) {

        for (int i = 0; i != 2; i++) {
            for (int j = 0; j != 3; j++) {
                jacobians[i][j] = 0;
            }
        }
        for (int i = 0; i != patch_num; i++) {
            jacobians[2][i] = 0;
        }

        Eigen::SparseMatrix<double> dIdw(param_.array_size_y * param_.array_size_x, 6 + patch_num);
        Intensity(intensity, &dIdw, w, v, z);
        /*LOG(INFO) << "derivative & intensity " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
                  << " nanoseconds";*/
        cv::Mat src, dst;

        for (int i = 0; i != 6 + patch_num; i++){
            Eigen::MatrixXd d = dIdw.col(i);
            d.resize(180, 240);
            cv::eigen2cv(d, src);
            cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
            cv::cv2eigen(dst, d);
            dIdw_.push_back(d.sparseView());
            dIm.push_back(d.mean());
        }

    } else {
        Intensity(intensity, NULL, w, v, z);
        /*LOG(INFO) << "intensity " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
                  << " nanoseconds";*/
    }

    // start = Clock::now();
    // intensity.sum() somehow incorrect?
    double Im = (intensity * Eigen::VectorXd::Ones(intensity.cols())).sum() / area;

    for (int s = 0; s < intensity.outerSize(); ++s) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(intensity, s); it; ++it) {
            double rho = it.value() - Im;
            residuals[0] += std::pow(rho, 2);

            int y_ = it.row();
            int x_ = it.col();

            if (jacobians != NULL && jacobians[0] != NULL) {
                for (int i = 0; i != 3; i++) {
                    jacobians[0][i] += rho * ((dIdw_[i]).coeffRef(y_, x_) - dIm[i]);
                }
                for (int i = 0; i != 3; i++) {
                    jacobians[1][i] += rho * ((dIdw_[i + 3]).coeffRef(y_, x_) - dIm[i + 3]);
                }
                for (int i = 0; i != patch_num; i++) {
                    jacobians[2][i] += rho * ((dIdw_[i + 6]).coeffRef(y_, x_) - dIm[i + 6]);
                }

            }
        }
    }
    residuals[0] += (area - intensity.nonZeros()) * std::pow(Im, 2);

    residuals[0] /= area;
    residuals[0] = 1./residuals[0];

    if (jacobians != NULL && jacobians[0] != NULL) {
        for (int i = 0; i != 3; i++) {
            jacobians[0][i] += (area - intensity.nonZeros()) * Im * dIm[i];
            jacobians[0][i] *= (-2 * std::pow(residuals[0], 2) / area);
//            LOG(INFO) << "jw:" << jacobians[0][i];
//            jacobians[0][i] = 0;
        }
        for (int i = 0; i != 3; i++) {
            jacobians[1][i] += (area - intensity.nonZeros()) * Im * dIm[i + 3];
            jacobians[1][i] *= (-2 * std::pow(residuals[0], 2) / area);
//            LOG(INFO) << "jv:" << jacobians[1][i];
//            jacobians[1][i] = 0;
        }
        for (int i = 0; i != patch_num; i++) {
            jacobians[2][i] += (area - intensity.nonZeros()) * Im * dIm[i + 6];
            jacobians[2][i] *= (-2 * std::pow(residuals[0], 2) / area);
//            LOG(INFO) << "jz:" << jacobians[2][i];
//            jacobians[2][i]  = 1000;
        }
    }
//    if (jacobians != NULL && jacobians[0] != NULL) {
//        LOG(INFO) << "jacobian & residual " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
//                  << " nanoseconds";
//    } else {
//        LOG(INFO) << "residual " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
//                  << " nanoseconds";
//    }

    return true;
}

void ComputeVarianceFunction::warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_v, Eigen::Vector3d& x,
                                   okvis::Duration& t, Eigen::Vector3d& w, Eigen::Vector3d& v, const double z) const {
    // z is the depth value of the first frame
    double t_ = t.toSec();
    Eigen::Vector3d x_w;
    if (w.norm() == 0 || t_ == 0) {
        x_w = x;
    } else {
        x_w = x + (w * t_).cross(x);
    }
    x_v = (z - v(2) * t_) * x_w / x_w(2) + v * t_;

    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
    // 0.03 heuristic
//    cameraMatrix_(0, 0) *= (z + 0.1 * w(2)) / z;
//    cameraMatrix_(1, 1) *= (z + 0.1 * w(2)) / z;

    if (dW != NULL) {
//        Eigen::MatrixXd dWdT = (cameraMatrix_ * (Eigen::Matrix3d::Identity()*t_-x_w*(Eigen::Vector3d() << 0, 0, t_).finished().transpose()/x_w(2))/x_w(2)).block(0, 0, 2, 3);

        Eigen::Vector3d dWdw1;
        dWdw1 << -(1 - v(2) * t_ / z) * x_w(0) * x(1) / x_w(2),
                 -(1 - v(2) * t_ / z) * (x_w(1) * x(1) / x_w(2) + 1),
                 0;
        Eigen::Vector3d dWdw2;
        dWdw2 << (1 - v(2) * t_ / z) * (x_w(0) * x(0) / x_w(2) + 1),
                 (1 - v(2) * t_ / z) * x_w(1) * x(0) / x_w(2),
                 0;
        Eigen::Vector3d dWdw3;
        dWdw3 << -(1 - v(2) * t_ / z) * x(1),
                  (1 - v(2) * t_ / z) * x(0),
                  0;
        Eigen::Vector3d dWdTx;
        dWdTx << 1, 0, 0;
        Eigen::Vector3d dWdTy;
        dWdTy << 0, 1, 0;
        Eigen::Vector3d dWdTz;
        dWdTz << -x(0), -x(1), 0;
        //        dWdT.block(0, 2, 2, 1) += 0.03 / z * (Eigen::Vector2d() << x_w(0)/x_w(2), x_w(1)/x_w(2)).finished();
        Eigen::Vector3d dWdz =  (v(2) * x_w / x_w(2) - v) / z;
//        dWdz -= 0.03 * w(2) / (z * z) * (Eigen::Vector2d() << x_w(0)/x_w(2), x_w(1)/x_w(2)).finished();
        Eigen::MatrixXd dW_(3, 7);
        dW_ << dWdw1, dWdw2, dWdw3, dWdTx, dWdTy, dWdTz, dWdz;
        (*dW) = (cameraMatrix_ * t_ / z * dW_).block(0, 0, 2, 7);
    }
    x_v = cameraMatrix_ * x_v;
}

void ComputeVarianceFunction::fuse(Eigen::SparseMatrix<double>& image, Eigen::Vector2d& p, bool& polarity) const {
    std::vector<std::pair<std::vector<int>, double>> pixels;
    biInterp(pixels, p, polarity);
    for (auto p_it = pixels.begin(); p_it != pixels.end(); p_it++) {
        image.coeffRef((p_it->first)[1], (p_it->first)[0]) += p_it->second;
    }
}

void ComputeVarianceFunction::biInterp(std::vector<std::pair<std::vector<int>, double>>& pixel_weight, Eigen::Vector2d& point, bool& polarity) const {
    auto valid = [](int x, int y)  -> bool {
        return (x >= 0 && x  < 240 && y >= 0 && y < 180);
    };

    pixel_weight.clear();

    int pol = int(polarity) * 2 - 1;

    int x1 = std::floor(point(0));
    int x2 = x1 + 1;
    int y1 = std::floor(point(1));
    int y2 = y1 + 1;

    if (valid(x1, y1)) {
        double a = (x2 - point(0)) * (y2 - point(1)) * pol;
        std::vector<int> p = {x1, y1};
        pixel_weight.push_back(std::make_pair(p, a));
    }

    if (valid(x1, y2)) {
        double a = - (x2 - point(0)) * (y1 - point(1)) * pol;
        std::vector<int> p = {x1, y2};
        pixel_weight.push_back(std::make_pair(p, a));
    }

    if (valid(x2, y1)) {
        double a = - (x1 - point(0)) * (y2 - point(1)) * pol;
        std::vector<int> p = {x2, y1};
        pixel_weight.push_back(std::make_pair(p, a));
    }

    if (valid(x2, y2)) {
        double a = (x1 - point(0)) * (y1 - point(1)) * pol;
        std::vector<int> p = {x2, y2};
        pixel_weight.push_back(std::make_pair(p, a));
    }
}

void ComputeVarianceFunction::Intensity(Eigen::SparseMatrix<double>& image, Eigen::SparseMatrix<double>* dIdw,
                                        Eigen::Vector3d& w, Eigen::Vector3d& v, const double* const* z) const {
    image = Eigen::SparseMatrix<double>(180, 240);
    okvis::Time t0 = em_->events.front().timeStamp;
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);

    // helper function, truncates value outside a range
    auto truncate = [](int value, int min_value, int max_value) -> int {
        return std::min(std::max(value, min_value), max_value);
    };

    // divide scene into patches
    auto& param = param_;
    auto patch = [&param, cameraMatrix_, truncate](Eigen::Vector3d p)  -> int {
        Eigen::Vector3d p_ = cameraMatrix_ * p;
        int patch_num_x = std::ceil(param.array_size_x / param.patch_width);
        int patch_num_y = std::ceil(param.array_size_y / param.patch_width);
        return truncate(std::floor(p_(0)/param.patch_width), 0, patch_num_y-1) * patch_num_y
                + truncate(std::floor(p_(1)/param.patch_width), 0, patch_num_x-1);
    };
    for(auto it = em_->events.begin(); it != em_->events.end(); it++) {
        Eigen::Vector3d p(it->measurement.x, it->measurement.y, it->measurement.z);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = it->timeStamp - t0;
        Eigen::MatrixXd dW;
        if (dIdw != NULL) {
            //start = Clock::now();
            warp(&dW, point_warped, p, t, w, v, (*z)[patch(p)]);
            /*LOG(INFO) << "warp " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
                      << " nanoseconds";*/
//            if (patch(p) == 0)
//                dW.col(6) = Eigen::Vector2d::Zero();
        } else {
            warp(NULL, point_warped, p, t, w, v, (*z)[patch(p)]);
        }

        // z'/z
        point_warped /= point_warped(2);
        Eigen::Vector2d point_camera_(point_warped(0), point_warped(1));
        if (dIdw != NULL) {

            Eigen::MatrixXd dWdwv = dW.block(0, 0, 2, 6);
            Eigen::Vector2d dWdz = dW.col(6);
            std::vector<std::pair<std::vector<int>, double>> pixel_weight;
            biInterp(pixel_weight, point_camera_, it->measurement.p);
            for (auto p_it = pixel_weight.begin(); p_it != pixel_weight.end(); p_it++) {
                std::vector<int> p_ = p_it->first;
                int patch_nr = patch(p);
                double dc = point_warped(0) - p_[0];
                double dr = point_warped(1) - p_[1];
                Eigen::VectorXd dwv = Eigen::VectorXd::Zero(6);
                double dz_ = 0.;
                int pol = int(it->measurement.p) * 2 - 1;
                //start = Clock::now();

                double dW;
                if (dr > 0) {
                    dW = (std::abs(dc) - 1) * pol;
                } else if (dr < 0) {
                    dW = (1 - std::abs(dc)) * pol;
                }
                dwv += dW * dWdwv.row(1);
                dz_ += dW * dWdz(1);
                if (dc > 0) {
                    dW = (std::abs(dr) - 1) * pol;
                } else if (dc < 0) {
                    dW = (1 - std::abs(dr)) * pol;
                }
                dwv += dW * dWdwv.row(0);
                dz_ += dW * dWdz(0);

                for (int i = 0; i != 6; i++) {
                    dIdw->coeffRef(p_[0] * 180 + p_[1], i) += dwv(i);
                }                
                dIdw->coeffRef(p_[0] * 180 + p_[1], 6 + patch_nr) += dz_;

                image.coeffRef(p_[1], p_[0]) += p_it->second;
                /*LOG(INFO) << "dwdwv " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()
                                                                                   << " nanoseconds";*/
            }

        } else {
            fuse(image, point_camera_, it->measurement.p);
        }
    }

    Eigen::MatrixXd image_ = image.toDense();
    cv::Mat src, dst;
    cv::eigen2cv(image_, src);
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image_);
    image = image_.sparseView();

}

void Contrast::warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration& t, Eigen::Vector3d& w) {
    Eigen::Vector3d x_ = x.homogeneous();
    if (w.norm() == 0) {
        //x_w = x_;
//        w(0) = 2;
        x_w = x_ + w * t.toSec();
    } else {
        Eigen::AngleAxisd aa(w.norm()* t.toSec(), w.normalized());
        x_w = aa.toRotationMatrix() * x_;
    }
    // x_w = x_ + (w * t.toSec()).cross(x_);
    x_w /= x_w(2);
}

void SE3::warp(Eigen::Vector3d& x_w, Eigen::Vector2d& x, okvis::Duration &t, Eigen::Vector3d& w, Eigen::Vector3d& tr) {
    Eigen::Vector3d x_ = x.homogeneous();
    if (w.norm() == 0) {
        x_w = x_ + tr * t.toSec();
    } else {
        Eigen::AngleAxisd aa(w.norm()* t.toSec(), w.normalized());
        x_w = aa.toRotationMatrix() * x_ + tr * t.toSec();
    }
    // x_w(2) = z_w/z;
    x_w /= x_w(2);
}

void Contrast::fuse(Eigen::MatrixXd& image, Eigen::Vector2d p, bool& polarity) {
#if 1
    int pol = int(polarity) * 2 - 1;
#else
    int pol = 1;
#endif

    Eigen::Vector2d p1(std::floor(p(0)), std::floor(p(1)));
    Eigen::Vector2d p2(p1(0), p1(1) + 1);
    Eigen::Vector2d p3(p1(0) + 1, p1(1));
    Eigen::Vector2d p4(p1(0) + 1, p1(1) + 1);

    double a1 = (p4(0) - p(0)) * (p4(1) - p(1));
    double a2 = -(p3(0) - p(0)) * (p3(1) - p(1));
    double a3 = -(p2(0) - p(0)) * (p2(1) - p(1));
    double a4 =  (p1(0) - p(0)) * (p1(1) - p(1));

    image(p1(1), p1(0)) += a1 * pol;
    image(p2(1), p2(0)) += a2 * pol;
    image(p3(1), p3(0)) += a3 * pol;
    image(p4(1), p4(0)) += a4 * pol;
}

void Contrast::synthesizeEventFrame(Eigen::MatrixXd &frame, std::shared_ptr<eventFrameMeasurement>& em) {
    Intensity(frame, em_, param_, Eigen::Vector3d::Zero());
}

void Contrast::polarityEventFrame(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em, bool po) {
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        if (it->measurement.p == po) {
            Eigen::Vector2d p(it->measurement.x, it->measurement.y);
            Eigen::Vector3d point_camera = cameraMatrix_ * p.homogeneous();

            // discard point outside frustum
            if (point_camera(0) > 0 && point_camera(0) < 179
                    && point_camera(1) > 0 && point_camera(1) < 239) {
                fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
            }
        }
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    int kernelSize = 3;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}

void Contrast::Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement> &em, Parameters& param, Eigen::Vector3d w) {
    okvis::Time t0 = em->events.front().timeStamp;
    Eigen::Matrix3d cameraMatrix_;
    cv::cv2eigen(param.cameraMatrix, cameraMatrix_);
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = it->timeStamp - t0;
        warp(point_warped, p, t, w);

        Eigen::Vector3d point_camera = cameraMatrix_ * point_warped;
        if (point_camera(0) > 0 && point_camera(0) < 239
                && point_camera(1) > 0 && point_camera(1) < 179) {
            fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
        } else {
            // LOG(INFO) << "discard point outside frustum";
        }
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    int kernelSize = 3;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}

//void Contrast::Intensity(Eigen::MatrixXd& image, Eigen::Vector3d& w, Eigen::Vector3d& v, double* z) {
//    image = Eigen::MatrixXd::Zero(param_.array_size_y, param_.array_size_x);
//    okvis::Time t0 = em_->events.front().timeStamp;
//    Eigen::Matrix3d cameraMatrix_;
//    cv::cv2eigen(param_.cameraMatrix, cameraMatrix_);
//    for(auto it = em_->events.begin(); it != em_->events.end(); it++) {
//        Eigen::Vector3d p(it->measurement.x, it->measurement.y, it->measurement.z);
//        Eigen::Vector3d point_warped;

//        // project to last frame
//        auto t = it->timeStamp - t0;
//        ComputeVarianceFunction::warp(NULL, point_warped, p, t, w, v, *z);

//        // z'/z
//        point_warped /= point_warped(2);
//        Eigen::Vector2d point_camera_(point_warped(0), point_warped(1));

//        fuse(image, point_camera_, it->measurement.p);

//    }
//    cv::Mat src, dst;
//    cv::eigen2cv(image, src);
//    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
//    cv::cv2eigen(dst, image);

//}

void SE3::Intensity(Eigen::MatrixXd& image, std::shared_ptr<eventFrameMeasurement>& em,
                    Parameters& param, Eigen::Vector3d& w, Eigen::Vector3d& tr) {
    okvis::Time t1 = em->events.back().timeStamp;
    for(auto it = em->events.begin(); it != em->events.end(); it++) {
        Eigen::Vector2d p(it->measurement.x, it->measurement.y);
        Eigen::Vector3d point_warped;

        // project to last frame
        auto t = t1 - it->timeStamp;
        warp(point_warped, p, t, w, tr);
        Eigen::Matrix3d cameraMatrix_;
        cv::cv2eigen(param.cameraMatrix, cameraMatrix_);
        Eigen::Vector3d point_camera = cameraMatrix_ * point_warped;
        if (point_camera(0) > 0 && point_camera(0) < 179
                && point_camera(1) > 0 && point_camera(1) < 239) {
            fuse(image, Eigen::Vector2d(point_camera(0), point_camera(1)), it->measurement.p);
        } else {
            // LOG(INFO) << "discard point outside frustum";
        }
    }
    cv::Mat src, dst;
    cv::eigen2cv(image, src);
    int kernelSize = 3;
    cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0, 0);
    cv::cv2eigen(dst, image);
}

double Contrast::getIntensity(int x, int y, Eigen::Vector3d& w) {
    if (x == 0 && y == 0) {
        intensity = Eigen::MatrixXd::Zero(180, 240);
        Intensity(intensity, em_, param_, w);
    }
    return intensity(x, y);
}

double SE3::getIntensity(int x, int y, Eigen::Vector3d& w, Eigen::Vector3d& tr) {
    if (x == 0 && y == 0) {
        intensity = Eigen::MatrixXd::Zero(180, 240);
        Intensity(intensity, em_, param_, w, tr);
    }
    return intensity(x, y);
}

// Constructor.
Frontend::Frontend()
    : isInitialized_(false),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(
          std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2) {
    initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(
        std::shared_ptr<okvis::MultiFrame> frameOut,
        const okvis::kinematics::Transformation& T_WC,
        const std::vector<cv::KeyPoint> * keypoints) {
    return true;
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
        okvis::Estimator& estimator,
        okvis::kinematics::Transformation& /*T_WS_propagated*/, // TODO sleutenegger: why is this not used here?
        const ev::Parameters & params,
        const std::shared_ptr<okvis::MapPointVector> /*map*/, // TODO sleutenegger: why is this not used here?
        std::shared_ptr<okvis::MultiFrame> framesInOut,
        bool *asKeyframe) {
    return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const ev::Parameters& Params,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const {

    return false;
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::needANewKeyframe(
        const okvis::Estimator& estimator,
        std::shared_ptr<okvis::MultiFrame> currentFrame) {

    return true;
}

// Match a new multiframe to existing keyframes
template<class MATCHING_ALGORITHM>
int Frontend::matchToKeyframes(okvis::Estimator& estimator,
                               const okvis::VioParameters & params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers) {
    int retCtr = 0;

    return retCtr;
}

// Match a new multiframe to the last frame.
template<class MATCHING_ALGORITHM>
int Frontend::matchToLastFrame(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers) {

    int retCtr = 0;
    return retCtr;
}

// Perform 3D/2D RANSAC.
int Frontend::runRansac3d2d(okvis::Estimator& estimator,
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers) {
    int numInliers = 0;
    return numInliers;
}

// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2d(okvis::Estimator& estimator,
                            const okvis::VioParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly) {
    return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void Frontend::initialiseBriskFeatureDetectors() {
    featureDetectorMutex_.lock();

    featureDetector_.clear();
    descriptorExtractor_.clear();

    featureDetector_ = brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
                briskDetectionThreshold_, briskDetectionOctaves_,
                briskDetectionAbsoluteThreshold_,
                briskDetectionMaximumKeypoints_);

    descriptorExtractor_= brisk::BriskDescriptorExtractor(
                briskDescriptionRotationInvariance_,
                briskDescriptionScaleInvariance_);

    featureDetectorMutex_.unlock();
}

}
