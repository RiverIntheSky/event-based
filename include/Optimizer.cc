#include "Optimizer.h"
#include "util/utils.h"


namespace ev {

Eigen::Matrix3d Optimizer::mPatchProjectionMat = Eigen::Matrix3d::Identity();
Eigen::Matrix3d Optimizer::mCameraProjectionMat = Eigen::Matrix3d::Identity();
bool Optimizer::inFrame = true;
bool Optimizer::toMap = false;
int Optimizer::sigma = 1;
int Optimizer::count_frame = 0;
int Optimizer::count_map = 0;
int Optimizer::height = 500;
int Optimizer::width = 500;

//double Optimizer::variance_map(const gsl_vector *vec, void *params) {
//    MapPoint* pMP = (MapPoint *) params;
//    cv::Mat src, dst;

//    // this has to be adjusted according to viewing angle!!
//    double psi = gsl_vector_get(vec, 1);
//    if (std::cos(psi) > 0)
//        return 0.;

//    // image = pMP->mBack;
//    intensity(src, vec, pMP);
//    cv::GaussianBlur(src, dst, cv::Size(0, 0), sigma, 0);
////    imwriteRescaled(dst, "/home/weizhen/Documents/dataset/shapes_translation/map_reloc/4000/back_" + std::to_string(count_frame) + ".jpg", NULL);
//    imshowRescaled(src, 1, "back_buffer");
//    cv::Scalar mean, stddev;
//    cv::meanStdDev(dst, mean, stddev);
//    double cost = -std::pow(stddev[0], 2);
//    src.copyTo(pMP->mBack);

//    return cost;
//}


//double Optimizer::variance_ba(const gsl_vector *vec, void *params) {
//    mapPointAndKeyFrames* mkf = (mapPointAndKeyFrames *) params;
//    cv::Mat src;

//    intensity(src, vec, mkf);
//    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
//    imshowRescaled(src, 1);
//    cv::Scalar mean, stddev;
//    cv::meanStdDev(src, mean, stddev);
//    double cost = -std::pow(stddev[0], 2);

//    return cost;
//}

//double Optimizer::variance_relocalization(const gsl_vector *vec, void *params) {
//    mapPointAndFrame* mf = (mapPointAndFrame *) params;
//    cv::Mat src;

//    intensity_relocalization(src, vec, mf);
//    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
//     imshowRescaled(src, 1);

//    cv::Scalar mean, stddev;
//    cv::meanStdDev(src, mean, stddev);
//    double cost = -std::pow(stddev[0], 2);

//    return cost;
//}

void Optimizer::warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                     const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& Rn, const Eigen::Matrix3d& H_) {
    // plane homography
    // v is actually v/dc
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
//    Eigen::Matrix3d H = R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose());
//    x_w = Rn * H_ * H.inverse() * x;
    x_w = R.inverse() * x;
    if (dW) {
        Eigen::MatrixXd dW_ = -t * mPatchProjectionMat * skew(x);
        *dW = (dW_ - x_w * dW_.row(2) / x_w(2)) / x_w(2);
        LOG(INFO) << *dW ;
    }
    x_w /= x_w(2);
    x_w = mPatchProjectionMat * x_w;
}

void Optimizer::warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                     const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& H_) {
    // plane homography
    // v is actually v/dc
//    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
//    Eigen::Matrix3d T = Eigen::Matrix3d::Identity() + v * t * nc.transpose();
//    Eigen::MatrixXd W = mPatchProjectionMat * H_ * T.inverse();
//    x_w = W * R.transpose() * x;
////    if (dW) {
////        Eigen::MatrixXd dW_ = Eigen::MatrixXd::Zero(dW->rows(), dW->cols());
////        dW_.block(0, 0, 3, 3) = -t * W * skew(x);
////        dW_.block(0, 3, 3, 3) = -t * nc.transpose() * R.inverse() * x/ (t * nc.transpose() * v + 1)  * W;
////        *dW = (dW_ - x_w * dW_.row(2) / x_w(2)) / x_w(2);
////    }
//    x_w /= x_w(2);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
    Eigen::Matrix3d H = R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose());
    x_w = H_ * H.inverse() * x;
    x_w /= x_w(2);
    x_w = mPatchProjectionMat * x_w;
}

void Optimizer::warp(Eigen::MatrixXd* dW, Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                     const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& H_, const Eigen::Vector2d& n) {
    // plane homography
    // v is actually v/dc
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity() + v * t * nc.transpose();
    Eigen::MatrixXd W = mPatchProjectionMat * H_ * T.inverse();
    x_w = W * R.transpose() * x;
    if (dW) {
//        LOG(INFO) << dW;
//        LOG(INFO) << dW->rows();
//        LOG(INFO) << dW->cols();
        dW->block(0, 0, 3, 3) = -t * W * skew(x);
        dW->block(0, 3, 3, 3) =  W * (-t * nc.transpose() * R.inverse() * x/ (t * nc.transpose() * v + 1))[0] ;

        Eigen::MatrixXd dndp(3, 2);
        dndp << -std::sin(n(0)) * std::sin(n(1)),  std::cos(n(0)) * std::cos(n(1)),
                 std::cos(n(0)) * std::sin(n(1)),  std::sin(n(0)) * std::cos(n(1)),
                 0,  -std::sin(n(1));
        dW->block(0, 6, 3, 2) = (-t * v.transpose() * R.inverse() * x/ (t * nc.transpose() * v + 1))[0] * W * dndp;

        *dW = (*dW - x_w * (*dW).row(2) / x_w(2)) / x_w(2);
    }
    x_w /= x_w(2);
//    LOG(INFO) << x_w;

}

void Optimizer::fuse(Eigen::MatrixXd* dIdW, Eigen::MatrixXd* dW, cv::Mat& image, Eigen::Vector3d& p, bool polarity) {
    // range check
    auto valid = [image](int x, int y)  -> bool {
        return (x >= 0 && x < image.cols && y >= 0 && y < image.rows);
    };

    // change to predefined parameter or drop completely!! (#if)
    int pol = int(polarity) * 2 - 1;
    int x1 = std::floor(p(0));
    int x2 = x1 + 1;
    int y1 = std::floor(p(1));
    int y2 = y1 + 1;

    int nVariables;
    if (dIdW) {
        nVariables = dIdW->cols();
//        LOG(INFO) << dW->row(0);
//        LOG(INFO) << dW->row(1);
//        LOG(INFO) << " ";
    }
//    LOG(INFO) << p(0) << " " << p(1);
//    LOG(INFO) << image.cols << " " << image.rows;

    if (valid(x1, y1)) {
        double a = (x2 - p(0)) * (y2 - p(1)) * pol;
        image.ptr<double>(y1)[x1] += a;
//        LOG(INFO) << x1 << " " << y1;
//        LOG(INFO) << image.ptr<double>(y1)[x1];
//        LOG(INFO) << cv::sum(image)[0];
        if (dIdW) {
            Eigen::VectorXd d = Eigen::VectorXd::Zero(nVariables);
            d += (p(1) - y2) * pol * dW->row(0);
            d += (p(0) - x2) * pol * dW->row(1);
            for (int i = 0; i != nVariables; i++) {
                (*dIdW)(x1 * image.rows + y1, i) += d(i);
            }
        }
    }

    if (valid(x1, y2)) {
        double a = -(x2 - p(0)) * (y1 - p(1)) * pol;
        image.ptr<double>(y2)[x1] += a;
        if (dIdW) {
            Eigen::VectorXd d = Eigen::VectorXd::Zero(nVariables);
            d += (-p(1) + y1) * pol * dW->row(0);
            d += (-p(0) + x2) * pol * dW->row(1);
            for (int i = 0; i != nVariables; i++) {
                (*dIdW)(x1 * image.rows + y2, i) += d(i);
            }
        }
    }

    if (valid(x2, y1)) {
        double a = - (x1 - p(0)) * (y2 - p(1)) * pol;
        image.ptr<double>(y1)[x2] += a;
        if (dIdW) {
            Eigen::VectorXd d = Eigen::VectorXd::Zero(nVariables);
            d += (-p(1) + y2) * pol * dW->row(0);
            d += (-p(0) + x1) * pol * dW->row(1);
            for (int i = 0; i != nVariables; i++) {
                (*dIdW)(x2 * image.rows + y1, i) += d(i);
            }
        }
    }

    if (valid(x2, y2)) {
        double a = (x1 - p(0)) * (y1 - p(1)) * pol;
        image.ptr<double>(y2)[x2] += a;
        if (dIdW) {
            Eigen::VectorXd d = Eigen::VectorXd::Zero(nVariables);
            d += (p(1) - y1) * pol * dW->row(0);
            d += (p(0) - x1) * pol * dW->row(1);
            for (int i = 0; i != nVariables; i++) {
                (*dIdW)(x2 * image.rows + y2, i) += d(i);
            }
        }
    }
}

//void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, MapPoint* pMP) {
//    {

//        lock_guard<mutex> lock(pMP->mMutexFeatures);

//        if (pMP->mFront.empty())
//            pMP->mFront = cv::Mat::zeros(height, width, CV_64F);

//        // reset back buffer
//        pMP->swap(false);

//        // draw to back buffer
//        image = pMP->mBack;

//        // normal direction of map point
//        double phi = gsl_vector_get(vec, 0);
//        double psi = gsl_vector_get(vec, 1);
//        Eigen::Vector3d nw;
//        nw << std::cos(phi) * std::sin(psi),
//             std::sin(phi) * std::sin(psi),
//             std::cos(psi);

//        Eigen::Vector3d z;
//        z << 0, 0, 1;
//        Eigen::Vector3d v = (-nw).cross(z);
//        double c = -z.dot(nw);
//        Eigen::Matrix3d Kn = ev::skew(v);
//        Eigen::Matrix3d Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);

//        // iterate all keyframes
//        auto KFs = pMP->getObservations();

//        Eigen::Matrix3d H_ = Eigen::Matrix3d::Identity();
//        double t;

//        int i = 0;
//        for (auto KFit = KFs.rbegin(); KFit != KFs.rend(); KFit++) {

//            okvis::Time t0 = (*KFit)->mTimeStamp;

//            // normal direction with respect to first pose
//            cv::Mat Rcw = (*KFit)->getRotation().t();

//            Eigen::Vector3d nc = Converter::toMatrix3d(Rcw) * nw;

//            // velocity
//            Eigen::Vector3d w;
//            w << gsl_vector_get(vec, 2 + i * 6),
//                 gsl_vector_get(vec, 3 + i * 6),
//                 gsl_vector_get(vec, 4 + i * 6);

//            Eigen::Vector3d v;
//            v << gsl_vector_get(vec, 5 + i * 6),
//                 gsl_vector_get(vec, 6 + i * 6),
//                 gsl_vector_get(vec, 7 + i * 6);

//            // getEvents()??
//            double theta = -w.norm();

//            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
//            if (theta != 0)
//                K = ev::skew(w.normalized());

//            for (const auto EVit: *((*KFit)->vEvents)) {

//                Eigen::Vector3d p, point_warped;
//                p << EVit->measurement.x ,EVit->measurement.y, 1;

//                // project to first frame
//                t = (EVit->timeStamp - t0).toSec();
//                warp(point_warped, p, t, theta, K, v, nc, Rn, H_);
//                fuse(image, point_warped, EVit->measurement.p);

//            }
//            Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
//            H_ = (R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose())).inverse() * H_;
//            i++;
//        }
//    }
//}

//void Optimizer::intensity_relocalization(cv::Mat& image, const gsl_vector *vec, mapPointAndFrame* mf) {
//    MapPoint* pMP = mf->mP;
//    Frame* frame = mf->frame;

//    // reset back buffer
//    pMP->swap(false);

//    // draw to back buffer
//    // test cv::addWeighted!!
//    image = pMP->mBack;

//    Eigen::Vector3d nw = Converter::toVector3d(pMP->getNormal());

//    // velocity
//    Eigen::Vector3d w, v, Rwc_w, twc, nc;

//    w << gsl_vector_get(vec, 0),
//         gsl_vector_get(vec, 1),
//         gsl_vector_get(vec, 2);

//    v << gsl_vector_get(vec, 3),
//         gsl_vector_get(vec, 4),
//         gsl_vector_get(vec, 5);

//    Rwc_w << gsl_vector_get(vec, 6),
//             gsl_vector_get(vec, 7),
//             gsl_vector_get(vec, 8);

//    Eigen::Matrix3d Rcw = axang2rotm(Rwc_w).transpose();
//    nc = Rcw * nw;

//    twc << gsl_vector_get(vec, 9),
//           gsl_vector_get(vec, 10),
//           gsl_vector_get(vec, 11);

//    Eigen::Matrix3d Rn = pMP->Rn;
//    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

//    okvis::Time t0 = frame->mTimeStamp;
//    double theta = -w.norm();

//    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
//    if (theta != 0)
//        K = ev::skew(w.normalized());

//    for (const auto EVit: frame->vEvents) {

//        Eigen::Vector3d p, point_warped;
//        p << EVit->measurement.x ,EVit->measurement.y, 1;

//        // project to first frame
//        warp(point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_);
//        fuse(image, point_warped, false);
//    }
//}

//void Optimizer::optimize(MapPoint* pMP) {
//    // for one map point and n keyframes, variable numbers 2 + nKFs * 6

//    mPatchProjectionMat(0, 0) = 200;
//    mPatchProjectionMat(1, 1) = 200;
//    mPatchProjectionMat(0, 2) = 250;
//    mPatchProjectionMat(1, 2) = 250;

//    auto kf = pMP->getObservations().end();
////    int nVariables = 2 + nKFs * 6;
//    int nVariables = 3;

//    gsl_multimin_fdfminimizer *s = NULL;
//    gsl_vector *x;
//    double result[nVariables] = {};

//    /* Starting point */
//    x = gsl_vector_alloc(nVariables);

//    // Normal direction of the plane
////    gsl_vector_set(x, 0, pMP->getNormalDirection().at(0));
////    gsl_vector_set(x, 1, pMP->getNormalDirection().at(1));

//    int i = 0;
////    for (auto KFit = KFs.rbegin(); KFit != KFs.rend(); KFit++) {
//        cv::Mat w = (*kf)->getAngularVelocity();
//        gsl_vector_set(x, i++, w.at<double>(0));
//        gsl_vector_set(x, i++, w.at<double>(1));
//        gsl_vector_set(x, i++, w.at<double>(2));

//        cv::Mat v = (*kf)->getLinearVelocity()/* / (*KFit)->mScale*/;
////        cv::Mat v = (*KFit)->getLinearVelocity();
//        gsl_vector_set(x, i++, v.at<double>(0));
//        gsl_vector_set(x, i++, v.at<double>(1));
//        gsl_vector_set(x, i++, v.at<double>(2));
////    }

//    optimize_with_df(f_frame, df_frame, fdf_frame, nVariables, pMP, s, x, result);

//    pMP->setNormalDirection(result[0], result[1]);
////    LOG(INFO) << "\nn\n" << pMP->getNormal();

//    // assume the depth of the center point of first camera frame is 1;
//    // right pos??
//    cv::Mat pos = (cv::Mat_<double>(3, 1) << 0, 0, 1);
//    pMP->setWorldPos(pos);

//    i = 0;
//    double scale = (*(KFs.rbegin()))->getScale();
//    cv::Mat Twc_last = (*(KFs.rbegin()))->getFirstPose();
//    for (auto KFit = KFs.rbegin(); KFit != KFs.rend(); KFit++) {
//        cv::Mat w = (cv::Mat_<double>(3,1) << result[2 + i * 6],
//                                              result[3 + i * 6],
//                                              result[4 + i * 6]);
//        (*KFit)->setAngularVelocity(w);

//        cv::Mat v = (cv::Mat_<double>(3,1) << result[5 + i * 6],
//                                              result[6 + i * 6],
//                                              result[7 + i * 6]);
//        v = v * scale;
//        (*KFit)->setLinearVelocity(v);
//        double dt = (*KFit)->dt;
//        cv::Mat dw = w * dt;
//        cv::Mat Rc1c2 = axang2rotm(dw);
//        cv::Mat tc1c2 = v * dt; // up to a global scale
//        (*KFit)->setFirstPose(Twc_last);
//        cv::Mat  Twc1 = Twc_last.clone();
//        cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
//        cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
//        cv::Mat Rwc2 = Rwc1 * Rc1c2;
//        cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
//        cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
//        Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
//        twc2.copyTo(Twc2.rowRange(0,3).col(3));
//        (*KFit)->setLastPose(Twc2);
//        (*KFit)->setScale(scale);
//        Twc2.copyTo(Twc_last);
//        scale = scale + (Rwc1 * tc1c2).dot(pMP->getNormal());
//        i++;
//    }
//}

//bool Optimizer::optimize(MapPoint* pMP, shared_ptr<KeyFrame>& pKF) {
//    // for one map point and n keyframes, variable numbers 2 + nKFs * 12 - 6
//    std::set<shared_ptr<KeyFrame>, idxOrder> KFs;
//    auto allKFs = pMP->getObservations();

//    // all the keyframes observing the map point should be added
//    for (auto KFit: allKFs) {
//        KFs.insert(KFit);
//    }
//    KFs.insert(pKF);

//    int nVariables = 2 + KFs.size() * 12 - 6;

//    gsl_multimin_fminimizer *s = NULL;
//    gsl_vector *x;
//    double result[nVariables] = {};

//    /* Starting point */
//    x = gsl_vector_alloc(nVariables);

//    // Normal direction of the plane
//    gsl_vector_set(x, 0, pMP->getNormalDirection().at(0));
//    gsl_vector_set(x, 1, pMP->getNormalDirection().at(1));

//    int i = 0;
//    cv::Mat Rwc, Rwc_w, twc, w, v;
//    for (auto KFit: KFs) {
//        w = KFit->getAngularVelocity();
//        gsl_vector_set(x, 2 + i * 12, w.at<double>(0));
//        gsl_vector_set(x, 3 + i * 12, w.at<double>(1));
//        gsl_vector_set(x, 4 + i * 12, w.at<double>(2));

//        v = KFit->getLinearVelocity() / KFit->mScale;
//        gsl_vector_set(x, 5 + i * 12, v.at<double>(0));
//        gsl_vector_set(x, 6 + i * 12, v.at<double>(1));
//        gsl_vector_set(x, 7 + i * 12, v.at<double>(2));

//        if  (KFit->mnId != 0) {
//            Rwc = KFit->getRotation();
//            Rwc_w = rotm2axang(Rwc);
//            gsl_vector_set(x, 8 + i * 12, Rwc_w.at<double>(0));
//            gsl_vector_set(x, 9 + i * 12, Rwc_w.at<double>(1));
//            gsl_vector_set(x, 10 + i * 12, Rwc_w.at<double>(2));

//            twc = KFit->getTranslation() / Frame::gScale;
//            gsl_vector_set(x, 11 + i * 12, twc.at<double>(0));
//            gsl_vector_set(x, 12 + i * 12, twc.at<double>(1));
//            gsl_vector_set(x, 13 + i * 12, twc.at<double>(2));
//        }

//        i++;
//    }

//    mapPointAndKeyFrames mkf{pMP, &KFs};

//    optimize_gsl(0.1, nVariables, variance_ba, &mkf, s, x, result, 200);

//    // check if the optimization is successful
//    cv::Mat current_frame, current_frame_blurred, other_frames, other_frames_blurred,
//            occupancy_map, occupancy_frame, occupancy_overlap, map;
//    current_frame = cv::Mat::zeros(height, width, CV_64F);
//    other_frames = cv::Mat::zeros(height, width, CV_64F);

//    double res[14] = {};
//    for (int j = 0; j < 14; j++)
//        res[j] = result[j];

//    auto kfit = KFs.begin();
//    intensity(current_frame, res, kfit->get());

//    int f = 1;
//    for (kfit++; f < pMP->observations(); f++, kfit++) {
//        for (int j = 2; j < 14; j++)
//            res[j] = result[f*12+j];
//        intensity(other_frames, res, kfit->get());
//    }
//    for (int j = 2; j < 8; j++)
//        res[j] = result[f*12+j];
//    for (int j = 8; j < 14; j++)
//        res[j] = 0;
//    intensity(other_frames, res, kfit->get());

//    cv::GaussianBlur(current_frame, current_frame_blurred, cv::Size(0, 0), 1, 0);
//    cv::GaussianBlur(other_frames, other_frames_blurred, cv::Size(0, 0), 1, 0);

//    int threshold_value = -1;
//    int const max_BINARY_value = 1;
//    cv::threshold(current_frame, occupancy_frame, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
//    cv::threshold(other_frames, occupancy_map, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
//    cv::bitwise_and(occupancy_frame, occupancy_map, occupancy_overlap);
//    double overlap_rate = (double)cv::countNonZero(occupancy_overlap) / cv::countNonZero(occupancy_frame);
//    LOG(INFO) << "key frame overlap " << overlap_rate;
//    if (overlap_rate < 0.3 || cv::countNonZero(occupancy_frame) < 10) {
//        return false;
//    }

//    // successful, update pose, velocity and map
//    cv::add(current_frame, other_frames, map);
//    map.copyTo(pMP->mBack);

//    pMP->setNormalDirection(result[0], result[1]);

//    i = 0;
//    for (auto KFit :KFs ) {
//        if  (KFit->mnId != 0) {
//            Rwc_w.at<double>(0) = result[8 + i * 12];
//            Rwc_w.at<double>(1) = result[9 + i * 12];
//            Rwc_w.at<double>(2) = result[10 + i * 12];
//            Rwc = axang2rotm(Rwc_w);

//            twc.at<double>(0) = result[11 + i * 12];
//            twc.at<double>(1) = result[12 + i * 12];
//            twc.at<double>(2) = result[13 + i * 12];
//            twc *= Frame::gScale;

//            double scale = Frame::gScale + twc.dot(pMP->getNormal());
//            KFit->setScale(scale);

//            cv::Mat Twc1 = cv::Mat::eye(4,4,CV_64F);
//            Rwc.copyTo(Twc1.rowRange(0,3).colRange(0,3));
//            twc.copyTo(Twc1.rowRange(0,3).col(3));
//            KFit->setFirstPose(Twc1);
//        } else {
//            Rwc = cv::Mat::eye(3, 3, CV_64F);
//            twc = cv::Mat::zeros(3, 1, CV_64F);
//        }
//        w.at<double>(0) = result[2 + i * 12];
//        w.at<double>(1) = result[3 + i * 12];
//        w.at<double>(2) = result[4 + i * 12];
//        KFit->setAngularVelocity(w);

//        v.at<double>(0) = result[5 + i * 12];
//        v.at<double>(1) = result[6 + i * 12];
//        v.at<double>(2) = result[7 + i * 12];
//        v *= KFit->getScale();
//        KFit->setLinearVelocity(v);

//        double dt = KFit->dt;
//        cv::Mat dw = w * dt;
//        cv::Mat Rc1c2 = axang2rotm(dw);
//        cv::Mat tc1c2 = v * dt; // up to a global scale
//        cv::Mat Rwc2 = Rwc * Rc1c2;
//        cv::Mat twc2 = Rwc * tc1c2 + twc;
//        cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
//        Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
//        twc2.copyTo(Twc2.rowRange(0,3).col(3));
//        KFit->setLastPose(Twc2);
//        i++;
//    }
//    return true;
//}

//bool Optimizer::optimize(MapPoint* pMP, Frame* frame, cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v) {

//    double scale = Frame::gScale + twc.dot(pMP->getNormal());
//    if (inFrame)
//    {
//        int nVariables = 6;
//        double result[nVariables] = {};

//        gsl_multimin_fminimizer *s = NULL;
//        gsl_vector *x;

//        /* Starting point */
//        x = gsl_vector_alloc(nVariables);

//        v /= scale;

//        gsl_vector_set(x, 0, w.at<double>(0));
//        gsl_vector_set(x, 1, w.at<double>(1));
//        gsl_vector_set(x, 2, w.at<double>(2));

//        gsl_vector_set(x, 3, v.at<double>(0));
//        gsl_vector_set(x, 4, v.at<double>(1));
//        gsl_vector_set(x, 5, v.at<double>(2));

//        optimize_gsl(1, nVariables, variance_frame, frame, s, x, result, 100);

//        w.at<double>(0) = result[0];
//        w.at<double>(1) = result[1];
//        w.at<double>(2) = result[2];

//        v.at<double>(0) = result[3];
//        v.at<double>(1) = result[4];
//        v.at<double>(2) = result[5];

//        v *= scale;
//    }

//    int nVariables = 12;
//    double result[nVariables] = {};

//    cv::Mat Rwc_w = rotm2axang(Rwc);
//    v /= scale;

//    mapPointAndFrame params{pMP, frame};
//    gsl_multimin_fminimizer *s = NULL;
//    gsl_vector *x;

//    /* Starting point */
//    x = gsl_vector_alloc(nVariables);

//    gsl_vector_set(x, 0, w.at<double>(0));
//    gsl_vector_set(x, 1, w.at<double>(1));
//    gsl_vector_set(x, 2, w.at<double>(2));

//    gsl_vector_set(x, 3, v.at<double>(0));
//    gsl_vector_set(x, 4, v.at<double>(1));
//    gsl_vector_set(x, 5, v.at<double>(2));

//    gsl_vector_set(x, 6, Rwc_w.at<double>(0));
//    gsl_vector_set(x, 7, Rwc_w.at<double>(1));
//    gsl_vector_set(x, 8, Rwc_w.at<double>(2));

//    gsl_vector_set(x, 9, twc.at<double>(0));
//    gsl_vector_set(x, 10, twc.at<double>(1));
//    gsl_vector_set(x, 11, twc.at<double>(2));

//    optimize_gsl(0.5, nVariables, variance_relocalization, &params, s, x, result, 100);

//    w.at<double>(0) = result[0];
//    w.at<double>(1) = result[1];
//    w.at<double>(2) = result[2];

//    v.at<double>(0) = result[3];
//    v.at<double>(1) = result[4];
//    v.at<double>(2) = result[5];

//    Rwc_w.at<double>(0) = result[6];
//    Rwc_w.at<double>(1) = result[7];
//    Rwc_w.at<double>(2) = result[8];

//    twc.at<double>(0) = result[9];
//    twc.at<double>(1) = result[10];
//    twc.at<double>(2) = result[11];

//    // maybe count area rather than pixels is more stable??
//    cv::Mat occupancy_map, occupancy_frame,
//            image_frame, occupancy_overlap;

//    intensity_relocalization(image_frame, result, &params);

//    int threshold_value = -1;
//    int const max_BINARY_value = 1;
//    cv::GaussianBlur(image_frame, image_frame, cv::Size(0, 0), sigma, 0);
//    cv::threshold(image_frame, occupancy_frame, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
//    cv::GaussianBlur(pMP->mFront, image_frame, cv::Size(0, 0), sigma, 0);
//    cv::threshold(image_frame, occupancy_map, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
//    cv::bitwise_and(occupancy_frame, occupancy_map, occupancy_overlap);
//    double overlap_rate = (double)cv::countNonZero(occupancy_overlap) / cv::countNonZero(occupancy_frame);
//    LOG(INFO) << "overlap " << overlap_rate;
//    if (overlap_rate > 0.7) {
//        cv::Mat Twc1 = cv::Mat::eye(4,4,CV_64F);
//        Rwc = axang2rotm(Rwc_w);
//        twc *= Frame::gScale;

//        double scale = Frame::gScale + twc.dot(pMP->getNormal());
//        v *= scale;

//        Rwc.copyTo(Twc1.rowRange(0,3).colRange(0,3));
//        twc.copyTo(Twc1.rowRange(0,3).col(3));

//        double dt = frame->dt;
//        cv::Mat dw = w * dt;
//        cv::Mat Rc1c2 = axang2rotm(dw);
//        cv::Mat tc1c2 = v * dt;

//        cv::Mat Rwc2 = Rwc * Rc1c2;
//        cv::Mat twc2 = Rwc * tc1c2 + twc;
//        cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
//        Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
//        twc2.copyTo(Twc2.rowRange(0,3).col(3));

//        frame->setFirstPose(Twc1);
//        frame->setLastPose(Twc2);
//        frame->setAngularVelocity(w);
//        frame->setLinearVelocity(v);
//        frame->setScale(scale);

//        return true;
//    }
//    return false;
//}

void Optimizer::optimize_gsl(double ss, int nv, double (*f)(const gsl_vector*, void*), void *params,
                             gsl_multimin_fminimizer* s, gsl_vector* x, double* res, size_t iter) {
    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;

    gsl_vector *step_size;
    gsl_multimin_function minex_func;

    size_t it = 0;
    int status;
    double size;

    step_size = gsl_vector_alloc(nv);
    gsl_vector_set_all(step_size, ss);

    minex_func.n = nv;
    minex_func.f = f;
    minex_func.params = params;

    s = gsl_multimin_fminimizer_alloc(T, nv);
    gsl_multimin_fminimizer_set(s, &minex_func, x, step_size);

    do
    {
        it++;
        status = gsl_multimin_fminimizer_iterate(s);

        if (status)
            break;

        size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, 1e-2);

        if (status == GSL_SUCCESS)
        {
            printf ("converged to minimum\n");
        }

    }
    while (status == GSL_CONTINUE && it < iter);

    for (int i = 0; i < nv; i++)
        res[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}

void Optimizer::gsl_fdf(double (*f)(const gsl_vector*, void*), void (*df)(const gsl_vector*, void*, gsl_vector*),
                                 void (*fdf)(const gsl_vector*, void*, double *, gsl_vector *), int nv, void *params,
                                 gsl_multimin_fdfminimizer* s, gsl_vector* x, double* res) {
    const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_conjugate_fr;

    gsl_multimin_function_fdf minex_func;

    size_t it = 0;
    int status;

    minex_func.n = nv;
    minex_func.f = f;
    minex_func.df = df;
    minex_func.fdf = fdf;
    minex_func.params = params;

    s = gsl_multimin_fdfminimizer_alloc(T, nv);

    gsl_multimin_fdfminimizer_set(s, &minex_func, x, 1, 0.1);

    do
    {
        it++;

        status = gsl_multimin_fdfminimizer_iterate(s);

        if (status)
            break;

        status = gsl_multimin_test_gradient (s->gradient, 1e-3);


        if (status == GSL_SUCCESS)
            printf ("Minimum found:\n");

    }
    while (status == GSL_CONTINUE && it < 100);

    for (int i = 0; i < nv; i++)
        res[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(x);
    gsl_multimin_fdfminimizer_free (s);
}

}
