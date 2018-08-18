#include "Optimizer.h"
#include "util/utils.h"


namespace ev {

double Optimizer::f_frame(const gsl_vector *vec, void *params) {
    Frame* frame = (Frame *) params;
    cv::Mat src;
LOG(INFO) << "---------";
    intensity(src, vec, NULL, frame);
    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
//    imwriteRescaled(src, "/home/weizhen/Documents/dataset/shapes_translation/map_reloc/4000/frame_" + std::to_string(count_frame) + ".jpg", NULL);
    imshowRescaled(src, 1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(src, mean, stddev);
    double cost = -std::pow(stddev[0], 2);
//    LOG(INFO) << cost;

    return cost;
}

void Optimizer::df_frame(const gsl_vector *vec, void *params, gsl_vector* df) {
    Frame* frame = (Frame *) params;
    cv::Mat image, dimage;
    Eigen::MatrixXd dIdw = Eigen::MatrixXd::Zero(height * width, vec->size);

    intensity(image, vec, &dIdw, frame);
    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma, 0);
    double mean = cv::mean(image)[0];
    double area = height * width;
    for (size_t i = 0; i != vec->size; i++) {
        double di = 0;
        Eigen::MatrixXd d = dIdw.col(i);
        d.resize(height, width);
        cv::eigen2cv(d, dimage);
        cv::GaussianBlur(dimage, dimage, cv::Size(0, 0), sigma, 0);
//        LOG(INFO) << dimage;
//        imshowRescaled(dimage, 0);
        double dmean = cv::mean(dimage)[0];
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                di -= (image.at<double>(r, c) - mean) * (dimage.at<double>(r, c) - dmean);
            }
        }
//        LOG(INFO) << "dw"<<i<<" mean = " << mean << " " << dmean;
//        LOG(INFO) << "dw"<<i<<" = " << di * 2 / area;
        gsl_vector_set(df, i, di * 2 / area);
    }
}

void Optimizer::fdf_frame(const gsl_vector *vec, void *params, double *f, gsl_vector* df) {
    Frame* frame = (Frame *) params;
    cv::Mat image, dimage;
    Eigen::MatrixXd dIdw = Eigen::MatrixXd::Zero(height * width, vec->size);

    intensity(image, vec, &dIdw, frame);
    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma, 0);
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    *f = -std::pow(stddev[0], 2);
// LOG(INFO) << *f;
    double area = height * width;
    for (size_t i = 0; i != vec->size; i++) {
        double di = 0;
        Eigen::MatrixXd d = dIdw.col(i);
        d.resize(height, width);
        cv::eigen2cv(d, dimage);
        cv::GaussianBlur(dimage, dimage, cv::Size(0, 0), sigma, 0);
        double dmean = cv::mean(dimage)[0];
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                di -= (image.at<double>(r, c) - mean[0]) * (dimage.at<double>(r, c) - dmean);
            }
        }
//        LOG(INFO) << "dw"<<i<<" = " << di * 2 / area;
        gsl_vector_set(df, i, di * 2 / area);
    }
}

//double Optimizer::variance_track(const gsl_vector *vec, void *params) {
//    mapPointAndFrame* mf = (mapPointAndFrame *) params;
//    cv::Mat src;

//    intensity(src, vec, mf);
//    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
//     imshowRescaled(src, 1);

//    cv::Scalar mean, stddev;
//    cv::meanStdDev(src, mean, stddev);
//    double cost = -std::pow(stddev[0], 2);

//    return cost;
//}

void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, Eigen::MatrixXd* dIdW, Frame* frame) {
    image = cv::Mat::zeros(height, width, CV_64F);
    // only works for planar scene!!
    Eigen::Vector3d nw = Converter::toVector3d(frame->mpMap->getAllMapPoints().front()->getNormal());
    Eigen::Matrix3d Rcw = Converter::toMatrix3d(frame->getRotation().t());
    Eigen::Vector3d twc = Converter::toVector3d(frame->getTranslation());
    //    Eigen::Vector3d nc = Rcw * nw;
    //    Eigen::Matrix3d Rn = frame->mpMap->getAllMapPoints().front()->Rn;
    //    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

    //    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d H_ = Eigen::Matrix3d::Identity();;

    okvis::Time t0 = frame->mTimeStamp;

    // velocity
    Eigen::Vector3d w;
    w << gsl_vector_get(vec, 0),
            gsl_vector_get(vec, 1),
            gsl_vector_get(vec, 2);

    Eigen::Vector3d v;
    v << gsl_vector_get(vec, 3),
            gsl_vector_get(vec, 4),
            gsl_vector_get(vec, 5);

    Eigen::Vector2d n;
    n << gsl_vector_get(vec, 6),
            gsl_vector_get(vec, 7);

    Eigen::Vector3d nc;
    nc << std::cos(n(0)) * std::sin(n(1)),
            std::sin(n(0)) * std::sin(n(1)),
            std::cos(n(1));

    double theta = -w.norm();

    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    if (theta != 0)
        K = skew(w.normalized());
    Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(3, 8);
    LOG(INFO) << "---------";
    for (const auto EVit: frame->vEvents) {

        Eigen::Vector3d p, point_warped;
        p << EVit->measurement.x ,EVit->measurement.y, 1;

        // project to first frame
        if (dIdW) {
            warp(&dW, point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_, n);
            fuse(dIdW, &dW, image, point_warped, EVit->measurement.p);
        } else {
            warp(NULL, point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_);
            fuse(NULL, NULL, image, point_warped, EVit->measurement.p);
        }
    }
}

//void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndFrame* mf) {
//    MapPoint* pMP = mf->mP;
//    Frame* frame = mf->frame;

//    // reset back buffer
//    pMP->swap(false);

//    // draw to back buffer
//    // test cv::addWeighted!!
//    image = pMP->mBack;

//    Eigen::Vector3d nw = Converter::toVector3d(pMP->getNormal());
//    Eigen::Matrix3d Rcw = Converter::toMatrix3d(frame->getRotation().t());
//    Eigen::Vector3d twc = Converter::toVector3d(frame->getTranslation()) / Frame::gScale;
//    Eigen::Vector3d nc = Rcw * nw;
//    Eigen::Matrix3d Rn = pMP->Rn;
//    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

//    okvis::Time t0 = frame->mTimeStamp;

//    // velocity
//    Eigen::Vector3d w;
//    w << gsl_vector_get(vec, 0),
//         gsl_vector_get(vec, 1),
//         gsl_vector_get(vec, 2);

//    Eigen::Vector3d v;
//    v << gsl_vector_get(vec, 3),
//         gsl_vector_get(vec, 4),
//         gsl_vector_get(vec, 5);

//    double theta = -w.norm();

//    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
//    if (theta != 0)
//        K = ev::skew(w.normalized());

////    LOG(INFO)<<"H_ " << H_;
////    LOG(INFO)<<"K_ " << K;
////    LOG(INFO)<<"v " << v;
////    LOG(INFO)<<"nc " << nc;
////    LOG(INFO)<<"theta " << theta;

//    for (const auto EVit: frame->vEvents) {

//        Eigen::Vector3d p, point_warped;
//        p << EVit->measurement.x ,EVit->measurement.y, 1;

//        // project to first frame
//        warp(point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_);
//        fuse(image, point_warped, false);
//    }
////         imshowRescaled(image, 0);

//}

//void Optimizer::intensity(cv::Mat& image, const double *vec, mapPointAndFrame* mf) {
//    MapPoint* pMP = mf->mP;
//    Frame* frame = mf->frame;

//    image = cv::Mat::zeros(pMP->mBack.size(), pMP->mBack.type());

//    Eigen::Vector3d nw = Converter::toVector3d(pMP->getNormal());
//    Eigen::Matrix3d Rcw = Converter::toMatrix3d(frame->getRotation().t());
//    Eigen::Vector3d twc = Converter::toVector3d(frame->getTranslation()) / Frame::gScale;
//    Eigen::Vector3d nc = Rcw * nw;
//    Eigen::Matrix3d Rn = pMP->Rn;
//    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

//    okvis::Time t0 = frame->mTimeStamp;

//    // velocity
//    Eigen::Vector3d w, v;
//    w << vec[0], vec[1], vec[2];
//    v << vec[3], vec[4], vec[5];

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

//void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndKeyFrames* mkf) {
//    MapPoint* pMP = mkf->mP;
//    std::set<shared_ptr<KeyFrame>, idxOrder> kfs = *(mkf->kfs);

//    cv::Mat zero = cv::Mat::zeros(pMP->mBack.size(), pMP->mBack.type());
//    zero.copyTo(pMP->mBack);
//    image = pMP->mBack;

//    // normal direction of map point
//    double phi = gsl_vector_get(vec, 0);
//    double psi = gsl_vector_get(vec, 1);
//    Eigen::Vector3d nw;
//    nw << std::cos(phi) * std::sin(psi),
//         std::sin(phi) * std::sin(psi),
//         std::cos(psi);

//    Eigen::Vector3d z;
//    z << 0, 0, 1;
//    Eigen::Vector3d v = (-nw).cross(z);
//    double c = -z.dot(nw);
//    Eigen::Matrix3d Kn = ev::skew(v);
//    Eigen::Matrix3d Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);

//    double t;

//    int i = 0;
//    for (auto KFit: kfs) {

//        okvis::Time t0 = KFit->mTimeStamp;

//        Eigen::Vector3d Rwc_w, twc, w, v, nc;
//        Eigen::Matrix3d Rcw;

//        w << gsl_vector_get(vec, 2 + i * 12),
//             gsl_vector_get(vec, 3 + i * 12),
//             gsl_vector_get(vec, 4 + i * 12);

//        v << gsl_vector_get(vec, 5 + i * 12),
//             gsl_vector_get(vec, 6 + i * 12),
//             gsl_vector_get(vec, 7 + i * 12);

//        if (KFit->mnId != 0) {
//            Rwc_w << gsl_vector_get(vec, 8 + i * 12),
//                     gsl_vector_get(vec, 9 + i * 12),
//                     gsl_vector_get(vec, 10 + i * 12);

//            Rcw = axang2rotm(Rwc_w).transpose();
//            nc = Rcw * nw;

//            twc << gsl_vector_get(vec, 11 + i * 12),
//                   gsl_vector_get(vec, 12 + i * 12),
//                   gsl_vector_get(vec, 13 + i * 12);
//        } else {
//            Rcw = Eigen::Matrix3d::Identity();
//            twc = Eigen::Vector3d::Zero();
//            nc = nw;
//        }

//        Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

//        double theta = -w.norm();

//        Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
//        if (theta != 0)
//            K = ev::skew(w.normalized());

//        for (const auto EVit: *(KFit->vEvents)) {

//            Eigen::Vector3d p, point_warped;
//            p << EVit->measurement.x ,EVit->measurement.y, 1;

//            // project to first frame
//            t = (EVit->timeStamp - t0).toSec();
//            warp(point_warped, p, t, theta, K, v, nc, H_);
//            fuse(image, point_warped, false);
//        }
//        i++;
//    }

//}

//void Optimizer::intensity(cv::Mat& image, const double *vec, KeyFrame* kF) {

//    double phi = vec[0];
//    double psi = vec[1];
//    Eigen::Vector3d nw;
//    nw << std::cos(phi) * std::sin(psi),
//          std::sin(phi) * std::sin(psi),
//          std::cos(psi);

//    Eigen::Vector3d z;
//    z << 0, 0, 1;
//    Eigen::Vector3d nv = (-nw).cross(z);
//    double c = -z.dot(nw);
//    Eigen::Matrix3d Kn = ev::skew(nv);
//    Eigen::Matrix3d Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);

//    okvis::Time t0 = kF->mTimeStamp;

//    // velocity
//    Eigen::Vector3d w, v;
//    w << vec[2], vec[3], vec[4];
//    v << vec[5], vec[6], vec[7];

//    Eigen::Vector3d Rwc_w;
//    Rwc_w << vec[8], vec[9], vec[10];
//    Eigen::Matrix3d Rcw = axang2rotm(Rwc_w).transpose();
//    Eigen::Vector3d nc = Rcw * nw;

//    Eigen::Vector3d twc;
//    twc << vec[11], vec[12], vec[13];

//    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

//    double theta = -w.norm();

//    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
//    if (theta != 0)
//        K = ev::skew(w.normalized());

//    for (const auto EVit: *(kF->vEvents)) {

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

void Optimizer::optimize(MapPoint* pMP, Frame* frame) {
    LOG(INFO) << "---------";
    int nVariables = 8;
    double result[nVariables] = {};

    auto n = pMP->getNormalDirection();
    cv::Mat w = frame->getAngularVelocity();
    cv::Mat v = frame->getLinearVelocity() /*/ frame->mScale*/;

    if (inFrame)
    {
        gsl_multimin_fdfminimizer *s = NULL;
        gsl_vector *x;

        /* Starting point */
        x = gsl_vector_alloc(nVariables);

        gsl_vector_set(x, 0, w.at<double>(0));
        gsl_vector_set(x, 1, w.at<double>(1));
        gsl_vector_set(x, 2, w.at<double>(2));

        gsl_vector_set(x, 3, v.at<double>(0));
        gsl_vector_set(x, 4, v.at<double>(1));
        gsl_vector_set(x, 5, v.at<double>(2));

        gsl_vector_set(x, 6, n[0]);
        gsl_vector_set(x, 7, n[1]);
LOG(INFO) << "---------";
        gsl_fdf(f_frame, df_frame, fdf_frame, nVariables, frame, s, x, result);
LOG(INFO) << "---------";
        w.at<double>(0) = result[0];
        w.at<double>(1) = result[1];
        w.at<double>(2) = result[2];

        v.at<double>(0) = result[3];
        v.at<double>(1) = result[4];
        v.at<double>(2) = result[5];

        pMP->setNormalDirection(result[6], result[7]);
    }

//    if (toMap) {
//        mapPointAndFrame params{pMP, frame};

//        gsl_multimin_fminimizer *s = NULL;
//        gsl_vector *x;

//        /* Starting point */
//        x = gsl_vector_alloc(nVariables);

//        gsl_vector_set(x, 0, w.at<double>(0));
//        gsl_vector_set(x, 1, w.at<double>(1));
//        gsl_vector_set(x, 2, w.at<double>(2));

//        gsl_vector_set(x, 3, v.at<double>(0));
//        gsl_vector_set(x, 4, v.at<double>(1));
//        gsl_vector_set(x, 5, v.at<double>(2));

//        optimize_gsl(0.5, nVariables, variance_track, &params, s, x, result, 100);

//        w.at<double>(0) = result[0];
//        w.at<double>(1) = result[1];
//        w.at<double>(2) = result[2];

//        v.at<double>(0) = result[3];
//        v.at<double>(1) = result[4];
//        v.at<double>(2) = result[5];

//        if ((*(pMP->getObservations().cbegin()))->mnFrameId != frame->mnId - 1) {

//            // maybe count area rather than pixels is more stable??
//            cv::Mat occupancy_map, occupancy_frame,
//                    image_frame, occupancy_overlap;

//            intensity(image_frame, result, &params);

//            int threshold_value = -1;
//            int const max_BINARY_value = 1;
//            cv::GaussianBlur(image_frame, image_frame, cv::Size(0, 0), sigma, 0);
//            cv::threshold(image_frame, occupancy_frame, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
//            cv::GaussianBlur(pMP->mFront, image_frame, cv::Size(0, 0), sigma, 0);
//            cv::threshold(image_frame, occupancy_map, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
//            cv::bitwise_and(occupancy_frame, occupancy_map, occupancy_overlap);
//            double overlap_rate = (double)cv::countNonZero(occupancy_overlap) / cv::countNonZero(occupancy_frame);
//            LOG(INFO) << "overlap " << overlap_rate;
//            if (overlap_rate < 0.8)
//                frame->shouldBeKeyFrame = true;
//        }
//    }
LOG(INFO) << "---------";
    frame->setAngularVelocity(w);
//    v *= frame->mScale;
    frame->setLinearVelocity(v);

    double dt = frame->dt;
    cv::Mat dw = w * dt;
    cv::Mat Rc1c2 = axang2rotm(dw);
    cv::Mat tc1c2 = v * dt; // up to a global scale
    cv::Mat Twc1 = frame->getFirstPose();
    cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
    cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
    cv::Mat Rwc2 = Rwc1 * Rc1c2;
    cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
    twc2.copyTo(Twc2.rowRange(0,3).col(3));
//    frame->setLastPose(Twc2);
    frame->setLastPose(Twc1);
}

}
