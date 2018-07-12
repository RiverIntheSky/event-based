#include "Optimizer.h"
#include "util/utils.h"


namespace ev {

Eigen::Matrix3d Optimizer::mPatchProjectionMat = Eigen::Matrix3d::Identity();
Eigen::Matrix3d Optimizer::mCameraProjectionMat = Eigen::Matrix3d::Identity();
bool Optimizer::inFrame = true;
bool Optimizer::toMap = true;
int Optimizer::sigma = 1;

double Optimizer::variance_map(const gsl_vector *vec, void *params) {
    MapPoint* pMP = (MapPoint *) params;
    cv::Mat src;

    // this has to be adjusted according to viewing angle!!
    double psi = gsl_vector_get(vec, 1);
    if (std::cos(psi) > 0)
        return 0.;

    // image = pMP->mBack;
    intensity(src, vec, pMP);
    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
    imwriteRescaled(src, "back_buffer.jpg", NULL);
    cv::Scalar mean, stddev;
    cv::meanStdDev(src, mean, stddev);
    double cost = -stddev[0];

    return cost;
}

double Optimizer::variance_frame(const gsl_vector *vec, void *params) {
    Frame* f = (Frame *) params;
    cv::Mat src;

    intensity(src, vec, f);
    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
    imshowRescaled(src, 1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(src, mean, stddev);
    double cost = -stddev[0];

    return cost;
}

double Optimizer::variance_track(const gsl_vector *vec, void *params) {
    mapPointAndFrame* mf = (mapPointAndFrame *) params;
    cv::Mat src;

    intensity(src, vec, mf);
    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
    imshowRescaled(src, 1, "frame_map");

    cv::Scalar mean, stddev;
    cv::meanStdDev(src, mean, stddev);
    double cost = -stddev[0];

    return cost;
}

double Optimizer::variance_ba(const gsl_vector *vec, void *params) {
    mapPointAndKeyFrames* mkf = (mapPointAndKeyFrames *) params;
    cv::Mat src;

    intensity(src, vec, mkf);
    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
    imshowRescaled(src, 1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(src, mean, stddev);
    double cost = -stddev[0];

    return cost;
}

void Optimizer::warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, const Eigen::Vector3d& w, const Eigen::Vector3d& v,const Eigen::Vector3d& n) {

    {
        // plane homography first taylor expansion
        Eigen::Matrix3d H = Eigen::Matrix3d::Identity() + ev::skew(-t * w) + v * t * n.transpose();

        x_w = H.inverse() * x;
        x_w /= x_w(2);
        x_w = mPatchProjectionMat * x_w;
    }
}

void Optimizer::warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                     const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& Rn, const Eigen::Matrix3d& H_) {
    // plane homography
    // v is actually v/dc
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
    Eigen::Matrix3d H = R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose());
    x_w = Rn * H_ * H.inverse() * x;
    x_w /= x_w(2);
    x_w = mPatchProjectionMat * x_w;
}

void Optimizer::warp(Eigen::Vector3d& x_w, const Eigen::Vector3d& x, double t, double theta, const Eigen::Matrix3d& K,
                     const Eigen::Vector3d& v, const Eigen::Vector3d& nc, const Eigen::Matrix3d& H_) {
    // plane homography
    // v is actually v/dc
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
    Eigen::Matrix3d H = R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose());
    x_w = H_ * H.inverse() * x;
    x_w /= x_w(2);
//    x_w = mCameraProjectionMat * x_w;
    x_w = mPatchProjectionMat * x_w;
}

void Optimizer::fuse(cv::Mat& image, Eigen::Vector3d& p, bool polarity) {
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
    if (valid(x1, y1)) {
        double a = (x2 - p(0)) * (y2 - p(1)) * pol;
        image.ptr<double>(y1)[x1] += a;
    }

    if (valid(x1, y2)) {
        double a = -(x2 - p(0)) * (y1 - p(1)) * pol;
        image.ptr<double>(y2)[x1] += a;
    }

    if (valid(x2, y1)) {
        double a = - (x1 - p(0)) * (y2 - p(1)) * pol;
        image.ptr<double>(y1)[x2] += a;
    }

    if (valid(x2, y2)) {
        double a = (x1 - p(0)) * (y1 - p(1)) * pol;
        image.ptr<double>(y2)[x2] += a;
    }
}

void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, MapPoint* pMP) {
    {

        lock_guard<mutex> lock(pMP->mMutexFeatures);

        if (pMP->mFront.empty())
            pMP->mFront = cv::Mat::zeros(500, 500, CV_64F);

        // reset back buffer
        pMP->swap(false);

        // draw to back buffer
        image = pMP->mBack;

        // normal direction of map point
        double phi = gsl_vector_get(vec, 0);
        double psi = gsl_vector_get(vec, 1);
        Eigen::Vector3d nw;
        nw << std::cos(phi) * std::sin(psi),
             std::sin(phi) * std::sin(psi),
             std::cos(psi);

        Eigen::Vector3d z;
        z << 0, 0, 1;
        Eigen::Vector3d v = (-nw).cross(z);
        double c = -z.dot(nw);
        Eigen::Matrix3d Kn = ev::skew(v);
        Eigen::Matrix3d Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);

        // iterate all keyframes
        // only one frame implemented now !!
        auto KFs = pMP->getObservations();

        Eigen::Matrix3d H_ = Eigen::Matrix3d::Identity();
        double t;

        int i = 0;
        for (auto KFit: KFs) {

            okvis::Time t0 = KFit->mTimeStamp;

            // normal direction with respect to first pose
            cv::Mat Rcw = KFit->getRotation().t();

            Eigen::Vector3d nc = Converter::toMatrix3d(Rcw) * nw;

            // velocity
            Eigen::Vector3d w;
            w << gsl_vector_get(vec, 2 + i * 6),
                 gsl_vector_get(vec, 3 + i * 6),
                 gsl_vector_get(vec, 4 + i * 6);

            Eigen::Vector3d v;
            v << gsl_vector_get(vec, 5 + i * 6),
                 gsl_vector_get(vec, 6 + i * 6),
                 gsl_vector_get(vec, 7 + i * 6);

            // getEvents()??
            double theta = -w.norm();

            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
            if (theta != 0)
                K = ev::skew(w.normalized());

            for (const auto EVit: *(KFit->vEvents)) {

                Eigen::Vector3d p, point_warped;
                p << EVit->measurement.x ,EVit->measurement.y, 1;

                // project to first frame
                t = (EVit->timeStamp - t0).toSec();
                warp(point_warped, p, t, theta, K, v, nc, Rn, H_);
                fuse(image, point_warped, EVit->measurement.p);

            }
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
            H_ = (R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose())).inverse() * H_;
            i++;
        }
    }
}

void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, Frame* frame) {
    image = cv::Mat::zeros(500, 500, CV_64F);
    // only works for planar scene!!
    Eigen::Vector3d nw = Converter::toVector3d(frame->mpMap->getAllMapPoints().front()->getNormal());
    Eigen::Matrix3d Rcw = Converter::toMatrix3d(frame->getRotation().t());
    Eigen::Vector3d Twc = Converter::toVector3d(frame->getTranslation());
    Eigen::Vector3d nc = Rcw * nw;
    Eigen::Matrix3d Rn = frame->mpMap->getAllMapPoints().front()->Rn;
    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + Twc * nw.transpose())).inverse();

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

    double theta = -w.norm();

    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    if (theta != 0)
        K = ev::skew(w.normalized());

    for (const auto EVit: frame->vEvents) {

        Eigen::Vector3d p, point_warped;
        p << EVit->measurement.x ,EVit->measurement.y, 1;

        // project to first frame
        warp(point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_);
        fuse(image, point_warped, EVit->measurement.p);
    }
}

void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndFrame* mf) {
    MapPoint* pMP = mf->mP;
    Frame* frame = mf->frame;

    // reset back buffer
    pMP->swap(false);

    // for determining whether new keyframe needed
    cv::Mat occupancy;

    // draw to back buffer
    // test cv::addWeighted!!
    image = pMP->mBack / pMP->observations();

    Eigen::Vector3d nw = Converter::toVector3d(pMP->getNormal());
    Eigen::Matrix3d Rcw = Converter::toMatrix3d(frame->getRotation().t());
    Eigen::Vector3d Twc = Converter::toVector3d(frame->getTranslation());
    Eigen::Vector3d nc = Rcw * nw;
    Eigen::Matrix3d Rn = pMP->Rn;
    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + Twc * nw.transpose())).inverse();

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

    double theta = -w.norm();

    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    if (theta != 0)
        K = ev::skew(w.normalized());

    for (const auto EVit: frame->vEvents) {

        Eigen::Vector3d p, point_warped;
        p << EVit->measurement.x ,EVit->measurement.y, 1;

        // project to first frame
        warp(point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_);
        fuse(image, point_warped, false);
    }
}

void Optimizer::intensity(cv::Mat& image, const double *vec, mapPointAndFrame* mf) {
    MapPoint* pMP = mf->mP;
    Frame* frame = mf->frame;

    image = cv::Mat::zeros(pMP->mBack.size(), pMP->mBack.type());

    Eigen::Vector3d nw = Converter::toVector3d(pMP->getNormal());
    Eigen::Matrix3d Rcw = Converter::toMatrix3d(frame->getRotation().t());
    Eigen::Vector3d Twc = Converter::toVector3d(frame->getTranslation());
    Eigen::Vector3d nc = Rcw * nw;
    Eigen::Matrix3d Rn = pMP->Rn;
    Eigen::Matrix3d H_ = Rn * (Rcw * (Eigen::Matrix3d::Identity() + Twc * nw.transpose())).inverse();

    okvis::Time t0 = frame->mTimeStamp;

    // velocity
    Eigen::Vector3d w, v;
    w << vec[0], vec[1], vec[2];
    v << vec[3], vec[4], vec[5];

    double theta = -w.norm();

    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    if (theta != 0)
        K = ev::skew(w.normalized());

    for (const auto EVit: frame->vEvents) {

        Eigen::Vector3d p, point_warped;
        p << EVit->measurement.x ,EVit->measurement.y, 1;

        // project to first frame
        warp(point_warped, p, (EVit->timeStamp - t0).toSec(), theta, K, v, nc, H_);
        fuse(image, point_warped, false);
    }
}

void Optimizer::intensity(cv::Mat& image, const gsl_vector *vec, mapPointAndKeyFrames* mkf) {
    MapPoint* pMP = mkf->mP;
    std::set<shared_ptr<KeyFrame>, idxOrder> kfs = *(mkf->kfs);

    cv::Mat zero = cv::Mat::zeros(pMP->mBack.size(), pMP->mBack.type());
    zero.copyTo(pMP->mBack);
    image = pMP->mBack;

    // normal direction of map point
    double phi = gsl_vector_get(vec, 0);
    double psi = gsl_vector_get(vec, 1);
    Eigen::Vector3d nw;
    nw << std::cos(phi) * std::sin(psi),
         std::sin(phi) * std::sin(psi),
         std::cos(psi);

    Eigen::Vector3d z;
    z << 0, 0, 1;
    Eigen::Vector3d v = (-nw).cross(z);
    double c = -z.dot(nw);
    Eigen::Matrix3d Kn = ev::skew(v);
    Eigen::Matrix3d Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);

    double t;

    int i = 0;
    for (auto KFit: kfs) {
        okvis::Time t0 = KFit->mTimeStamp;

        Eigen::Vector3d Rwc_w, twc, w, v;

        Rwc_w << gsl_vector_get(vec, 2 + i * 12),
                 gsl_vector_get(vec, 3 + i * 12),
                 gsl_vector_get(vec, 4 + i * 12);

        Eigen::Matrix3d Rwc = axang2rotm(Rwc_w);
        Eigen::Vector3d nc = Rwc.transpose() * nw;

        twc << gsl_vector_get(vec, 5 + i * 12),
               gsl_vector_get(vec, 6 + i * 12),
               gsl_vector_get(vec, 7 + i * 12);

        w << gsl_vector_get(vec, 8 + i * 12),
             gsl_vector_get(vec, 9 + i * 12),
             gsl_vector_get(vec, 10 + i * 12);

        v << gsl_vector_get(vec, 11 + i * 12),
             gsl_vector_get(vec, 12 + i * 12),
             gsl_vector_get(vec, 13 + i * 12);

        Eigen::Matrix3d H_ = (Rwc.transpose() * (Eigen::Matrix3d::Identity() + twc * nw.transpose())).inverse();

        double theta = -w.norm();

        Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
        if (theta != 0)
            K = ev::skew(w.normalized());

        for (const auto EVit: *(KFit->vEvents)) {

            Eigen::Vector3d p, point_warped;
            p << EVit->measurement.x ,EVit->measurement.y, 1;

            // project to first frame
            t = (EVit->timeStamp - t0).toSec();
            warp(point_warped, p, t, theta, K, v, nc, Rn, H_);
            fuse(image, point_warped, false);
        }
        i++;
    }
}

void Optimizer::optimize(MapPoint* pMP) {
    // for one map point and n keyframes, variable numbers 2 + nKFs * 6

    mPatchProjectionMat(0, 0) = 200;
    mPatchProjectionMat(1, 1) = 200;
    mPatchProjectionMat(0, 2) = 250;
    mPatchProjectionMat(1, 2) = 250;

    int nKFs = pMP->observations();
    auto KFs = pMP->getObservations();
    int nVariables = 2 + nKFs * 6;

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;
    double result[nVariables] = {};

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    // Normal direction of the plane
    gsl_vector_set(x, 0, pMP->getNormalDirection().at(0));
    gsl_vector_set(x, 1, pMP->getNormalDirection().at(1));

    int i = 0;
    for (auto KFit: KFs) {
        cv::Mat w = KFit->getAngularVelocity();
        gsl_vector_set(x, 2 + i * 6, w.at<double>(0));
        gsl_vector_set(x, 3 + i * 6, w.at<double>(1));
        gsl_vector_set(x, 4 + i * 6, w.at<double>(2));

        cv::Mat v = KFit->getLinearVelocity() / KFit->mScale;
        gsl_vector_set(x, 5 + i * 6, v.at<double>(0));
        gsl_vector_set(x, 6 + i * 6, v.at<double>(1));
        gsl_vector_set(x, 7 + i * 6, v.at<double>(2));
        i++;
    }

    optimize_gsl(0.05, nVariables, variance_map, pMP, s, x, result, 100);

    pMP->setNormalDirection(result[0], result[1]);
//    LOG(INFO) << "\nn\n" << pMP->getNormal();

    // assume the depth of the center point of first camera frame is 1;
    // right pos??
    cv::Mat pos = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    pMP->setWorldPos(pos);

    i = 0;
    double scale = (*(KFs.begin()))->getScale();
    for (auto KFit :KFs ) {
        cv::Mat w = (cv::Mat_<double>(3,1) << result[2 + i * 6],
                                              result[3 + i * 6],
                                              result[4 + i * 6]);
        KFit->setAngularVelocity(w);

        cv::Mat v = (cv::Mat_<double>(3,1) << result[5 + i * 6],
                                              result[6 + i * 6],
                                              result[7 + i * 6]);
        v = v * scale;
        KFit->setLinearVelocity(v);
        double dt = KFit->dt;
        cv::Mat dw = w * dt;
        cv::Mat Rc1c2 = axang2rotm(dw);
        cv::Mat tc1c2 = v * dt; // up to a global scale
        cv::Mat Twc1 = KFit->getFirstPose();
        cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
        cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
        cv::Mat Rwc2 = Rwc1 * Rc1c2;
        cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
        cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
        Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
        twc2.copyTo(Twc2.rowRange(0,3).col(3));
        KFit->setLastPose(Twc2);
        KFit->setScale(scale);
        scale = scale + (Rwc1 * tc1c2).dot(pMP->getNormal());
        i++;
    }
}

void Optimizer::optimize(MapPoint* pMP, Frame* frame) {
    int nVariables = 6;
    double result[nVariables] = {};

    cv::Mat w = frame->getAngularVelocity();
    cv::Mat v = frame->getLinearVelocity() / frame->mScale;

    if (inFrame)
    {
        gsl_multimin_fminimizer *s = NULL;
        gsl_vector *x;

        /* Starting point */
        x = gsl_vector_alloc(nVariables);

        gsl_vector_set(x, 0, w.at<double>(0));
        gsl_vector_set(x, 1, w.at<double>(1));
        gsl_vector_set(x, 2, w.at<double>(2));

        gsl_vector_set(x, 3, v.at<double>(0));
        gsl_vector_set(x, 4, v.at<double>(1));
        gsl_vector_set(x, 5, v.at<double>(2));

        optimize_gsl(0.1, nVariables, variance_frame, frame, s, x, result, 100);

        w.at<double>(0) = result[0];
        w.at<double>(1) = result[1];
        w.at<double>(2) = result[2];

        v.at<double>(0) = result[3];
        v.at<double>(1) = result[4];
        v.at<double>(2) = result[5];
    }

    if (toMap) {
        mapPointAndFrame params{pMP, frame};

        gsl_multimin_fminimizer *s = NULL;
        gsl_vector *x;

        /* Starting point */
        x = gsl_vector_alloc(nVariables);

        gsl_vector_set(x, 0, w.at<double>(0));
        gsl_vector_set(x, 1, w.at<double>(1));
        gsl_vector_set(x, 2, w.at<double>(2));

        gsl_vector_set(x, 3, v.at<double>(0));
        gsl_vector_set(x, 4, v.at<double>(1));
        gsl_vector_set(x, 5, v.at<double>(2));

        optimize_gsl(0.05, nVariables, variance_track, &params, s, x, result, 100);

        w.at<double>(0) = result[0];
        w.at<double>(1) = result[1];
        w.at<double>(2) = result[2];

        v.at<double>(0) = result[3];
        v.at<double>(1) = result[4];
        v.at<double>(2) = result[5];

        // maybe count area rather than pixels is more stable??
        cv::Mat occupancy_map, occupancy_frame,
                image_frame, occupancy_overlap;

        intensity(image_frame, result, &params);

        int threshold_value = -1;
        int const max_BINARY_value = 1;
        cv::threshold(image_frame, occupancy_frame, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
        cv::threshold(pMP->mFront, occupancy_map, threshold_value, max_BINARY_value, cv::THRESH_BINARY_INV);
        cv::bitwise_and(occupancy_frame, occupancy_map, occupancy_overlap);
        double overlap_rate = (double)cv::countNonZero(occupancy_overlap) / cv::countNonZero(occupancy_frame);
        LOG(INFO) << "overlap " << overlap_rate;
        if (overlap_rate < 0.8)
            frame->shouldBeKeyFrame = true;
    }

    frame->setAngularVelocity(w);
    v *= frame->mScale;
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
    frame->setLastPose(Twc2);
}

void Optimizer::optimize(MapPoint* pMP, shared_ptr<KeyFrame>& pKF) {
    // for one map point and n keyframes, variable numbers 2 + nKFs * 12
    std::set<shared_ptr<KeyFrame>, idxOrder> KFs;
    auto allKFs = pMP->getObservations();

    // all the keyframes observing the map point should be added
    for (auto KFit: allKFs) {
        KFs.insert(KFit);
    }
    KFs.insert(pKF);

    int nVariables = 2 + KFs.size() * 12;

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;
    double result[nVariables] = {};

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    // Normal direction of the plane
    gsl_vector_set(x, 0, pMP->getNormalDirection().at(0));
    gsl_vector_set(x, 1, pMP->getNormalDirection().at(1));

    int i = 0;
    cv::Mat Rwc, Rwc_w, twc, w, v;
    for (auto KFit: KFs) {
        Rwc = KFit->getRotation();
        Rwc_w = rotm2axang(Rwc);
        gsl_vector_set(x, 2 + i * 12, Rwc_w.at<double>(0));
        gsl_vector_set(x, 3 + i * 12, Rwc_w.at<double>(1));
        gsl_vector_set(x, 4 + i * 12, Rwc_w.at<double>(2));

        twc = KFit->getTranslation();
        gsl_vector_set(x, 5 + i * 12, twc.at<double>(0));
        gsl_vector_set(x, 6 + i * 12, twc.at<double>(1));
        gsl_vector_set(x, 7 + i * 12, twc.at<double>(2));

        w = KFit->getAngularVelocity();
        gsl_vector_set(x, 8 + i * 12, w.at<double>(0));
        gsl_vector_set(x, 9 + i * 12, w.at<double>(1));
        gsl_vector_set(x, 10 + i * 12, w.at<double>(2));

        v = KFit->getLinearVelocity() / KFit->mScale;
        gsl_vector_set(x, 11 + i * 12, v.at<double>(0));
        gsl_vector_set(x, 12 + i * 12, v.at<double>(1));
        gsl_vector_set(x, 13 + i * 12, v.at<double>(2));
        i++;
    }

    mapPointAndKeyFrames mkf{pMP, &KFs};

    optimize_gsl(0.05, nVariables, variance_ba, &mkf, s, x, result, 100);

    pMP->setNormalDirection(result[0], result[1]);

    i = 0;
    for (auto KFit :KFs ) {
        Rwc_w.at<double>(0) = result[2 + i * 12];
        Rwc_w.at<double>(1) = result[3 + i * 12];
        Rwc_w.at<double>(2) = result[4 + i * 12];

        Rwc = axang2rotm(Rwc_w);

        twc.at<double>(0) = result[5 + i * 12];
        twc.at<double>(1) = result[6 + i * 12];
        twc.at<double>(2) = result[7 + i * 12];
        // can not resolve scale drift!!
        twc *= KFit->mScale;

        cv::Mat Twc1 = cv::Mat::eye(4,4,CV_64F);
        Rwc.copyTo(Twc1.rowRange(0,3).colRange(0,3));
        twc.copyTo(Twc1.rowRange(0,3).col(3));
        KFit->setFirstPose(Twc1);

        w.at<double>(0) = result[8 + i * 12];
        w.at<double>(1) = result[9 + i * 12];
        w.at<double>(2) = result[10 + i * 12];
        KFit->setAngularVelocity(w);

        v.at<double>(0) = result[11 + i * 12];
        v.at<double>(1) = result[12 + i * 12];
        v.at<double>(2) = result[13 + i * 12];
        v *= KFit->mScale;
        KFit->setLinearVelocity(v);

        double dt = KFit->dt;
        cv::Mat dw = w * dt;
        cv::Mat Rc1c2 = axang2rotm(dw);
        cv::Mat tc1c2 = v * dt; // up to a global scale
        cv::Mat Rwc2 = Rwc * Rc1c2;
        cv::Mat twc2 = Rwc * tc1c2 + twc;
        cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
        Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
        twc2.copyTo(Twc2.rowRange(0,3).col(3));
        KFit->setLastPose(Twc2);
        i++;
    }
}

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

}
