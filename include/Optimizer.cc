#include "Optimizer.h"
#include "util/utils.h"


namespace ev {

Eigen::Matrix3d Optimizer::mPatchProjectionMat = Eigen::Matrix3d::Identity();

double Optimizer::variance(const gsl_vector *vec, void *params) {
    MapPoint* pMP = (MapPoint *) params;
    cv::Mat src;
    int sigma = 1;

    double psi = gsl_vector_get(vec, 1);
    if (std::cos(psi) > 0)
        return 0.;

    // image = pMP->mBack;
    intensity(src, vec, pMP);
    cv::GaussianBlur(src, src, cv::Size(0, 0), sigma, 0);
    imshowRescaled(src, 1, "back_buffer.jpg", NULL);
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
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(t*theta) * K + (1 - std::cos(t*theta)) * K * K;
    Eigen::Matrix3d H = R * (Eigen::Matrix3d::Identity() + v * t * nc.transpose());
    x_w = Rn * H_ * H.inverse() * x;
    x_w /= x_w(2);
    x_w = mPatchProjectionMat * x_w;

}

void Optimizer::fuse(cv::Mat& image, Eigen::Vector3d& p, bool& polarity) {
    // range check
    auto valid = [image](int x, int y)  -> bool {
        return (x >= 0 && x < image.cols && y >= 0 && y < image.rows);
    };

    // change to predefined parameter or drop completely!! (#if)
//    int pol = 1;
//    if (param->use_polarity) {
    int pol = int(polarity) * 2 - 1;
//    }

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

        // projection matrix of the map point
        Eigen::Vector3d z;
        z << 0, 0, 1;
        Eigen::Vector3d v = (-nw).cross(z);
        double c = -z.dot(nw);
        Eigen::Matrix3d Kn = ev::skew(v);
        Eigen::Matrix3d Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);

        // iterate all keyframes
        // only one frame implemented now !!
        auto KFs = pMP->getObservations();
        int i = 0;
        Eigen::Matrix3d H_ = Eigen::Matrix3d::Identity();
        double t;

        for (auto KFit: KFs) {

            auto& e0 = *(KFit->vEvents)->begin();
            okvis::Time t0 = e0->timeStamp;

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
//            H_ = (Eigen::Matrix3d::Identity() + 200 * v * nc.transpose()).inverse() * H_;
            i++;
        }
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

    LOG(INFO) << nVariables;

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *step_size, *x;
    gsl_multimin_function minex_func;

    size_t iter = 0;
    int status;
    double size;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    // Normal direction of the plane
    gsl_vector_set(x, 0, pMP->getNormalDirection().at(0));
    gsl_vector_set(x, 1, pMP->getNormalDirection().at(1));

    int i = 0;
    for (auto KFit :KFs ) {

        cv::Vec3d w = KFit->getAngularVelocity();
        gsl_vector_set(x, 2 + i * 6, w(0));
        gsl_vector_set(x, 3 + i * 6, w(1));
        gsl_vector_set(x, 4 + i * 6, w(2));

        cv::Vec3d v = KFit->getLinearVelocity();
        gsl_vector_set(x, 5 + i * 6, v(0));
        gsl_vector_set(x, 6 + i * 6, v(1));
        gsl_vector_set(x, 7 + i * 6, v(2));
        i++;
    }

    /* Set initial step sizes to 1 */
    step_size = gsl_vector_alloc(nVariables);
    // should the step_size be adjusted??
    gsl_vector_set_all(step_size, 0.05);

    /* Initialize method and iterate */
    minex_func.n = nVariables;
    minex_func.f = variance;
    minex_func.params = pMP;

    s = gsl_multimin_fminimizer_alloc(T, nVariables);
    gsl_multimin_fminimizer_set(s, &minex_func, x, step_size);

    do
    {
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);

        if (status)
            break;

        size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, 1e-2);

        if (status == GSL_SUCCESS)
        {
            printf ("converged to minimum at\n");
        }

//        LOG(INFO) << size;

    }
    while (status == GSL_CONTINUE && iter < 100);

    pMP->setNormalDirection(gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));
    LOG(INFO) << "\nn\n" << pMP->getNormal();

    // assume the depth of the center point of first camera frame is 1;
    cv::Mat pos = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    pMP->setWorldPos(pos);

    i = 0;

    for (auto KFit :KFs ) {
        cv::Mat w = (cv::Mat_<double>(3,1) << gsl_vector_get(s->x, 2 + i * 6),
                                              gsl_vector_get(s->x, 3 + i * 6),
                                              gsl_vector_get(s->x, 4 + i * 6));
        KFit->setAngularVelocity(w);

        cv::Mat v = (cv::Mat_<double>(3,1) << gsl_vector_get(s->x, 5 + i * 6),
                                              gsl_vector_get(s->x, 6 + i * 6),
                                              gsl_vector_get(s->x, 7 + i * 6));
        KFit->setLinearVelocity(v);

        double dt = ((*(KFit->vEvents->crbegin()))->timeStamp - (*(KFit->vEvents->cbegin()))->timeStamp).toSec();
        cv::Mat dw = -w * dt;
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
        i++;
    }

     // remeber to set the kf and mp pos!!

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}

}
