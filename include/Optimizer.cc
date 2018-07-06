#include "Optimizer.h"
#include "util/utils.h"


namespace ev {

double static variance(const gsl_vector *vec, void *params);

// vector Mat_<double>(3,1)
void Optimizer::warp(cv::Mat& x_w, cv::Mat& x, okvis::Duration& t, const cv::Mat& w, const cv::Mat& v, const cv::Mat& n) const {
    double t_ = t.toSec();

    // plane homography
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F) + ev::skew(-t_ * w) + v * t_ * n.t();
    x_w = H.inv() * x;
    x_w /= x_w.at<double>(2);
    // should be changed for general patches !!
    x_w = param.cameraMatrix_ * x_w;
}

void Optimizer::fuse(cv::Mat& image, cv::Point2d& p, bool& polarity) const {
    // range check
    auto valid = [](int x, int y)  -> bool {
        return (x >= 0 && x < image.cols && y >= 0 && y < image.rows);
    };

    // change to predefined parameter or drop completely!! (#if)
    int pol = 1;
    if (param.use_polarity) {
        pol = int(polarity) * 2 - 1;
    }

    int x1 = std::floor(p.x);
    int x2 = x1 + 1;
    int y1 = std::floor(p.y);
    int y2 = y1 + 1;
    if (valid(x1, y1)) {
        double a = (x2 - p.x) * (y2 - p.y) * pol;
        image.ptr<double>(y1)[x1] += a;
    }

    if (valid(x1, y2)) {
        double a = -(x2 - p.x) * (y1 - p.y) * pol;
        image.ptr<double>(y2)[x1] += a;
    }

    if (valid(x2, y1)) {
        double a = - (x1 - p.x) * (y2 - p.y) * pol;
        image.ptr<double>(y1)[x2] += a;
    }

    if (valid(x2, y2)) {
        double a = (x1 - p.x) * (y1 - p.y) * pol;
        image.ptr<double>(y2)[x2] += a;
    }
}

void Optimizer::intensity(const gsl_vector *vec, shared_ptr<MapPoint>& pMP) const {
    {
        lock_guard<mutex> lock(pMP->mMutexFeatures);
        if (pMP->mPatch.empty())
            pMP->mPatch = cv::Mat::zeros(180, 240, CV_64F);
        // only one frame now !!
        auto KFs = pMP->getObservations();
        for (auto KFit: KFs) {
            okvis::Time t0 = KFit->vEvents.begin()->timeStamp;
            // getEvents?()?
            for (auto EVit: KFit->vEvents) {
                cv::Mat p = (cv::Mat_<double, 3, 1> << EVit->measurement.x ,EVit->measurement.y, 1);
                cv::Mat point_warped;

                // project to first frame
                auto t = EVit->timeStamp - t0;
                warp()
                ;

            }
        }

    }

}
void Optimizer::optimize(shared_ptr<MapPoint>& pMP) {
    // for one map point and n keyframes, variable numbers 3 + n * 5
    int nKFs = pMP->observations();
    auto KFs = pMP->getObservations();
    int nVariables = 2 + nKFs * 6;

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

    for (auto KFit :KFs ) {
        cv::Vec3d w = KFit->getAngularVelocity();
        gsl_vector_set(x, 2 + i * 6, w(0));
        gsl_vector_set(x, 3 + i * 6, w(1));
        gsl_vector_set(x, 4 + i * 6, w(2));

        cv::Vec3d v = KFit->getLinearVelocity();
        gsl_vector_set(x, 5 + i * 6, v(0));
        gsl_vector_set(x, 6 + i * 6, v(1));
        gsl_vector_set(x, 7 + i * 6, v(2));
    }

    /* Set initial step sizes to 1 */
    step_size = gsl_vector_alloc(nVariables);
    // should the step_size be adjusted??
    gsl_vector_set_all(step_size, 1);

    /* Initialize method and iterate */
    minex_func.n = nVariables;
    minex_func.f = variance;
    minex_func.params = &param;

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

        LOG(INFO) << size;

    }
    while (status == GSL_CONTINUE && iter < 100);

    pMP->setNormalDirection(gsl_vector_get(s->x, 0), gsl_vector_get(s->x, 1));

    for (auto KFit :KFs ) {
        cv::Vec3d w{gsl_vector_get(s->x, 2 + i * 6),
                    gsl_vector_get(s->x, 3 + i * 6),
                    gsl_vector_get(s->x, 4 + i * 6)};
        KFit->setAngularVelocity(w);

        cv::Vec3d v{gsl_vector_get(s->x, 5 + i * 6),
                    gsl_vector_get(s->x, 6 + i * 6),
                    gsl_vector_get(s->x, 7 + i * 6)};
        KFit->setLinearVelocity(v);
    }

     // remeber to set the kf and mp pos!!

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}

}
