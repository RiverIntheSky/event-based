#include "MapDrawer.h"

namespace ev {

void MapDrawer::initialize_map() {
    updateFrame();

    int numMapPoints = map->mspMapPoints.size();
    int nVariables = numMapPoints * 3 /* depth and normal */
                                  + 5 /* degrees of freedom */;
    double result[nVariables] = {};

    double w[] = {0, 0, 0};
    double v[] = {0, M_PI/2}; /* normalized velocity direction */
                              /* could not represent zero velocity */

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    int i = 0;
    for (; i < 3 * numMapPoints; i += 3) {
        gsl_vector_set(x, i, 0.);
        gsl_vector_set(x, i+1, M_PI);
        gsl_vector_set(x, i+2, 0.01);
    }

    gsl_vector_set(x, i, w[0]); i++;
    gsl_vector_set(x, i, w[1]); i++;
    gsl_vector_set(x, i, w[2]); i++;

    gsl_vector_set(x, i, v[0]); i++;
    gsl_vector_set(x, i, v[1]); i++;

    optimize_gsl(1, nVariables, initialize_cost_func, this, s, x, result, 500);

//    w.at<double>(0) = result[0];
//    w.at<double>(1) = result[1];
//    w.at<double>(2) = result[2];

//    v.at<double>(0) = result[3];
//    v.at<double>(1) = result[4];

//    frame->setAngularVelocity(w);
//    v *= frame->mScale;
//    frame->setLinearVelocity(v);

//    double dt = frame->dt;
//    cv::Mat dw = w * dt;
//    cv::Mat Rc1c2 = axang2rotm(dw);
//    cv::Mat tc1c2 = v * dt; // up to a global scale
//    cv::Mat Twc1 = frame->getFirstPose();
//    cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
//    cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
//    cv::Mat Rwc2 = Rwc1 * Rc1c2;
//    cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
//    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_64F);
//    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
//    twc2.copyTo(Twc2.rowRange(0,3).col(3));
//    frame->setLastPose(Twc2);


}

double MapDrawer::initialize_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    int numMapPoints = drawer->map->mspMapPoints.size();
    cv::Mat nw = cv::Mat::zeros(3, numMapPoints, CV_32F);
    std::vector<float> depth(numMapPoints);

    int i = 0;
    for (; i < numMapPoints; i++) {
        double phi = gsl_vector_get(vec, 3*i);
        double psi = gsl_vector_get(vec, 3*i+1);
        nw.at<float>(0, i) = std::cos(phi) * std::sin(psi);
        nw.at<float>(1, i) = std::sin(phi) * std::sin(psi);
        nw.at<float>(2, i) = std::cos(psi);
        depth[i] = gsl_vector_get(vec, 3*i+2);
    }

    i*=3;
    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
            v = cv::Mat::zeros(3, 1, CV_32F);
    w.at<float>(0) = gsl_vector_get(vec, i); i++;
    w.at<float>(1) = gsl_vector_get(vec, i); i++;
    w.at<float>(2) = gsl_vector_get(vec, i); i++;

    double phi = gsl_vector_get(vec, i); i++;
    double psi = gsl_vector_get(vec, i);
    v.at<float>(0) = std::cos(phi) * std::sin(psi);
    v.at<float>(1) = std::sin(phi) * std::sin(psi);
    v.at<float>(2) = std::cos(psi);

    return (double)drawer->initialize_map_draw(nw, depth, w, v);
}

void MapDrawer::optimize_gsl(double ss, int nv, double (*f)(const gsl_vector*, void*), void *params,
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

        LOG(INFO) << size;

    }
    while (status == GSL_CONTINUE && it < iter);

    for (int i = 0; i < nv; i++)
        res[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}
}
