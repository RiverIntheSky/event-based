#include "MapDrawer.h"

namespace ev {

void MapDrawer::initialize_map() {

    int numMapPoints = frame->mvpMapPoints.size();
    int nVariables = 1;
    double result[nVariables] = {};

    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat v = cv::Mat::zeros(3, 1, CV_32F);

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    gsl_vector_set(x, 0, v.at<float>(0));

    optimize_gsl(1, nVariables, initialize_cost_func, this, s, x, result, 500);

    gsl_vector *vec;
    vec = gsl_vector_alloc(nVariables);
    for (int i = 0; i < nVariables; i++)
        gsl_vector_set(vec, i, result[i]);
    initialize_cost_func(vec, this);
    gsl_vector_free(vec);
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_RECTANGLE, blurredImage);
        glUseProgram(quadShader);
        drawQuad();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    v.at<float>(0) = float(result[0]);

    frame->setAngularVelocity(w);
    frame->setLinearVelocity(v);
   
    float dt = frame->dt;

    cv::Mat dw = w * dt;
    cv::Mat Rc1c2 = axang2rotm(dw);
    cv::Mat tc1c2 = v * dt;
    cv::Mat Twc1 = frame->getFirstPose();
    cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
    cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
    cv::Mat Rwc2 = Rwc1 * Rc1c2;
    cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_32F);
    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
    twc2.copyTo(Twc2.rowRange(0,3).col(3));
    frame->setLastPose(Twc2);
    frame->mpMap->mspMapPoints.insert(*(frame->mvpMapPoints.begin()));
}

void MapDrawer::track() {
    int nVariables = 6 /* degrees of freedom */;

    double result[nVariables] = {};

    cv::Mat w = tracking->w.clone();
    cv::Mat v = tracking->v.clone();

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    gsl_vector_set(x, 0, w.at<float>(0));
    gsl_vector_set(x, 1, w.at<float>(1));
    gsl_vector_set(x, 2, w.at<float>(2));

    gsl_vector_set(x, 3, v.at<float>(0));
    gsl_vector_set(x, 4, v.at<float>(1));
    gsl_vector_set(x, 5, v.at<float>(2));

    optimize_gsl(1, nVariables, tracking_cost_func, this, s, x, result, 500);

    w.at<float>(0) = float(result[0]);
    w.at<float>(1) = float(result[1]);
    w.at<float>(2) = float(result[2]);

    v.at<float>(0) = float(result[3]);
    v.at<float>(1) = float(result[4]);
    v.at<float>(2) = float(result[5]);

    LOG(INFO) << "w\n"<<w;
    LOG(INFO) << "v\n"<<v;
    LOG(INFO) << "T\n"<<frame->getFirstPose();

    frame->setAngularVelocity(w);
    frame->setLinearVelocity(v);

    float dt = frame->dt;
    cv::Mat dw = w * dt;
    cv::Mat Rc1c2 = axang2rotm(dw);
    cv::Mat tc1c2 = v * dt;
    cv::Mat Twc1 = frame->getFirstPose();
    cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
    cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
    cv::Mat Rwc2 = Rwc1 * Rc1c2;
    cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_32F);
    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
    twc2.copyTo(Twc2.rowRange(0,3).col(3));
    frame->setLastPose(Twc2);
}

void MapDrawer::optimize_frame() {
    int numMapPoints = map->mspMapPoints.size();
    int numFrames = map->mspFrames.size();
    int nVariables = numMapPoints * 3 - 1 /* depth and normal */
                      + numFrames * 6 /* degrees of freedom */;

    double result[nVariables] = {};

    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat v_phi = (cv::Mat_<float>(2, 1) << 0, M_PI/2); /* normalized velocity direction */
                              /* could not represent zero velocity */

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    int i = 0;
    for (; i < 3 * numMapPoints; i += 3) {
        gsl_vector_set(x, i, 0);
        gsl_vector_set(x, i+1, M_PI);
        gsl_vector_set(x, i+2, 0.01);
    }

    gsl_vector_set(x, i, w.at<float>(0)); i++;
    gsl_vector_set(x, i, w.at<float>(1)); i++;
    gsl_vector_set(x, i, w.at<float>(2)); i++;

    gsl_vector_set(x, i, v_phi.at<float>(0)); i++;
    gsl_vector_set(x, i, v_phi.at<float>(1));

    optimize_gsl(1, nVariables, initialize_cost_func, this, s, x, result, 500);

    float depth_norm = 0.;

    for (i = 0; i < 3 * numMapPoints; i += 3) {
        float d = result[i+2];
        if (d > 0)
            depth_norm += 1/(d*d);
    }
    depth_norm = std::sqrt(depth_norm);

    auto it = (frame->mvpMapPoints).begin();
    for (i = 0; i < 3 * numMapPoints; i += 3) {
        float d = float(result[i+2]);
        auto current = it++;
        if (d > 0) {
            (*current)->setNormalDirection(float(result[i]), float(result[i+1]));
            (*current)->d = float(result[i+2]) * depth_norm;
            map->mspMapPoints.insert(*current);
        } else {
            frame->mvpMapPoints.erase(current);
        }
    }

    w.at<float>(0) = float(result[i]); i++;
    w.at<float>(1) = float(result[i]); i++;
    w.at<float>(2) = float(result[i]); i++;

    float phi = float(result[i]); i++;
    float psi = float(result[i]);
    cv::Mat v_normalized = (cv::Mat_<float>(3, 1) << std::cos(phi) * std::sin(psi),
                                                      std::sin(phi) * std::sin(psi),
                                                      std::cos(psi));
    v_normalized /= depth_norm;

    frame->setAngularVelocity(w);
    frame->setLinearVelocity(v_normalized);

    float dt = frame->dt;

    cv::Mat dw = w * dt;
    cv::Mat Rc1c2 = axang2rotm(dw);
    cv::Mat tc1c2 = v_normalized * dt;
    cv::Mat Twc1 = frame->getFirstPose();
    cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
    cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
    cv::Mat Rwc2 = Rwc1 * Rc1c2;
    cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_32F);
    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
    twc2.copyTo(Twc2.rowRange(0,3).col(3));
    frame->setLastPose(Twc2);
}

void MapDrawer::optimize_map() {
    int numMapPoints = map->mspMapPoints.size();
    int numFrames = map->mspFrames.size();

    int nVariables = numFrames;

    double result[nVariables] = {};

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    int i = 0;
    for (auto f: map->mspFrames) {
        cv::Mat v = f->getLinearVelocity();

        gsl_vector_set(x, i, v.at<float>(0)); i++;
    }

    optimize_gsl(0.5, nVariables, sliding_window_cost_func, this, s, x, result, 500);

    gsl_vector *vec;
    vec = gsl_vector_alloc(nVariables);
    for (i = 0; i < nVariables; i++)
        gsl_vector_set(vec, i, result[i]);
    sliding_window_cost_func(vec, this);
    gsl_vector_free(vec);
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_RECTANGLE, blurredImage);
        glUseProgram(quadShader);
        drawQuad();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    i = 0;
    cv::Mat Twc1 = cv::Mat::eye(4,4,CV_32F);
    for (auto f: map->mspFrames) {
        cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
                v = cv::Mat::zeros(3, 1, CV_32F);

        v.at<float>(0) = float(result[i]); i++;
        LOG(INFO) << v.at<float>(0);

        f->setAngularVelocity(w);
        f->setLinearVelocity(v);

        float dt = f->dt;

        cv::Mat dw = w * dt;
        cv::Mat Rc1c2 = axang2rotm(dw);
        cv::Mat tc1c2 = v * dt;
        cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
        cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
        cv::Mat Rwc2 = Rwc1 * Rc1c2;
        cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
        cv::Mat Twc2 = cv::Mat::eye(4,4,CV_32F);
        Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
        twc2.copyTo(Twc2.rowRange(0,3).col(3));
        f->setFirstPose(Twc1);
        f->setLastPose(Twc2);
        Twc2.copyTo(Twc1);
    }
}

//void MapDrawer::optimize_frame() {

//    int numMapPoints = map->mspMapPoints.size();

//    int nVariables = numMapPoints * 3 /* depth and normal */
//                                  + 5 /* degrees of freedom */;
//    double result[nVariables] = {};

//    cv::Mat w = tracking->w.clone();
//    cv::Mat v = tracking->v.clone();

//    gsl_multimin_fminimizer *s = NULL;
//    gsl_vector *x;

//    /* Starting point */
//    x = gsl_vector_alloc(nVariables);

//    auto mpit = map->mspMapPoints.begin();    int i = 0;
//    std::array<float, 2> mpN = (*mpit)->getNormalDirection();
//    gsl_vector_set(x, i, mpN[i]); i++;
//    gsl_vector_set(x, i, mpN[i]); i++;
//    for (mpit++; mpit != map->mspMapPoints.end(); mpit++) {
//        mpN = (*mpit)->getNormalDirection();
//        gsl_vector_set(x, i, mpN[0]);
//        gsl_vector_set(x, i+1, mpN[1]);
//        gsl_vector_set(x, i+2, (*mpit)->d);
//        i+=3;
//    }

//    gsl_vector_set(x, i, w.at<float>(0)); i++;
//    gsl_vector_set(x, i, w.at<float>(1)); i++;
//    gsl_vector_set(x, i, w.at<float>(2)); i++;

//    gsl_vector_set(x, i, v.at<float>(0)); i++;
//    gsl_vector_set(x, i, v.at<float>(1)); i++;
//    gsl_vector_set(x, i, v.at<float>(2));

//    optimize_gsl(1, nVariables, optimize_cost_func, this, s, x, result, 500);

//    auto it = map->mspMapPoints.begin();
//    (*it)->setNormalDirection(float(result[0]), float(result[1]));
//    for (i = 2, it++; it != map->mspMapPoints.end();i+=3) {
//        float d = float(result[i+2]);
//        auto current = it++;
//        if (d > 0) {
//            (*current)->setNormalDirection(float(result[i]), float(result[i+1]));
//            (*current)->d = d;
//        } else {
//            map->mspMapPoints.erase(current);
//        }
//    }

//    w.at<float>(0) = float(result[i]); i++;
//    w.at<float>(1) = float(result[i]); i++;
//    w.at<float>(2) = float(result[i]); i++;

//    v.at<float>(0) = float(result[i]); i++;
//    v.at<float>(1) = float(result[i]); i++;
//    v.at<float>(2) = float(result[i]);

//    frame->setAngularVelocity(w);
//    frame->setLinearVelocity(v);
//}

double MapDrawer::initialize_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    cv::Mat nw = cv::Mat::zeros(3, 1, CV_32F);
    std::vector<float> depth(1);

        float phi = 0.;
        float psi = M_PI;
        nw.at<float>(0) = std::cos(phi) * std::sin(psi);
        nw.at<float>(1) = std::sin(phi) * std::sin(psi);
        nw.at<float>(2) = std::cos(psi);
        depth[0] = 1.;

    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
            v = cv::Mat::zeros(3, 1, CV_32F);

    v.at<float>(0) = gsl_vector_get(vec, 0);

    return (double)drawer->initialize_map_draw(nw, depth, w, v);
}

double MapDrawer::tracking_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
            v = cv::Mat::zeros(3, 1, CV_32F);

    w.at<float>(0) = gsl_vector_get(vec, 0);
    w.at<float>(1) = gsl_vector_get(vec, 1);
    w.at<float>(2) = gsl_vector_get(vec, 2);

    v.at<float>(0) = gsl_vector_get(vec, 3);
    v.at<float>(1) = gsl_vector_get(vec, 4);
    v.at<float>(2) = gsl_vector_get(vec, 5);

    return (double)drawer->cost_func(w, v);
}

double MapDrawer::optimize_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    int numMapPoints = drawer->map->mspMapPoints.size();
    cv::Mat nw = cv::Mat::zeros(3, numMapPoints, CV_32F);
    std::vector<float> depth(numMapPoints);

    auto mpit = drawer->map->mspMapPoints.begin(); int i = 0;
    double phi = gsl_vector_get(vec, 0);
    double psi = gsl_vector_get(vec, 1);
    depth[0] = (*mpit)->d;
    nw.at<float>(0, i) = std::cos(phi) * std::sin(psi);
    nw.at<float>(1, i) = std::sin(phi) * std::sin(psi);
    nw.at<float>(2, i) = std::cos(psi);

    for (mpit++, i++; mpit != drawer->map->mspMapPoints.end(); mpit++, i++) {
        phi = gsl_vector_get(vec, 3*i-1);
        psi = gsl_vector_get(vec, 3*i);
        depth[i] = gsl_vector_get(vec, 3*i+1);
    }

    i = i*3-1;
    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
            v = cv::Mat::zeros(3, 1, CV_32F);
    w.at<float>(0) = gsl_vector_get(vec, i); i++;
    w.at<float>(1) = gsl_vector_get(vec, i); i++;
    w.at<float>(2) = gsl_vector_get(vec, i); i++;

    v.at<float>(0) = gsl_vector_get(vec, i); i++;
    v.at<float>(1) = gsl_vector_get(vec, i); i++;
    v.at<float>(2) = gsl_vector_get(vec, i);

    return (double)drawer->optimize_map_draw(nw, depth, w, v);
}

double MapDrawer::sliding_window_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    int numMapPoints = drawer->map->mspMapPoints.size();
    int numFrames = drawer->map->mspFrames.size();

    cv::Mat nw = cv::Mat::zeros(3, 1, CV_32F);
    std::vector<float> depth(1);

    auto mpit = drawer->map->mspMapPoints.begin(); int i = 0;
    float phi = 0;
    float psi = M_PI;
    depth[0] = (*mpit)->d;
    nw.at<float>(0) = std::cos(phi) * std::sin(psi);
    nw.at<float>(1) = std::sin(phi) * std::sin(psi);
    nw.at<float>(2) = std::cos(psi);

    cv::Mat w = cv::Mat::zeros(3, numFrames, CV_32F),
            v = cv::Mat::zeros(3, numFrames, CV_32F);

    for (i = 0; i < numFrames; i++) {
        v.at<float>(0, i) = gsl_vector_get(vec, i);
    }

    return (double)drawer->optimize_map_draw(nw, depth, w, v);
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

    }
    while (status == GSL_CONTINUE && it < iter);

    for (int i = 0; i < nv; i++)
        res[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}
}
