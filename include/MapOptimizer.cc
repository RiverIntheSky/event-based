#include "MapDrawer.h"

namespace ev {

void MapDrawer::initialize_map() {

    int numMapPoints = frame->mvpMapPoints.size();
    int nVariables = numMapPoints * 3 /* depth and normal */;

    double result[nVariables] = {};

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    int i = 0;
    for (; i < 3 * numMapPoints; i += 3) {
        gsl_vector_set(x, i, 0);
        gsl_vector_set(x, i+1, M_PI);
        gsl_vector_set(x, i+2, 1.);
    }

    set_use_polarity(true);
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

    LOG(INFO) << map->mspMapPoints.size();
}

void MapDrawer::track() {
    LOG(INFO) << "---------------------";

    float th = 0.7 /* threshold */;

    cv::Mat w = frame->getAngularVelocity();
    cv::Mat v = frame->getLinearVelocity();
    cv::Mat Rwc = frame->getRotation();
    cv::Mat Rwc_w = rotm2axang(Rwc);
    cv::Mat twc = frame->getTranslation();

    set_use_polarity(true);
    float overlap1 = overlap(Rwc, twc, w, v);
    LOG(INFO) << overlap1;
//    float overlap1 = 0;

    if (overlap1 < th) {

        std::set<shared_ptr<KeyFrame>, idxOrder> KFs;
        std::set<shared_ptr<MapPoint>> MPs;

        for (auto pMP: map->getAllMapPoints()) {
            //                    if (inFrame(pMP->getWorldPos(), Rwc, twc)) { /* test on planar scene */
            MPs.insert(pMP);
            for (auto kf: pMP->mObservations) {
                KFs.insert(kf);
            }
            //                    }
        }

        int nMPs = MPs.size() /* all the in current frame observable points */;
        int nKFs = KFs.size();
        int nVariables = nMPs * 3;

        double result[nVariables] = {};

        gsl_multimin_fminimizer *s = NULL;
        gsl_vector *x;

        x = gsl_vector_alloc(nVariables);

        int i = 0;
        for (auto mpit = MPs.begin(); mpit != MPs.end(); mpit++, i+= 3) {
            std::array<float, 2> mpN = (*mpit)->getNormalDirection();
            gsl_vector_set(x, i, mpN[0]);
            gsl_vector_set(x, i+1, mpN[1]);
            gsl_vector_set(x, i+2, (*mpit)->d);
        }

        paramSet params{this, &KFs, &MPs, true};
        set_use_polarity(false);
        optimize_gsl(0.1, nVariables, ba, &params, s, x, result, 500);

        gsl_vector *vec;
        vec = gsl_vector_alloc(nVariables);
        for (int i = 0; i < nVariables; i++)
            gsl_vector_set(vec, i, result[i]);
        params.optimize = false;
        ba(vec, &params);
        gsl_vector_free(vec);
        float overlap4 = overlap(tmpFramebuffer, tmpImage, mapFramebuffer, mapImage);
        LOG(INFO) << overlap4;
        if (overlap4 > overlap1) {
            int i = 0;
            for (auto mpit = MPs.begin(); mpit != MPs.end(); mpit++) {
                float phi = result[i++];
                float psi = result[i++];
                (*mpit)->setNormalDirection(phi, psi);
                (*mpit)->d = result[i++];
            }

            auto pKF = make_shared<KeyFrame>(*frame);
            map->addKeyFrame(pKF);
            LOG(INFO) << "new keyframe added";
            // add pMP !!
            for (auto pMP: MPs) {
                cv::Mat pos = pMP->getWorldPos();
                pos = pos /(-pMP->d * pos.dot(pMP->getNormal())); /* n'x + d_ = 0 */
                pMP->setWorldPos(pos);
                pMP->addObservation(pKF);
            }
        }
    }
}

double MapDrawer::initialize_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    int numMapPoints = drawer->frame->mvpMapPoints.size();
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

    cv::Mat w = drawer->frame->getAngularVelocity();
    cv::Mat v = drawer->frame->getLinearVelocity();

    return (double)drawer->initialize_map_draw(nw, depth, w, v);
}

double MapDrawer::frame_cost_func(const gsl_vector *vec, void *params) {
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

double MapDrawer::global_tracking_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
            v = cv::Mat::zeros(3, 1, CV_32F),
            Rwc_w = cv::Mat::zeros(3, 1, CV_32F),
            twc = cv::Mat::zeros(3, 1, CV_32F);

    w.at<float>(0) = gsl_vector_get(vec, 0);
    w.at<float>(1) = gsl_vector_get(vec, 1);
    w.at<float>(2) = gsl_vector_get(vec, 2);

    v.at<float>(0) = gsl_vector_get(vec, 3);
    v.at<float>(1) = gsl_vector_get(vec, 4);
    v.at<float>(2) = gsl_vector_get(vec, 5);

    Rwc_w.at<float>(0) = gsl_vector_get(vec, 6);
    Rwc_w.at<float>(1) = gsl_vector_get(vec, 7);
    Rwc_w.at<float>(2) = gsl_vector_get(vec, 8);

    twc.at<float>(0) = gsl_vector_get(vec, 9);
    twc.at<float>(1) = gsl_vector_get(vec, 10);
    twc.at<float>(2) = gsl_vector_get(vec, 11);

    cv::Mat Rwc = axang2rotm(Rwc_w);

    return (double)drawer->tracking_cost_func(Rwc, twc, w, v);
}

double MapDrawer::ba(const gsl_vector *vec, void *params) {
    paramSet* p = (paramSet *)params;
    MapDrawer* drawer = p->drawer;
    auto KFs = *(p->KFs);
    auto MPs = *(p->MPs);

    int nMPs = MPs.size();
    int nFs = KFs.size() + 1;

    cv::Mat nw = cv::Mat::zeros(3, nMPs, CV_32F);
    std::vector<float> depth(nMPs);

    int i = 0, j = 0;
    for (auto mpit = MPs.begin(); mpit != MPs.end(); mpit++, i++) {
        float phi = float(gsl_vector_get(vec, j++));
        float psi = float(gsl_vector_get(vec, j++));
        nw.at<float>(0, i) = std::cos(phi) * std::sin(psi);
        nw.at<float>(1, i) = std::sin(phi) * std::sin(psi);
        nw.at<float>(2, i) = std::cos(psi);
        depth[i] = float(gsl_vector_get(vec, j++));
    }

    cv::Mat w = cv::Mat::zeros(3, nFs, CV_32F),
            v = cv::Mat::zeros(3, nFs, CV_32F),
            Rwc_w = cv::Mat::zeros(3, nFs, CV_32F),
            twc = cv::Mat::zeros(3, nFs, CV_32F);

    i = 0;
    for (auto KFit: KFs) {
        KFit->getAngularVelocity().copyTo(w.col(i));
        KFit->getLinearVelocity().copyTo(v.col(i));
        cv::Mat Rwc = KFit->getRotation();
        rotm2axang(Rwc).copyTo(Rwc_w.col(i));
        KFit->getTranslation().copyTo(twc.col(i));
        i++;
    }

    drawer->frame->getAngularVelocity().copyTo(w.col(i));
    drawer->frame->getLinearVelocity().copyTo(v.col(i));
    cv::Mat Rwc = drawer->frame->getRotation();
    rotm2axang(Rwc).copyTo(Rwc_w.col(i));
    drawer->frame->getTranslation().copyTo(twc.col(i));

    return (double)drawer->optimize_map_draw(p, nw, depth, Rwc_w, twc, w, v);
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

    return (double)drawer->tracking_cost_func(w, v);
}

//double MapDrawer::optimize_cost_func(const gsl_vector *vec, void *params) {
//    MapDrawer* drawer = (MapDrawer *)params;

//    int numMapPoints = drawer->map->mspMapPoints.size();
//    cv::Mat nw = cv::Mat::zeros(3, numMapPoints, CV_32F);
//    std::vector<float> depth(numMapPoints);

//    auto mpit = drawer->map->mspMapPoints.begin(); int i = 0;
//    double phi = gsl_vector_get(vec, 0);
//    double psi = gsl_vector_get(vec, 1);
//    depth[0] = (*mpit)->d;
//    nw.at<float>(0, i) = std::cos(phi) * std::sin(psi);
//    nw.at<float>(1, i) = std::sin(phi) * std::sin(psi);
//    nw.at<float>(2, i) = std::cos(psi);

//    for (mpit++, i++; mpit != drawer->map->mspMapPoints.end(); mpit++, i++) {
//        phi = gsl_vector_get(vec, 3*i-1);
//        psi = gsl_vector_get(vec, 3*i);
//        depth[i] = gsl_vector_get(vec, 3*i+1);
//    }

//    i = i*3-1;
//    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
//            v = cv::Mat::zeros(3, 1, CV_32F);
//    w.at<float>(0) = gsl_vector_get(vec, i); i++;
//    w.at<float>(1) = gsl_vector_get(vec, i); i++;
//    w.at<float>(2) = gsl_vector_get(vec, i); i++;

//    v.at<float>(0) = gsl_vector_get(vec, i); i++;
//    v.at<float>(1) = gsl_vector_get(vec, i); i++;
//    v.at<float>(2) = gsl_vector_get(vec, i);

//    return (double)drawer->optimize_map_draw(nw, depth, w, v);
//}

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

//        LOG(INFO) << "size " << size;

    }
    while (status == GSL_CONTINUE && it < iter);

    for (int i = 0; i < nv; i++)
        res[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}
}
