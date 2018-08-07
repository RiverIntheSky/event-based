#include "MapDrawer.h"

namespace ev {

void MapDrawer::initialize_map() {

    int numMapPoints = frame->mvpMapPoints.size();
    int nVariables = numMapPoints * 3 - 1/* depth and normal */
                                  + 6 /* degrees of freedom */;
    double result[nVariables] = {};

    cv::Mat w = frame->getAngularVelocity();
    cv::Mat v = frame->getLinearVelocity();

    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *x;

    /* Starting point */
    x = gsl_vector_alloc(nVariables);

    int i = 0;

    float phi = 0;
    float psi = M_PI;
    gsl_vector_set(x, i++, phi);
    gsl_vector_set(x, i++, psi);

    for (; i < 3 * numMapPoints-1;) {
        gsl_vector_set(x, i++, phi);
        gsl_vector_set(x, i++, psi);
        gsl_vector_set(x, i++, 1);
    }

    gsl_vector_set(x, i++, w.at<float>(0));
    gsl_vector_set(x, i++, w.at<float>(1));
    gsl_vector_set(x, i++, w.at<float>(2));

    gsl_vector_set(x, i++, v.at<float>(0));
    gsl_vector_set(x, i++, v.at<float>(1));
    gsl_vector_set(x, i++, v.at<float>(2));

    set_use_polarity(true);
    optimize_gsl(1, nVariables, initialize_cost_func, this, s, x, result, 500);

    auto it = (frame->mvpMapPoints).begin();
    (*it)->setNormalDirection(float(result[0]), float(result[1]));
    map->mspMapPoints.insert(*it);

    for (i = 2, it++; i < 3 * numMapPoints-1; i += 3) {
        float d = float(result[i+2]);
        auto current = it++;

        if (d > 0) {
            (*current)->setNormalDirection(float(result[i]), float(result[i+1]));
            (*current)->d = float(result[i+2]);
            map->mspMapPoints.insert(*current);
        } else {
            frame->mvpMapPoints.erase(current);
        }
    }

    w.at<float>(0) = float(result[i++]);
    w.at<float>(1) = float(result[i++]);
    w.at<float>(2) = float(result[i++]);

    v.at<float>(0) = float(result[i++]);
    v.at<float>(1) = float(result[i++]);
    v.at<float>(2) = float(result[i++]);

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

void MapDrawer::track() {
    LOG(INFO) << "---------------------";
    int nVariables = 6 /* degrees of freedom */;
    float th = 0.8 /* threshold */;

    double result[nVariables] = {};

    cv::Mat w = frame->getAngularVelocity();
    cv::Mat v = frame->getLinearVelocity();
    cv::Mat Rwc = frame->getRotation();
    cv::Mat Rwc_w = rotm2axang(Rwc);
    cv::Mat twc = frame->getTranslation();
    // optimize in frame
    {
        draw_map_patch();
        set_use_polarity(true);
        gsl_multimin_fminimizer *s = NULL;
        gsl_vector *x;

        /* Starting point */
        x = gsl_vector_alloc(nVariables);

        int i = 0;
        gsl_vector_set(x, i++, w.at<float>(0));
        gsl_vector_set(x, i++, w.at<float>(1));
        gsl_vector_set(x, i++, w.at<float>(2));

        gsl_vector_set(x, i++, v.at<float>(0));
        gsl_vector_set(x, i++, v.at<float>(1));
        gsl_vector_set(x, i++, v.at<float>(2));

        optimize_gsl(0.1, nVariables, frame_cost_func, this, s, x, result, 500);

        i = 0;
        w.at<float>(0) = float(result[i++]);
        w.at<float>(1) = float(result[i++]);
        w.at<float>(2) = float(result[i++]);

        v.at<float>(0) = float(result[i++]);
        v.at<float>(1) = float(result[i++]);
        v.at<float>(2) = float(result[i++]);
    }

    float overlap1 = overlap(Rwc, twc, w, v);
    LOG(INFO) << overlap1;

    if (overlap1 < 1) {
        // match to map
        set_use_polarity(false);
        draw_map_texture(mapFramebuffer);
        draw_map_patch();

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

        cv::Mat w_ = (cv::Mat_<float>(3, 1) << result[0], result[1], result[2]);
        cv::Mat v_ = (cv::Mat_<float>(3, 1) << result[3], result[4], result[5]);

        float overlap2 = overlap(Rwc, twc, w_, v_);
        LOG(INFO) << overlap2;

        if (overlap2 > overlap1) {
            w_.copyTo(w);
            v_.copyTo(v);
        }

//        if (overlap2 < th) {
//            // correct pose
//            double result[nVariables*2] = {};
//            set_use_polarity(false);

//            gsl_multimin_fminimizer *s = NULL;
//            gsl_vector *x;

//            /* Starting point */
//            x = gsl_vector_alloc(nVariables * 2);

//            int i = 0;
//            gsl_vector_set(x, i++, w.at<float>(0));
//            gsl_vector_set(x, i++, w.at<float>(1));
//            gsl_vector_set(x, i++, w.at<float>(2));

//            gsl_vector_set(x, i++, v.at<float>(0));
//            gsl_vector_set(x, i++, v.at<float>(1));
//            gsl_vector_set(x, i++, v.at<float>(2));

//            gsl_vector_set(x, i++, Rwc_w.at<float>(0));
//            gsl_vector_set(x, i++, Rwc_w.at<float>(1));
//            gsl_vector_set(x, i++, Rwc_w.at<float>(2));

//            gsl_vector_set(x, i++, twc.at<float>(0));
//            gsl_vector_set(x, i++, twc.at<float>(1));
//            gsl_vector_set(x, i++, twc.at<float>(2));

//            optimize_gsl(0.1, nVariables*2, global_tracking_cost_func, this, s, x, result, 500);

//            cv::Mat w_ = (cv::Mat_<float>(3, 1) << result[0], result[1], result[2]);
//            cv::Mat v_ = (cv::Mat_<float>(3, 1) << result[3], result[4], result[5]);
//            cv::Mat Rwc_w_ = (cv::Mat_<float>(3, 1) << result[6], result[7], result[8]);
//            cv::Mat Rwc_ = axang2rotm(Rwc_w_);
//            cv::Mat twc_ = (cv::Mat_<float>(3, 1) << result[9], result[10], result[11]);

//            float overlap3 = overlap(Rwc_, twc_, w_, v_);
//            LOG(INFO) << overlap3;

//            if (overlap3 > overlap2 && overlap3 > overlap1) {
//                w_.copyTo(w);
//                v_.copyTo(v);
//                Rwc_.copyTo(Rwc);
//                twc_.copyTo(twc);
//            }

        if (overlap2 < th) {

            std::set<shared_ptr<KeyFrame>, idxOrder> KFs;

            for (auto pMP: map->getAllMapPoints()) {
                float x, y;
                if (inFrame(pMP->getWorldPos(), Rwc, twc, x, y)) {
                    frame->mvpMapPoints.insert(pMP);
                    for (auto kf: pMP->mObservations) {
                        KFs.insert(kf);
                    }
                }
            }

            std::set<shared_ptr<MapPoint>> MPs(frame->mvpMapPoints);

            int nMPs = MPs.size() /* all the in current frame observable points */;
            int nKFs = KFs.size();
            nVariables = nMPs * 3 - 1 /* fix the distance of the first point */
                    + nKFs * 12 - 6 /* all but the pose of the first kf + velocity */
                    + 12            /* current frame */;

            double result[nVariables] = {};

            gsl_multimin_fminimizer *s = NULL;
            gsl_vector *x;

            x = gsl_vector_alloc(nVariables);

            auto mpit = MPs.begin(); int i = 0;
            std::array<float, 2> mpN = (*mpit)->getNormalDirection();
            gsl_vector_set(x, i, mpN[i]); i++;
            gsl_vector_set(x, i, mpN[i]); i++;
            for (mpit++; mpit != MPs.end(); mpit++) {
                mpN = (*mpit)->getNormalDirection();
                gsl_vector_set(x, i, mpN[0]);
                gsl_vector_set(x, i+1, mpN[1]);
                gsl_vector_set(x, i+2, (*mpit)->d);
                i+=3;
            }

            for (auto KFit: KFs) {
                auto w_ = KFit->getAngularVelocity();
                gsl_vector_set(x, i++, w_.at<float>(0));
                gsl_vector_set(x, i++, w_.at<float>(1));
                gsl_vector_set(x, i++, w_.at<float>(2));

                auto v_ = KFit->getLinearVelocity();
                gsl_vector_set(x, i++, v_.at<float>(0));
                gsl_vector_set(x, i++, v_.at<float>(1));
                gsl_vector_set(x, i++, v_.at<float>(2));

                if  (KFit->mnId != (*KFs.begin())->mnId) {
                    auto Rwc_ = KFit->getRotation();
                    auto Rwc_w_ = rotm2axang(Rwc_);
                    gsl_vector_set(x, i++, Rwc_w_.at<float>(0));
                    gsl_vector_set(x, i++, Rwc_w_.at<float>(1));
                    gsl_vector_set(x, i++, Rwc_w_.at<float>(2));

                    auto twc_ = KFit->getTranslation();
                    gsl_vector_set(x, i++, twc_.at<float>(0));
                    gsl_vector_set(x, i++, twc_.at<float>(1));
                    gsl_vector_set(x, i++, twc_.at<float>(2));
                }
            }

            gsl_vector_set(x, i++, w.at<float>(0));
            gsl_vector_set(x, i++, w.at<float>(1));
            gsl_vector_set(x, i++, w.at<float>(2));

            gsl_vector_set(x, i++, v.at<float>(0));
            gsl_vector_set(x, i++, v.at<float>(1));
            gsl_vector_set(x, i++, v.at<float>(2));

            gsl_vector_set(x, i++, Rwc_w.at<float>(0));
            gsl_vector_set(x, i++, Rwc_w.at<float>(1));
            gsl_vector_set(x, i++, Rwc_w.at<float>(2));

            gsl_vector_set(x, i++, twc.at<float>(0));
            gsl_vector_set(x, i++, twc.at<float>(1));
            gsl_vector_set(x, i++, twc.at<float>(2));

            paramSet params{this, &KFs, &MPs, true};
            optimize_gsl(0.1, nVariables, ba, &params, s, x, result, 500);

            gsl_vector *vec;
            vec = gsl_vector_alloc(nVariables);
            for (int i = 0; i < nVariables; i++)
                gsl_vector_set(vec, i, result[i]);
            params.optimize = false;
            ba(vec, &params);
            gsl_vector_free(vec);

            i = 0;
            LOG(INFO) << "currently tracked points " << MPs.size();
            for (auto mpit = MPs.begin();mpit != MPs.end();) {
                auto current = mpit++;
                if (!((*current)->dirty)) {
                    float phi = result[i++];
                    float psi = result[i++];
                    (*current)->setNormalDirection(phi, psi);
                    if (i != 2) {
                        (*current)->d = result[i++];
                    }
                } else {
                    if (i == 0) {
                        i+=2;
                    } else {
                        i+=3;
                    }
                    MPs.erase(*current);
                    frame->mvpMapPoints.erase(*current);
                }
            }

            LOG(INFO) << "after overlap checking " << MPs.size();

            if (MPs.size() < 5) {
                tracking->mState = tracking->LOST;
            } else {
                frame->shouldBeKeyFrame = true;

                for (auto KFit: KFs) {
                    w.at<float>(0) = float(result[i++]);
                    w.at<float>(1) = float(result[i++]);
                    w.at<float>(2) = float(result[i++]);
                    KFit->setAngularVelocity(w);

                    v.at<float>(0) = float(result[i++]);
                    v.at<float>(1) = float(result[i++]);
                    v.at<float>(2) = float(result[i++]);
                    KFit->setLinearVelocity(v);

                    if  (KFit->mnId != (*KFs.begin())->mnId) {
                        Rwc_w.at<float>(0) = float(result[i++]);
                        Rwc_w.at<float>(1) = float(result[i++]);
                        Rwc_w.at<float>(2) = float(result[i++]);
                        Rwc = axang2rotm(Rwc_w);

                        twc.at<float>(0) = float(result[i++]);
                        twc.at<float>(1) = float(result[i++]);
                        twc.at<float>(2) = float(result[i++]);

                        cv::Mat Twc1 = cv::Mat::eye(4,4,CV_32F);
                        Rwc.copyTo(Twc1.rowRange(0,3).colRange(0,3));
                        twc.copyTo(Twc1.rowRange(0,3).col(3));
                        KFit->setFirstPose(Twc1);
                    }

                    float dt = KFit->dt;
                    cv::Mat dw = w * dt;
                    cv::Mat Rc1c2 = axang2rotm(dw);
                    cv::Mat tc1c2 = v * dt;
                    cv::Mat Twc1 = KFit->getFirstPose();
                    cv::Mat Rwc1 = Twc1.rowRange(0,3).colRange(0,3);
                    cv::Mat twc1 = Twc1.rowRange(0,3).col(3);
                    cv::Mat Rwc2 = Rwc1 * Rc1c2;
                    cv::Mat twc2 = Rwc1 * tc1c2 + twc1;
                    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_32F);
                    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
                    twc2.copyTo(Twc2.rowRange(0,3).col(3));
                    KFit->setLastPose(Twc2);
                }

                w.at<float>(0) = float(result[i++]);
                w.at<float>(1) = float(result[i++]);
                w.at<float>(2) = float(result[i++]);

                v.at<float>(0) = float(result[i++]);
                v.at<float>(1) = float(result[i++]);
                v.at<float>(2) = float(result[i++]);

                Rwc_w.at<float>(0) = float(result[i++]);
                Rwc_w.at<float>(1) = float(result[i++]);
                Rwc_w.at<float>(2) = float(result[i++]);
                Rwc = axang2rotm(Rwc_w);

                twc.at<float>(0) = float(result[i++]);
                twc.at<float>(1) = float(result[i++]);
                twc.at<float>(2) = float(result[i++]);

                for (auto mpit = MPs.begin();mpit != MPs.end(); mpit++) {
                    map->mspMapPoints.insert(*mpit);
                }
            }
        }
    }

    frame->setAngularVelocity(w);
    frame->setLinearVelocity(v);

    float dt = frame->dt;
    cv::Mat dw = w * dt;
    cv::Mat Rc1c2 = axang2rotm(dw);
    cv::Mat tc1c2 = v * dt;
    cv::Mat Twc1 = cv::Mat::eye(4,4,CV_32F);

    Rwc.copyTo(Twc1.rowRange(0,3).colRange(0,3));
    twc.copyTo(Twc1.rowRange(0,3).col(3));
    frame->setFirstPose(Twc1);

    cv::Mat Rwc2 = Rwc * Rc1c2;
    cv::Mat twc2 = Rwc * tc1c2 + twc;
    cv::Mat Twc2 = cv::Mat::eye(4,4,CV_32F);
    Rwc2.copyTo(Twc2.rowRange(0,3).colRange(0,3));
    twc2.copyTo(Twc2.rowRange(0,3).col(3));
    frame->setLastPose(Twc2);
}

double MapDrawer::initialize_cost_func(const gsl_vector *vec, void *params) {
    MapDrawer* drawer = (MapDrawer *)params;

    int numMapPoints = drawer->frame->mvpMapPoints.size();
    cv::Mat nw = cv::Mat::zeros(3, numMapPoints, CV_32F);
    std::vector<float> depth(numMapPoints);

    int i = 0;
    for (int j = 0; j < numMapPoints; j++) {
        float phi = gsl_vector_get(vec, i++);
        float psi = gsl_vector_get(vec, i++);
        nw.at<float>(0, j) = std::cos(phi) * std::sin(psi);
        nw.at<float>(1, j) = std::sin(phi) * std::sin(psi);
        nw.at<float>(2, j) = std::cos(psi);
        if (i != 2) {
            depth[j] = gsl_vector_get(vec, i++);
        } else {
            depth[j] = 1.;
        }
    }

    cv::Mat w = cv::Mat::zeros(3, 1, CV_32F),
            v = cv::Mat::zeros(3, 1, CV_32F);
    w.at<float>(0) = gsl_vector_get(vec, i++);
    w.at<float>(1) = gsl_vector_get(vec, i++);
    w.at<float>(2) = gsl_vector_get(vec, i++);

    v.at<float>(0) = gsl_vector_get(vec, i++);
    v.at<float>(1) = gsl_vector_get(vec, i++);
    v.at<float>(2) = gsl_vector_get(vec, i++);

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

    auto mpit = MPs.begin(); int i = 0, j = 0;
    float phi = float(gsl_vector_get(vec, j++));
    float psi = float(gsl_vector_get(vec, j++));
    depth[0] = (*mpit)->d;
    nw.at<float>(0, i) = std::cos(phi) * std::sin(psi);
    nw.at<float>(1, i) = std::sin(phi) * std::sin(psi);
    nw.at<float>(2, i) = std::cos(psi);

    for (mpit++, i++; mpit != MPs.end(); mpit++, i++) {
        phi = float(gsl_vector_get(vec, j++));
        psi = float(gsl_vector_get(vec, j++));
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
        w.at<float>(0, i) = gsl_vector_get(vec, j++);
        w.at<float>(1, i) = gsl_vector_get(vec, j++);
        w.at<float>(2, i) = gsl_vector_get(vec, j++);

        v.at<float>(0, i) = gsl_vector_get(vec, j++);
        v.at<float>(1, i) = gsl_vector_get(vec, j++);
        v.at<float>(2, i) = gsl_vector_get(vec, j++);

        if  (KFit->mnId != (*KFs.begin())->mnId) {
            Rwc_w.at<float>(0, i) = gsl_vector_get(vec, j++);
            Rwc_w.at<float>(1, i) = gsl_vector_get(vec, j++);
            Rwc_w.at<float>(2, i) = gsl_vector_get(vec, j++);

            twc.at<float>(0, i) = gsl_vector_get(vec, j++);
            twc.at<float>(1, i) = gsl_vector_get(vec, j++);
            twc.at<float>(2, i) = gsl_vector_get(vec, j++);
        } else {
            Rwc_w.at<float>(0, i) = 0.f; Rwc_w.at<float>(1, i) = 0.f; Rwc_w.at<float>(2, i) = 0.f;
            twc.at<float>(0, i) = 0.f; twc.at<float>(1, i) = 0.f; twc.at<float>(2, i) = 0.f;
        }
        i++;
    }

    w.at<float>(0, i) = gsl_vector_get(vec, j++);
    w.at<float>(1, i) = gsl_vector_get(vec, j++);
    w.at<float>(2, i) = gsl_vector_get(vec, j++);

    v.at<float>(0, i) = gsl_vector_get(vec, j++);
    v.at<float>(1, i) = gsl_vector_get(vec, j++);
    v.at<float>(2, i) = gsl_vector_get(vec, j++);

    Rwc_w.at<float>(0, i) = gsl_vector_get(vec, j++);
    Rwc_w.at<float>(1, i) = gsl_vector_get(vec, j++);
    Rwc_w.at<float>(2, i) = gsl_vector_get(vec, j++);

    twc.at<float>(0, i) = gsl_vector_get(vec, j++);
    twc.at<float>(1, i) = gsl_vector_get(vec, j++);
    twc.at<float>(2, i) = gsl_vector_get(vec, j++);

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
        status = gsl_multimin_test_size(size, 5e-2);

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
