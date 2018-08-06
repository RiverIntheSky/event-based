#include "Tracking.h"

namespace ev {

int Tracking::nInitializer = 1;
int Tracking::nMapper = 3;
cv::Mat Tracking::w = cv::Mat::zeros(3, 1, CV_32F);
cv::Mat Tracking::v = cv::Mat::zeros(3, 1, CV_32F);
cv::Mat Tracking::R = cv::Mat::eye(3, 3, CV_32F);
cv::Mat Tracking::r = cv::Mat::zeros(3, 1, CV_32F);
cv::Mat Tracking::t = cv::Mat::zeros(3, 1, CV_32F);
cv::Mat Tracking::nw = cv::Mat::zeros(3, 1, CV_32F);
float Tracking::phi = 0;
float Tracking::psi = M_PI/2;
float Tracking::phi_w = 0;
float Tracking::psi_w = M_PI;

shared_ptr<Frame> Tracking::getCurrentFrame() {
    if (!mCurrentFrame)
        mCurrentFrame = make_shared<Frame>(mpMap);
    return mCurrentFrame;
}

void Tracking::Track() {
    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    undistortEvents();
    mCurrentFrame->mTimeStamp = (*(mCurrentFrame->vEvents.cbegin()))->timeStamp;
    mCurrentFrame->dt = ((*(mCurrentFrame->vEvents.crbegin()))->timeStamp - mCurrentFrame->mTimeStamp).toSec();

    if(mState==NOT_INITIALIZED) {
        // assume success??
        init();
    } else if(mState==OK) {
        init();//estimate();
    }

    // add frame to map
    mpMap->addFrame(mCurrentFrame);
    LOG(INFO) << "keyframes: " << mpMap->keyFramesInMap();
    LOG(INFO) << "current velocity model:";
    LOG(INFO) << "\nT\n" << mCurrentFrame->getFirstPose();
    LOG(INFO) << "\nw\n" << mCurrentFrame->w;
    LOG(INFO) << "\nv\n" << mCurrentFrame->v;
    LOG(INFO);
    (*(mCurrentFrame->mvpMapPoints.begin()))->getNormal().copyTo(nw);
    mCurrentFrame->w.copyTo(w);
    mCurrentFrame->v.copyTo(v);
    mCurrentFrame->getRotation().copyTo(R);
    r = rotm2axang(R);
    mCurrentFrame->getTranslation().copyTo(t);
    mCurrentFrame = make_shared<Frame>(*mCurrentFrame);


    // under certain conditions, Create KeyFrame

    // delete currentFrame
}

void Tracking::Track(cv::Mat R_, cv::Mat t_, cv::Mat w_, cv::Mat v_) {
    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    undistortEvents();
    mCurrentFrame->mTimeStamp = (*(mCurrentFrame->vEvents.cbegin()))->timeStamp;
    mCurrentFrame->dt = ((*(mCurrentFrame->vEvents.crbegin()))->timeStamp - mCurrentFrame->mTimeStamp).toSec();
    cv::Mat Twc1 = cv::Mat::eye(4,4,CV_32F);
    R_.copyTo(Twc1.rowRange(0,3).colRange(0,3));
    t_.copyTo(Twc1.rowRange(0,3).col(3));
    mCurrentFrame->setFirstPose(Twc1);
//    mCurrentFrame->setAngularVelocity(w_);
//    mCurrentFrame->setLinearVelocity(v_);

    LOG(INFO) << "groundtruth velocity model:";
    LOG(INFO) << "\nT\n" << mCurrentFrame->getFirstPose();
    LOG(INFO) << "\nw\n" << mCurrentFrame->w;
    LOG(INFO) << "\nv\n" << mCurrentFrame->v;
    LOG(INFO);

    if(mState==NOT_INITIALIZED) {
        // assume success??
        init();
    } else if(mState==OK) {/*estimate();*/
        estimate();
    }
    if (mState == LOST) {
        cv::Mat R_ = mCurrentFrame->getRotation();
        cv::Mat t_ = mCurrentFrame->getTranslation();
        if (!relocalize(R_, t_, mCurrentFrame->w, mCurrentFrame->v)) {
            relocalize(R, t, w, v);
            // need better solution;
            mState = OK;
        }
    }

    // add frame to map
    mpMap->addFrame(mCurrentFrame);

    LOG(INFO) << "current velocity model:";
    LOG(INFO) << "\nT\n" << mCurrentFrame->getFirstPose();
    LOG(INFO) << "\nw\n" << mCurrentFrame->getAngularVelocity();
    LOG(INFO) << "\nv\n" << mCurrentFrame->getLinearVelocity();
    LOG(INFO);
    mCurrentFrame->w.copyTo(w);
    mCurrentFrame->v.copyTo(v);
    mCurrentFrame->getRotation().copyTo(R);
    r = rotm2axang(R);
    mCurrentFrame->getTranslation().copyTo(t);
    LOG(INFO)<<"------------------";
    mCurrentFrame = make_shared<Frame>(*mCurrentFrame);
    LOG(INFO)<<"------------------";
    // under certain conditions, Create KeyFrame

    // delete currentFrame
}

bool Tracking::init() {
    Optimizer::optimize(mCurrentFrame.get());

//    if (!mpMap->mapPointsInMap()) {
//        mpMap->addMapPoint(make_shared<MapPoint>());
//    }
    auto pMPs = mpMap->getAllMapPoints();
    auto pKF = make_shared<KeyFrame>(*mCurrentFrame);
    mpMap->addKeyFrame(pKF);
    for (auto pMP: pMPs) {
        pMP->addObservation(pKF);
        cv::Mat pos = pMP->getWorldPos();
        pos = pos /(-pMP->d * pos.dot(pMP->getNormal())); /* n'x + d_ = 0 */
        pMP->setWorldPos(pos);
    }


//    // at initialization phase there is at most one element in mspMapPoints
//    Optimizer::optimize(pMP.get());

//    // set pose of current frame to be the same as current keyframe??
//    mCurrentFrame->setAngularVelocity(pKF->getAngularVelocity());
//    mCurrentFrame->setLinearVelocity(pKF->getLinearVelocity());
//    mCurrentFrame->setFirstPose(pKF->getFirstPose());
//    mCurrentFrame->setLastPose(pKF->getLastPose());
//    mCurrentFrame->setScale(pKF->getScale());

//    if (pMP->observations() >= nInitializer) {
        mState = OK;
//        pMP->swap(true);
//    }
    return true;
}

bool Tracking::estimate() {
    // WIP
    newFrame = true;
    mpMap->isDirty = true;
    while (newFrame) {std::this_thread::yield();}

//    auto pMP = mpMap->getAllMapPoints().front();
//    Optimizer::optimize(pMP.get(), mCurrentFrame.get());
//    if (mCurrentFrame->shouldBeKeyFrame) {
//        shared_ptr<KeyFrame> pKF = make_shared<KeyFrame>(*mCurrentFrame);
//        if(insertKeyFrame(pKF)) {
//            mCurrentFrame->setAngularVelocity(pKF->getAngularVelocity());
//            mCurrentFrame->setLinearVelocity(pKF->getLinearVelocity());
//            mCurrentFrame->setFirstPose(pKF->getFirstPose());
//            mCurrentFrame->setLastPose(pKF->getLastPose());

//            pMP->addObservation(pKF);
//            pMP->swap(true);
//            mpMap->addKeyFrame(pKF);
//            LOG(INFO) << "keyframe id " << pKF->mnFrameId;
//        }
//    }
    return true;
}

bool Tracking::relocalize(cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v) {
    auto pMP = mpMap->getAllMapPoints().front();
    if(Optimizer::optimize(pMP.get(), mCurrentFrame.get(), Rwc, twc, w, v)) {
        mState = OK;
        return true;
    }
    return false;
}

bool Tracking::insertKeyFrame(shared_ptr<KeyFrame>& pKF) {
    auto pMP = mpMap->getAllMapPoints().front();

    if (Optimizer::optimize(pMP.get(), pKF)) {
        return true;
    } else {
        mState = LOST;
        LOG(ERROR) << "LOST";
        KeyFrame::nNextId--;
        return false;
    }
}

bool Tracking::undistortEvents() {
    std::vector<cv::Point2d> inputDistortedPoints;
    std::vector<cv::Point2d> outputUndistortedPoints;
    for (auto it: mCurrentFrame->vEvents) {
        cv::Point2d point(it->measurement.x, it->measurement.y);
        inputDistortedPoints.push_back(point);
    }
    cv::undistortPoints(inputDistortedPoints, outputUndistortedPoints,
                        mK, mDistCoeffs);
    auto it = mCurrentFrame->vEvents.begin();
    auto p_it = outputUndistortedPoints.begin();
    for (; it != mCurrentFrame->vEvents.end(); it++, p_it++) {
        (*it)->measurement.x = p_it->x;
        (*it)->measurement.y = p_it->y;
    }
    return true;
}

}
