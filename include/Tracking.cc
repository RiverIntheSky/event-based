#include "Tracking.h"

namespace ev {

int Tracking::nInitializer = 4;
int Tracking::nMapper = 3;
cv::Mat Tracking::w = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat Tracking::v = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat Tracking::r = cv::Mat::zeros(3, 1, CV_64F);
cv::Mat Tracking::t = cv::Mat::zeros(3, 1, CV_64F);

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
        estimate();
    }
    //

    // add frame to map
    mpMap->addFrame(mCurrentFrame);
//    LOG(INFO) << "current velocity model:";
//    LOG(INFO) << "\nw\n" << mCurrentFrame->w;
//    LOG(INFO) << "\nv\n" << mCurrentFrame->v;
//    LOG(INFO);
    w = mCurrentFrame->w;
    v = mCurrentFrame->v;
    cv::Mat R = mCurrentFrame->getRotation();
    r = rotm2axang(R);
    t = mCurrentFrame->getTranslation();
    mCurrentFrame = make_shared<Frame>(*mCurrentFrame);

    // under certain conditions, Create KeyFrame

    // delete currentFrame
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

bool Tracking::init() {
    if (!mpMap->mapPointsInMap()) {
        mpMap->addMapPoint(make_shared<MapPoint>());
    }
    auto pMP = mpMap->getAllMapPoints().front();
    auto pKF = make_shared<KeyFrame>(*mCurrentFrame);
    pMP->addObservation(pKF);
    mpMap->addKeyFrame(pKF);

    // at initialization phase there is at most one element in mspMapPoints
    Optimizer::optimize(pMP.get());

    // set pose of current frame to be the same as current keyframe??
    mCurrentFrame->setAngularVelocity(pKF->getAngularVelocity());
    mCurrentFrame->setLinearVelocity(pKF->getLinearVelocity());
    mCurrentFrame->setFirstPose(pKF->getFirstPose());
    mCurrentFrame->setLastPose(pKF->getLastPose());
    mCurrentFrame->setScale(pKF->getScale());

    if (pMP->observations() >= nInitializer) {
        mState = OK;
        pMP->swap(true);
    }
    return true;
}

bool Tracking::estimate() {
    // WIP
    auto pMP = mpMap->getAllMapPoints().front();
    Optimizer::optimize(pMP.get(), mCurrentFrame.get());
    if (mCurrentFrame->shouldBeKeyFrame) {
        shared_ptr<KeyFrame> pKF = make_shared<KeyFrame>(*mCurrentFrame);
        if(insertKeyFrame(pKF)) {
            mCurrentFrame->setAngularVelocity(pKF->getAngularVelocity());
            mCurrentFrame->setLinearVelocity(pKF->getLinearVelocity());
            mCurrentFrame->setFirstPose(pKF->getFirstPose());
            mCurrentFrame->setLastPose(pKF->getLastPose());

            pMP->addObservation(pKF);
            pMP->swap(true);
            mpMap->addKeyFrame(pKF);
        }
    }
    return true;
}

bool Tracking::insertKeyFrame(shared_ptr<KeyFrame>& pKF) {
    auto pMP = mpMap->getAllMapPoints().front();

    if (Optimizer::optimize(pMP.get(), pKF)) {
        return true;
    } else {
        KeyFrame::nNextId--;
        return false;
    }
}

}
