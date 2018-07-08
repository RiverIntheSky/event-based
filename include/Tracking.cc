#include "Tracking.h"

namespace ev {

shared_ptr<Frame> Tracking::getCurrentFrame() {
    if (!mCurrentFrame)
        mCurrentFrame = make_shared<Frame>();
    return mCurrentFrame;
}

void Tracking::Track() {
    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    undistortEvents();
    if(mState==NOT_INITIALIZED) {
        // assume success??
        init();
    }
    //

    // add frame to map
    mpMap->addFrame(mCurrentFrame);

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
    pMP->addObservation(make_shared<KeyFrame>(*mCurrentFrame));

    // at initialization phase there is at most one element in mspMapPoints
    Optimizer::optimize(pMP.get());
    if (pMP->observations() >= nInitializer)
        mState = OK;
    return true;
}

}
