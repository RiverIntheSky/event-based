#include "Tracking.h"
#include "Optimizer.h"

namespace ev {

shared_ptr<Frame> Tracking::getCurrentFrame() {
    if (!mCurrentFrame)
        mCurrentFrame = make_shared<Frame>(Frame());
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
    auto f(mCurrentFrame);
    mpMap->addFrame(f);

    // under certain conditions, Create KeyFrame

    // delete currentFrame
}

bool Tracking::undistortEvents() {
    std::vector<cv::Point2d> inputDistortedPoints;
    std::vector<cv::Point2d> outputUndistortedPoints;
    for (auto it = mCurrentFrame.vEvents.begin(); it != mCurrentFrame.vEvents.end(); it++) {
        cv::Point2d point(it->measurement.x, it->measurement.y);
        inputDistortedPoints.push_back(point);
    }
    cv::undistortPoints(inputDistortedPoints, outputUndistortedPoints,
                        mK, mDistCoeffs);
    auto it = mCurrentFrame.vEvents.begin();
    auto p_it = outputUndistortedPoints.begin();
    for (; it != mCurrentFrame.vEvents.end(); it++, p_it++) {
        it->measurement.x = p_it->x;
        it->measurement.y = p_it->y;
    }
    return true;
}

bool Tracking::init() {
    if (!mpMap->mapPointsInMap()) {
        MapPoint pMP;
        KeyFrame pKF(mCurrentFrame);
        pMP.addObservation(make_shared<KeyFrame>(pKF));
        mpMap->addMapPoint(make_shared<MapPoint>(pMP));
    } else {
        auto pMP = mpMap->getAllMapPoints().front();
        KeyFrame pKF(mCurrentFrame);
        pMP->addObservation(make_shared<KeyFrame>(pKF));
    }
    // at initialization phase there is at most one element in mspMapPoints
    auto pMP = mpMap->getAllKeyFrames().front();
    Optimizer::optimize(pMP);

}

}
