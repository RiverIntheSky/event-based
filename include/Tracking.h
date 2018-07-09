#pragma once
#include "Map.h"
#include "Optimizer.h"

namespace ev {
class Tracking
{
public:
    Tracking();

    Tracking(cv::Mat& K, cv::Vec<double, 5>& DistCoeffs, Map* mpMap_)
        : mState(NOT_INITIALIZED), mK(K), mDistCoeffs(DistCoeffs), mpMap(mpMap_), nInitializer(5), nMapper(3) {}

    shared_ptr<Frame> getCurrentFrame();
    void Track();
    bool undistortEvents();
    bool init();

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
protected:
    shared_ptr<Frame> mCurrentFrame;
    cv::Mat mK;
    cv::Vec<double, 5> mDistCoeffs;
    Map* mpMap;
    int nInitializer;
    int nMapper;
};
}