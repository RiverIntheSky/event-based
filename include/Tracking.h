#pragma once
#include "Map.h"

namespace ev {
class Tracking
{
public:
    Tracking();

    Tracking(cv::Mat& K, cv::Mat& DistCoeffs)
        : mK(K), mDistCoeffs(DistCoeffs), mState(NOT_INITIALIZED), nInitializer(5), nMapper(3) {}

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
    cv::Mat mDistCoeffs;
    Map* mpMap;
    unsigned int nInitializer;
    unsigned int nMapper;
};
}
