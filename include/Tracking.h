#pragma once
#include "Map.h"
#include "Optimizer.h"
#include "util/utils.h"

namespace ev {
class Tracking
{
public:
    Tracking();

    Tracking(cv::Mat& K, cv::Vec<double, 5>& DistCoeffs, Map* mpMap_)
        : mState(NOT_INITIALIZED), mK(K), mDistCoeffs(DistCoeffs), mpMap(mpMap_){}

    shared_ptr<Frame> getCurrentFrame();

    void Track();

    // Debugging purpose; use this function to initialize the optimization near ground truth
    void Track(cv::Mat R, cv::Mat t, cv::Mat w, cv::Mat v);

    bool init();
    bool estimate();
    bool relocalize(cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v);
    bool insertKeyFrame(shared_ptr<KeyFrame>& pKF);

    // Correct the distorted events of the currently tracked frame
    bool undistortEvents();

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
    static int nInitializer;
    static int nMapper;
    static cv::Mat w;
    static cv::Mat v;
    static cv::Mat R;
    static cv::Mat r;
    static cv::Mat t;
protected:
    shared_ptr<Frame> mCurrentFrame;
    cv::Mat mK;
    cv::Vec<double, 5> mDistCoeffs;
    Map* mpMap;

};
}
