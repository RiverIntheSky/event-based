#pragma once
#include "Map.h"
#include "Optimizer.h"
#include "util/utils.h"
#include <atomic>

namespace ev {
class Map;
class Frame;
class KeyFrame;

class Tracking
{
public:
    Tracking();

    Tracking(cv::Mat& K, cv::Vec<float, 5>& DistCoeffs, Map* mpMap_)
        : mState(NOT_INITIALIZED), mpMap(mpMap_), newFrame(false), mK(K), mDistCoeffs(DistCoeffs) {}

    std::shared_ptr<Frame> getCurrentFrame();
    void Track();
    void Track(cv::Mat R, cv::Mat t, cv::Mat w, cv::Mat v);
    bool init();
    bool estimate();
    bool relocalize(cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v);
    bool insertKeyFrame(std::shared_ptr<KeyFrame>& pKF);
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
    Map* mpMap;
    std::shared_ptr<Frame> mCurrentFrame;
    std::atomic<bool> newFrame;
protected:
    cv::Mat mK;
    cv::Vec<float, 5> mDistCoeffs;

};
}
