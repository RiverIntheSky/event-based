#pragma once

#include <vector>
#include <mutex>
#include <memory>

#include "event.h"
#include "Frame.h"
#include <opencv2/opencv.hpp>

namespace ev {
class KeyFrame
{
public:
    KeyFrame();
    KeyFrame(Frame& F);

    // Pose functions
    void setPose(const cv::Mat &Twc);
    cv::Mat getPose();
    cv::Mat getRotation();
    cv::Mat getTranslation();
public:
    // Frame and keyframe id
    unsigned mnId;
    static unsigned nNextId;
    const unsigned mnFrameId;

    // all events within keyframe; a pointer to the events in frame
    // safe implementation??
    std::set<EventMeasurement*>* vEvents;

    //Motion Model
    std::array<float,4> w;
    std::array<float,4> v;

    // camera pose
    cv::Mat mTwc;

    // rotation and translation
    cv::Mat mRwc;
    cv::Mat mtwc;

    // timestamp of first event in frame
    okvis::Time mTimeStamp;

    // mutex
    std::mutex mMutexPose;
};
}
