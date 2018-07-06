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

    // Transformation
    cv::Vec3d getAngularVelocity();
    cv::Vec3d getLinearVelocity();
    void setAngularVelocity(const cv::Vec3d& w_);
    void setLinearVelocity(const cv::Vec3d& v_);
    cv::Mat getRotation();
    cv::Mat getTranslation();
public:
    // Frame and keyframe id
    unsigned mnId;
    static unsigned nNextId;
    const unsigned mnFrameId;

    // all events within keyframe; a pointer to the events in frame
    // safe implementation??
    std::set<EventMeasurement*, idxOrder>& vEvents;

    // Motion Model
    // in world coord??
    cv::Vec3d w;
    cv::Vec3d v;

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
