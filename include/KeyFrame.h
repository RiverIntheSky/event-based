#pragma once

#include <vector>
#include <mutex>
#include <memory>

#include "event.h"
#include "Frame.h"
#include <opencv2/opencv.hpp>

namespace ev {

class Frame;
struct timeOrder;

class KeyFrame      
{
public:
    KeyFrame(Frame& F);

    // Pose functions
    void setFirstPose(const cv::Mat &Twc);
    void setLastPose(const cv::Mat &Twc);
    cv::Mat getFirstPose();
    cv::Mat getLastPose();

    // Transformation
    cv::Mat getAngularVelocity();
    cv::Mat getLinearVelocity();
    void setAngularVelocity(const cv::Mat& w_);
    void setLinearVelocity(const cv::Mat& v_);
    cv::Mat getRotation();
    cv::Mat getTranslation();
    void setScale(float& scale);
    float& getScale();
public:
    // Frame and keyframe id
    unsigned mnId;
    static unsigned nNextId;
    const unsigned mnFrameId;

    // all events within keyframe; a pointer to the events in frame
    // safe implementation??
    std::set<EventMeasurement*, timeOrder>* vEvents;

    // Motion Model
    // in world coord??
    cv::Mat w;
    cv::Mat v;

    // scale
    float mScale;

    // camera pose of first event
    cv::Mat mTwc1;
    // camera pose of last event
    cv::Mat mTwc2;

    // rotation and translation
    cv::Mat mRwc;
    cv::Mat mtwc;

    // timestamp of first event in frame
    okvis::Time mTimeStamp;

    // timespan of the events
    float dt;

    // shader support
    GLuint vao, vbo;

    // mutex
    std::mutex mMutexPose;
};
}
