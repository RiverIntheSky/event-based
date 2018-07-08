#pragma once

#include "event.h"

namespace ev {

struct timeOrder {
    bool operator()(const EventMeasurement* lhs, const EventMeasurement* rhs) const {
        return lhs->timeStamp < rhs->timeStamp;
    }
};

class Frame
{
public:
    Frame();

    // initinize motion model from last frame
    Frame(Frame& other);

    void addEvent(EventMeasurement* event);

    // number of events
    unsigned events();

    // Set the camera pose.
    void setPose(const cv::Mat& Twc);

public:
    // Frame id
    unsigned int mnId;
    static unsigned int nNextId;

    // all events within keyframe

    std::set<EventMeasurement*, timeOrder> vEvents;

    // Motion Model
    // in world coord??
    cv::Mat w;
    cv::Mat v;

    // camera pose
    cv::Mat mTwc;

    // rotation and translation
    cv::Mat mRwc;
    cv::Mat mtwc;

    // timestamp of first event in frame
    okvis::Time mTimeStamp;
};
}
