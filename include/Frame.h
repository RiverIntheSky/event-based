#pragma once

#include "event.h"

namespace ev {
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
    void setPose(cv::Mat Twc);

public:
    // Frame id
    unsigned int mnId;
    static unsigned int nNextId;

    // all events within keyframe
    std::set<EventMeasurement*> vEvents;

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

    // current events number in the frame
    unsigned int nbEvents;
};
}
