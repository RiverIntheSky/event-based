#pragma once

#include "event.h"
#include "KeyFrame.h"
#include "MapPoint.h"

namespace ev {

struct idxOrder {
    // ?? maybe useful
    bool operator()(const EventMeasurement* lhs, const EventMeasurement* rhs) const {
        return lhs->timeStamp < rhs->timeStamp;
    }
    bool operator()(const std::shared_ptr<KeyFrame>& lhs, const std::shared_ptr<KeyFrame>& rhs) const {
        return lhs->mnId < rhs->mnId;
    }
    bool operator()(const std::shared_ptr<Frame>& lhs, const std::shared_ptr<Frame>& rhs) const {
        return lhs->mnId < rhs->mnId;
    }
    bool operator()(const std::shared_ptr<MapPoint>& lhs, const std::shared_ptr<MapPoint>& rhs) const {
        return lhs->mnId < rhs->mnId;
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
    void setPose(cv::Mat Twc);

public:
    // Frame id
    unsigned int mnId;
    static unsigned int nNextId;

    // all events within keyframe
    std::set<EventMeasurement*, idxOrder> vEvents;

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

    // current events number in the frame
    unsigned int nbEvents;
};
}
