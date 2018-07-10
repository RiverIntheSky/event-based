#pragma once

#include "event.h"
#include "Map.h"

namespace ev {
class Map;
struct timeOrder {
    bool operator()(const EventMeasurement* lhs, const EventMeasurement* rhs) const {
        return lhs->timeStamp < rhs->timeStamp;
    }
};

class Frame
{
public:
    Frame(Map *pMap);

    // initinize motion model from last frame
    Frame(Frame& other);

    void addEvent(EventMeasurement* event);

    // number of events
    unsigned events();

    // Pose functions
    void setFirstPose(const cv::Mat &Twc);
    void setLastPose(const cv::Mat &Twc);
    cv::Mat getFirstPose();
    cv::Mat getLastPose();
    void setAngularVelocity(const cv::Mat& w_);
    void setLinearVelocity(const cv::Mat& v_);
    cv::Mat getRotation();
    cv::Mat getTranslation();

public:
    // Frame id
    unsigned int mnId;
    static unsigned int nNextId;

    // all events within keyframe

    std::set<EventMeasurement*, timeOrder> vEvents;

    // Motion Model
    // in local coord??
    cv::Mat w;
    cv::Mat v;

    // camera pose of first event
    cv::Mat mTwc1;
    // camera pose of last event
    cv::Mat mTwc2;

    // rotation and translation
    cv::Mat mRwc;
    cv::Mat mtwc;

    // timespan of the events
    double dt;

    // timestamp of first event in frame
    okvis::Time mTimeStamp;

    Map* mpMap;
};
}
