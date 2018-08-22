#pragma once

#include "event.h"
#include "Map.h"
#include <glog/logging.h>

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

    // initialize motion model from last frame
    Frame(Frame& other);

    void addEvent(EventMeasurement* event);

    // number of events
    unsigned events();

    // Pose functions
    void setFirstPose(const cv::Mat &Twc);
    void setLastPose(const cv::Mat &Twc);
    cv::Mat getFirstPose();
    cv::Mat getLastPose();
    cv::Mat getAngularVelocity();
    cv::Mat getLinearVelocity();
    void setAngularVelocity(const cv::Mat& w_);
    void setLinearVelocity(const cv::Mat& v_);
    cv::Mat getRotation();
    cv::Mat getTranslation();
    void setScale(double& scale);
    double& getScale();

public:
    // Frame id
    unsigned int mnId;
    static unsigned int nNextId;

    // all events within keyframe
    std::set<EventMeasurement*, timeOrder> vEvents;

    // Motion Model
    cv::Mat w;
    cv::Mat v;

    // scale
    double mScale; /* distance from current frame to the map */
    static double gScale; /* distance from the first frame to the map */

    // camera pose of first event
    cv::Mat mTwc1;
    // camera pose of last event
    cv::Mat mTwc2;

    // orientation and translation
    cv::Mat mRwc;
    cv::Mat mtwc;

    // timespan of the events
    double dt;

    // timestamp of first event in frame
    okvis::Time mTimeStamp;

    // whether should be added to keyframe
    bool shouldBeKeyFrame;

    // pointer to the map
    Map* mpMap;
};
}
