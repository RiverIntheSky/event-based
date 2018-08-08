#pragma once

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <atomic>

using namespace std;

namespace ev {

class Frame;
class KeyFrame;
class MapPoint;
class MapDrawer;
struct idxOrder;

class Map{

public:
    Map();

    void addFrame(shared_ptr<Frame> pF);
    void addKeyFrame(shared_ptr<KeyFrame> pKF);
    void addMapPoint(shared_ptr<MapPoint> pMP);

    std::vector<shared_ptr<KeyFrame>> getAllKeyFrames();
    std::vector<shared_ptr<MapPoint>> getAllMapPoints();
    unsigned mapPointsInMap();
    unsigned keyFramesInMap();

    mutex mMutexMapUpdate;
    std::atomic<bool> isDirty;
    MapDrawer* drawer;
    std::set<shared_ptr<MapPoint>> mspMapPoints;

    // forward declaration doesn't work?
    struct idxOrder {
       template <class F>
       bool operator()(const std::shared_ptr<F>& lhs, const std::shared_ptr<F>& rhs) const {
           return lhs->mnId < rhs->mnId;
       }
    };

    std::set<shared_ptr<KeyFrame>, idxOrder> mspKeyFrames;
protected:
    // not yet sure if it's necessary to keep all frames
    std::set<shared_ptr<Frame>, idxOrder> mspFrames;
    mutex mMutexMap;
};
}
