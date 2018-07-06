#pragma once

#include "MapPoint.h"
#include "KeyFrame.h"

using namespace std;

namespace ev {
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
protected:
    std::set<shared_ptr<MapPoint>, idxOrder> mspMapPoints;
    std::set<shared_ptr<KeyFrame>, idxOrder> mspKeyFrames;

    // not yet sure if it's necessary to keep all frames
    std::set<shared_ptr<Frame>, idxOrder> mspFrames;
    mutex mMutexMap;
};
}
