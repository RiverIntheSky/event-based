#include "Map.h"

using namespace std;
namespace ev {

Map::Map():isDirty(false) {}

void Map::addFrame(shared_ptr<Frame> pF)
{
    lock_guard<mutex> lock(mMutexMap);
    mspFrames.insert(pF);
}

void Map::addKeyFrame(shared_ptr<KeyFrame> pKF)
{
    lock_guard<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
}

void Map::addMapPoint(shared_ptr<MapPoint> pMP)
{
    lock_guard<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

vector<shared_ptr<KeyFrame>> Map::getAllKeyFrames()
{
    lock_guard<mutex> lock(mMutexMap);
    return vector<shared_ptr<KeyFrame>>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<shared_ptr<MapPoint>> Map::getAllMapPoints()
{
    lock_guard<mutex> lock(mMutexMap);
    return vector<shared_ptr<MapPoint>>(mspMapPoints.begin(),mspMapPoints.end());
}

unsigned Map::mapPointsInMap()
{
    lock_guard<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

unsigned Map::keyFramesInMap()
{
    lock_guard<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

}
