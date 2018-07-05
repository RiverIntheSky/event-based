#include "MapPoint.h"
using namespace std;
namespace ev {

mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint() {
    // also pose??
    setNormalDirection(0, M_PI);
}

cv::Mat MapPoint::GetNormal() {
    lock_guard<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

void MapPoint::setNormalDirection(double phi, double psi) {
    lock_guard<mutex> lock(mMutexPos);
    mNormalDirection[0] = phi;
    mNormalDirection[1] = psi;
    mNormalVector = cv::Mat(1, 3, CV_64F, {cos(phi) * sin(psi),sin(phi) * sin(psi),cos(psi)});
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
//    unique_lock<mutex> lock2(mGlobalMutex);
    lock_guard<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

void MapPoint::addObservation(shared_ptr<KeyFrame> pKF)
{
    lock_guard<mutex> lock(mMutexFeatures);
    mObservations.insert(pKF);
}

std::set<shared_ptr<KeyFrame>> MapPoint::getObservations()
{
    lock_guard<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::observations()
{
    lock_guard<mutex> lock(mMutexFeatures);
    return mObservations.size();
}

}
