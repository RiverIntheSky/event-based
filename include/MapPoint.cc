#include "MapPoint.h"
#include "util/utils.h"

using namespace std;
namespace ev {

mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint() {
    // also pose??
    d = 1.;
    setNormalDirection(0, M_PI);  // normal (0, 0, -1)
                                  // psi \in (M_PI/2, 3*M_PI/2)
                                  // phi \in (0, 2 * M_PI)
}

MapPoint::MapPoint(const cv::Mat &Pos) {
    Pos.copyTo(mFramePos);
    setWorldPos(mFramePos);
    d = 1.;
    dirty = false;
    setNormalDirection(0, M_PI);
}

cv::Mat MapPoint::getNormal() {
    lock_guard<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

void MapPoint::setNormalDirection(float phi, float psi) {
    lock_guard<mutex> lock(mMutexPos);
    mNormalDirection[0] = phi;
    mNormalDirection[1] = psi;
    float data[] = {cos(phi) * sin(psi), sin(phi) * sin(psi), cos(psi)};
    cv::Mat(3, 1, CV_32F, data).copyTo(mNormalVector);
    Eigen::Vector3d z;
    z << 0, 0, 1;
    Eigen::Vector3d nw;
    nw << cos(phi) * sin(psi), sin(phi) * sin(psi), cos(psi);
    Eigen::Vector3d v = (-nw).cross(z);
    float c = -z.dot(nw);
    Eigen::Matrix3d Kn = ev::skew(v);
    Rn = Eigen::Matrix3d::Identity() + Kn + Kn * Kn / (1 + c);
}

std::array<float, 2>& MapPoint::getNormalDirection() {
    lock_guard<mutex> lock(mMutexPos);
    return mNormalDirection;
}

void MapPoint::setWorldPos(const cv::Mat &Pos) {
//    unique_lock<mutex> lock2(mGlobalMutex);
    lock_guard<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::getFramePos(){
    lock_guard<mutex> lock(mMutexPos);
    return mFramePos.clone();
}

cv::Mat MapPoint::getWorldPos(){
    lock_guard<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

void MapPoint::addObservation(shared_ptr<KeyFrame>& pKF) {
    lock_guard<mutex> lock(mGlobalMutex);
    mObservations.insert(pKF);
}

std::set<shared_ptr<KeyFrame>, idxOrder>& MapPoint::getObservations() {
    lock_guard<mutex> lock(mGlobalMutex);
    return mObservations;
}

int MapPoint::observations() {
    lock_guard<mutex> lock(mGlobalMutex);
    return mObservations.size();
}

void MapPoint::swap(bool success) {
    if (success) {
        mFront = -abs(mBack);
//        cv::swap(mFront, mBack);
    } else {
        mFront.copyTo(mBack);
    }
}

}
