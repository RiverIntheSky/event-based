#pragma once

#include<opencv2/core/core.hpp>
#include "KeyFrame.h"

using namespace std;

namespace ev {

class KeyFrame;
class Frame;

struct idxOrder {
   template <class F>
   bool operator()(const std::shared_ptr<F>& lhs, const std::shared_ptr<F>& rhs) const {
       return lhs->mnId > rhs->mnId;
   }
};

class MapPoint{
public:
    MapPoint();
    MapPoint(const cv::Mat &Pos);
    void setWorldPos(const cv::Mat &Pos);
    cv::Mat getWorldPos();

    cv::Mat getNormal();
    // not implemented
    void setNormal();
    std::array<double, 2>& getNormalDirection();
    void setNormalDirection(double phi, double psi);

    void addObservation(shared_ptr<KeyFrame>& pKF);

    std::set<shared_ptr<KeyFrame>, idxOrder>& getObservations();
    int observations();
    void swap(bool success);
public:
    unsigned int mnId;
    // Normal vector of the plane
    std::array<double, 2> mNormalDirection;
    cv::Mat mNormalVector;

    // Synthetic image of the plane
    // access not yet thread safe !!
    cv::Mat mFront;
    cv::Mat mBack;

    // Keyframes that observe the point
    std::set<shared_ptr<KeyFrame>, idxOrder> mObservations;

    static mutex mGlobalMutex;

    // for accessing mPatch??
    mutex mMutexFeatures;

    // projection from world frame to patch frame;
    Eigen::Matrix3d Rn;

    // a point on the plane, could be center of the patch
    cv::Mat mWorldPos;

    // inverse depth to the origin
    double d;

    mutex mMutexPos;

};
}
