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
    /**
     *  \brief
     *  A MapPoint is a plane with normal direction, position and texture
     *  In planar scene only one MapPoint is present
     */
public:
    MapPoint();
    void setWorldPos(const cv::Mat &Pos);
    cv::Mat getWorldPos();

    cv::Mat getNormal();
    // not implemented
    void setNormal();
    std::array<double, 2>& getNormalDirection();
    void setNormalDirection(double phi, double psi);

    void addObservation(shared_ptr<KeyFrame>& pKF);

    std::set<shared_ptr<KeyFrame>, idxOrder>& getObservations();

    // number of observations
    int observations();

    // if the optimization is successful, we copy the image from back buffer to front buffer;
    // otherwise, we copy the front buffer to the back buffer and restart the optimization
    // process
    void swap(bool success);
public:
    unsigned int mnId;
    // Normal vector of the plane
    std::array<double, 2> mNormalDirection;
    cv::Mat mNormalVector;

    // buffers to store the texture;
    // mBack is the back buffer, will be constanly changed to compute the cost of different parameters;
    // mFront is the optimized texture
    cv::Mat mFront;
    cv::Mat mBack;

    // Keyframes that observe the point
    std::set<shared_ptr<KeyFrame>, idxOrder> mObservations;

    static mutex mGlobalMutex;

    // for accessing mPatch
    mutex mMutexFeatures;

    // projection from world frame to patch frame;
    Eigen::Matrix3d Rn;

protected:
    // center of the patch
    // not used por planar_slam
    cv::Mat mWorldPos;

    mutex mMutexPos;

};
}
