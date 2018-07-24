#include "KeyFrame.h"
using namespace std;

namespace ev
{

unsigned KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame& F):
    // keyframe has the same motion model from the derived frame
    mnFrameId(F.mnId), w(F.w), v(F.v), mTimeStamp(F.mTimeStamp), dt(F.dt) {
    mnId = nNextId++;
    // same pose??
    setFirstPose(F.getFirstPose());
    setLastPose(F.getLastPose());
    vEvents = &(F.vEvents);
    mScale = F.mScale;
}

cv::Mat KeyFrame::getFirstPose()
{
    lock_guard<mutex> lock(mMutexPose);
    return mTwc1.clone();
}

cv::Mat KeyFrame::getLastPose()
{
    lock_guard<mutex> lock(mMutexPose);
    return mTwc2.clone();
}

void KeyFrame::setFirstPose(const cv::Mat &Twc_)
{
    lock_guard<mutex> lock(mMutexPose);
    Twc_.copyTo(mTwc1);
//    mRwc = mTwc1.rowRange(0,3).colRange(0,3);
//    mtwc = mTwc1.rowRange(0,3).col(3);
}

void KeyFrame::setLastPose(const cv::Mat &Twc_)
{
    lock_guard<mutex> lock(mMutexPose);
    Twc_.copyTo(mTwc2);
//    mRwc = mTwc2.rowRange(0,3).colRange(0,3);
//    mtwc = mTwc2.rowRange(0,3).col(3);
}

cv::Mat KeyFrame::getAngularVelocity() {
    lock_guard<mutex> lock(mMutexPose);
    return w.clone();
}

cv::Mat KeyFrame::getLinearVelocity() {
    lock_guard<mutex> lock(mMutexPose);
    return v.clone();
}

void KeyFrame::setAngularVelocity(const cv::Mat& w_) {
    lock_guard<mutex> lock(mMutexPose);
    w_.copyTo(w);
}
void KeyFrame::setLinearVelocity(const cv::Mat& v_) {
    lock_guard<mutex> lock(mMutexPose);
    v_.copyTo(v);
}

cv::Mat KeyFrame::getRotation() {
    lock_guard<mutex> lock(mMutexPose);
    return mTwc1.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::getTranslation() {
    lock_guard<mutex> lock(mMutexPose);
    return mTwc1.rowRange(0,3).col(3).clone();
}

void KeyFrame::setScale(float& scale) {
    lock_guard<mutex> lock(mMutexPose);
    mScale = scale;
}

float& KeyFrame::getScale() {
    lock_guard<mutex> lock(mMutexPose);
    return mScale;
}

}
