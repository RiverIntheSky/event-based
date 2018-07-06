#include "KeyFrame.h"
using namespace std;

namespace ev
{

unsigned KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame& F):
    // keyframe has the same motion model from the derived frame
    mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), w(F.w), v(F.v) {
    mnId = nNextId++;
    // same pose??
    setPose(F.mTwc);
    vEvents = F.vEvents;
}

cv::Mat KeyFrame::getPose()
{
    lock_guard<mutex> lock(mMutexPose);
    return mTwc.clone();
}
void KeyFrame::setPose(const cv::Mat &Twc_)
{
    lock_guard<mutex> lock(mMutexPose);
    Twc_.copyTo(mTwc);
    mRwc = mTwc.rowRange(0,3).colRange(0,3);
    mtwc = mTwc.rowRange(0,3).col(3);
}

cv::Mat KeyFrame::getAngularVelocity() {
    lock_guard<mutex> lock(mMutexPose);
    return w.clone();
}

cv::Vec3d KeyFrame::getLinearVelocity() {
    lock_guard<mutex> lock(mMutexPose);
    return v.clone();
}

void KeyFrame::setAngularVelocity(const cv::Vec3d& w_) {
    lock_guard<mutex> lock(mMutexPose);
    w_.copyTo(w);
}
void KeyFrame::setLinearVelocity(const cv::Vec3d& v_) {
    lock_guard<mutex> lock(mMutexPose);
    v_.copyTo(v);
}

cv::Mat KeyFrame::getRotation() {
    lock_guard<mutex> lock(mMutexPose);
    return mTwc.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::getTranslation() {
    lock_guard<mutex> lock(mMutexPose);
    return mTwc.rowRange(0,3).col(3).clone();
}

}
