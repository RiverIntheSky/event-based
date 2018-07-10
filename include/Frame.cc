#include "Frame.h"

namespace ev
{

unsigned int Frame::nNextId = 0;

// first pose of a frame is always set at construction
Frame::Frame(Map *pMap): mpMap(pMap) {
    mnId = nNextId++;
    cv::Mat Twc = cv::Mat::eye(4,4,CV_64F);
    setFirstPose(Twc);
    w = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    v = (cv::Mat_<double>(3, 1) << 0, 0, 0);
}

Frame::Frame(Frame &other): mpMap(other.mpMap){
    mnId = nNextId++;
    setFirstPose(other.getLastPose());
    w = other.w.clone();
    cv::Mat n = mpMap->getAllMapPoints().front()->getNormal();
    v = other.v / (1 + (other.getRotation() * other.getTranslation()).dot(n));
//    LOG(INFO) << other.v;
//    LOG(INFO) << (1 + (other.getRotation() * other.getTranslation()).dot(n));
//    LOG(INFO) << v;
    ev::EventMeasurement* event = new ev::EventMeasurement();
    *event = **(other.vEvents.rbegin());
    vEvents.insert(event);
}

cv::Mat Frame::getFirstPose()
{
    return mTwc1.clone();
}

cv::Mat Frame::getLastPose()
{
    return mTwc2.clone();
}

void Frame::setFirstPose(const cv::Mat &Twc_)
{
    Twc_.copyTo(mTwc1);
//    mRwc = mTwc1.rowRange(0,3).colRange(0,3);
//    mtwc = mTwc1.rowRange(0,3).col(3);
}

void Frame::setLastPose(const cv::Mat &Twc_)
{
    Twc_.copyTo(mTwc2);
//    mRwc = mTwc2.rowRange(0,3).colRange(0,3);
//    mtwc = mTwc2.rowRange(0,3).col(3);
}

void Frame::setAngularVelocity(const cv::Mat& w_) {
    w_.copyTo(w);
}
void Frame::setLinearVelocity(const cv::Mat& v_) {
    v_.copyTo(v);
}

cv::Mat Frame::getRotation() {
    return mTwc1.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat Frame::getTranslation() {
    return mTwc1.rowRange(0,3).col(3).clone();
}

void Frame::addEvent(EventMeasurement* event) {
    // first event ??
    if (vEvents.empty())
        mTimeStamp = event->timeStamp;
    vEvents.insert(event);
}

unsigned Frame::events() {
    return vEvents.size();
}

}
