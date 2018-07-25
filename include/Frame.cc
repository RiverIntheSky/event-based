#include "Frame.h"

namespace ev
{
float Frame::gScale = 1.;
unsigned int Frame::nNextId = 0;

// first pose of a frame is always set at construction
Frame::Frame(Map *pMap): mpMap(pMap), shouldBeKeyFrame(false){
    mnId = nNextId++;
    cv::Mat Twc = cv::Mat::eye(4,4,CV_32F);
    setFirstPose(Twc);
    w = (cv::Mat_<float>(3, 1) << 0, 0, 0);
    v = (cv::Mat_<float>(3, 1) << 0, 0, 0);
    mScale = gScale;
    depthMap = cv::Mat(240, 180, CV_32F, std::numeric_limits<float>::max());
}

Frame::Frame(Frame &other): mpMap(other.mpMap), shouldBeKeyFrame(false){
    mnId = nNextId++;
//    setFirstPose(other.getLastPose());
//    cv::Mat n = mpMap->getAllMapPoints().front()->getNormal();
//    cv::Mat twc = getFirstPose().rowRange(0,3).col(3);
//    mScale = gScale + twc.dot(n);
//    LOG(INFO) << mScale;
    w = other.w.clone();
    v = other.v.clone();
    ev::EventMeasurement* event = new ev::EventMeasurement();
    *event = **(other.vEvents.rbegin());
    vEvents.insert(event);
    // ??
//    depthMap = cv::Mat(240, 180, CV_32F, std::numeric_limits<float>::max());
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

cv::Mat Frame::getAngularVelocity() {
    return w.clone();
}

cv::Mat Frame::getLinearVelocity() {
    return v.clone();
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

void Frame::setScale(float& scale) {
    mScale = scale;
}

float& Frame::getScale() {
    return mScale;
}

void Frame::addEvent(EventMeasurement* event) {
    vEvents.insert(event);
}

unsigned Frame::events() {
    return vEvents.size();
}

}
