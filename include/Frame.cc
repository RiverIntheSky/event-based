#include "Frame.h"

namespace ev
{
unsigned int Frame::nNextId = 0;

Frame::Frame() {
    mnId = nNextId++;
    cv::Mat Twc = cv::Mat::eye(4,4,CV_64F);
    setPose(Twc);
    w = {0, 0, 0};
    v = {0, 0, 0};
}

Frame::Frame(Frame &other) {
    mnId = nNextId++;
    // set last pose???
    setPose(other.mTwc);
    w = other.w;
    v = other.v;
}

void Frame::setPose(const cv::Mat &Twc_)
{
    Twc_.copyTo(mTwc);
    mRwc = mTwc.rowRange(0,3).colRange(0,3);
    mtwc = mTwc.rowRange(0,3).col(3);
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
