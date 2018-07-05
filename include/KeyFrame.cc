#include "KeyFrame.h"
using namespace std;

namespace ev
{

unsigned KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame& F):
    mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp) {
    mnId = nNextId++;
    setPose(F.mTwc);
    w = F.w;
    v = F.v;
    vEvents = &(F.vEvents);
}

cv::Mat KeyFrame::getPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc.clone();
}
void KeyFrame::setPose(const cv::Mat &Twc_)
{
    unique_lock<mutex> lock(mMutexPose);
    Twc_.copyTo(mTwc);
    mRwc = mTwc.rowRange(0,3).colRange(0,3);
    mtwc = mTwc.rowRange(0,3).col(3);
}
}
