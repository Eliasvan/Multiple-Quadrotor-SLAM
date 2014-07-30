#ifndef QUATERNIONS_H
#define QUATERNIONS_H

#include <math.h>
#include <opencv2/core/core.hpp>


// Picked from stackoverflow (by "Jav_Rock")
cv::Mat quatFromRVec(const cv::Mat &rvec);

// Converted from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
cv::Mat rvecFromQuat(const cv::Mat &qwt);

/**
 * Quaternion multiplication (taken from stackoverflow (by "Jav_Rock"))
 */
cv::Mat multQuat(const cv::Mat &q1, const cv::Mat &q2);

cv::Mat invQuat(const cv::Mat &qwt);

#endif // QUATERNIONS_H
