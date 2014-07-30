#include "Quaternions.h"
#include <iostream>


// Picked from stackoverflow (by "Jav_Rock")
cv::Mat quatFromRVec(const cv::Mat &rvec)
{
    cv::Mat qwt(4, 1, CV_64FC1);
    const double angle = cv::norm(rvec);  // magnitude of angular velocity

    if (angle > 0.0)
    {
        qwt.at<double>(0)= rvec.at<double>(0) * sin(angle/2.0)/angle;
        qwt.at<double>(1)= rvec.at<double>(1) * sin(angle/2.0)/angle;
        qwt.at<double>(2)= rvec.at<double>(2) * sin(angle/2.0)/angle;
        qwt.at<double>(3)=              1.0 * cos(angle/2.0);
    }else    //to avoid illegal expressions
    {
        qwt.at<double>(0)=qwt.at<double>(1)=qwt.at<double>(2)= 0.0;
        qwt.at<double>(3)= 1.0;
    }
    
    return qwt;
}

// Converted from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
cv::Mat rvecFromQuat(const cv::Mat &qwt) {
    cv::Mat rvec(1, 3, CV_64FC1), qwt_tmp(4, 1, CV_64FC1);
    double x, y, z, angle;
    
    if (qwt_tmp.at<double>(3) > 1) {
        cv::normalize(qwt, qwt_tmp); // if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised
    }
    const double r = qwt_tmp.at<double>(3);
    
    const double s = sqrt(1 - r*r); // assuming quaternion normalised then w is less than 1, so term always positive.
    if (s < 0.001) { // test to avoid divide by zero, s is always positive due to sqrt
        // if s close to zero then direction of axis not important
        x = 1.0; // it is important that axis is normalised, so replace with x=1; y=z=0;
        y = 0.0;
        z = 0.0;
    } else {
        x = qwt_tmp.at<double>(0) / s; // normalise axis
        y = qwt_tmp.at<double>(1) / s;
        z = qwt_tmp.at<double>(2) / s;
    }
    angle = 2 * acos(r);
    
    rvec.at<double>(0) = x*angle;
    rvec.at<double>(1) = y*angle;
    rvec.at<double>(2) = z*angle;
    
    return rvec;
}

/**
 * Quaternion multiplication (taken from stackoverflow (by "Jav_Rock"))
 */
cv::Mat multQuat(const cv::Mat &q1, const cv::Mat &q2)
{
    cv::Mat qwt(4, 1, CV_64FC1);
    
    // First quaternion q1 (x1 y1 z1 r1)
    const double x1=q1.at<double>(0);
    const double y1=q1.at<double>(1);
    const double z1=q1.at<double>(2);
    const double r1=q1.at<double>(3);

    // Second quaternion q2 (x2 y2 z2 r2)
    const double x2=q2.at<double>(0);
    const double y2=q2.at<double>(1);
    const double z2=q2.at<double>(2);
    const double r2=q2.at<double>(3);
    
    qwt.at<double>(0)= x1*r2 + r1*x2 + y1*z2 - z1*y2;   // x component
    qwt.at<double>(1)= r1*y2 - x1*z2 + y1*r2 + z1*x2;   // y component
    qwt.at<double>(2)= r1*z2 + x1*y2 - y1*x2 + z1*r2;   // z component
    qwt.at<double>(3)= r1*r2 - x1*x2 - y1*y2 - z1*z2;   // r component
    
    return qwt;
}

cv::Mat invQuat(const cv::Mat &qwt)
{
    cv::Mat qwt_inv(4, 1, CV_64FC1);
    
    const double x=qwt.at<double>(0);
    const double y=qwt.at<double>(1);
    const double z=qwt.at<double>(2);
    const double r=qwt.at<double>(3);
    
    const double mag2 = (r*r + x*x + y*y + z*z);
    
    qwt_inv.at<double>(0)= -x / mag2;
    qwt_inv.at<double>(1)= -y / mag2;
    qwt_inv.at<double>(2)= -z / mag2;
    qwt_inv.at<double>(3)= 1.0 / mag2;
    
    return qwt_inv;
}
