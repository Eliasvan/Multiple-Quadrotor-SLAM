/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 9 of the cookbook:  
   Computer Vision Programming using the OpenCV Library. 
   by Robert Laganiere, Packt Publishing, 2011.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2010-2011 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include "Quaternions.h"
#include "CameraCalibrator.h"

// Open chessboard images and extract corner points
int CameraCalibrator::addChessboardPoints(
         const std::vector<std::string>& filelist, 
         cv::Size & boardSize)
{
	// the points on the chessboard
    std::vector<cv::Point2f> imageCorners;
    std::vector<cv::Point3f> objectCorners;

    // 3D Scene Points:
    // Initialize the chessboard corners 
    // in the chessboard reference frame
	// The corners are at 3D location (X,Y,Z)= (i,j,0)
	for (int i=0; i<boardSize.height; i++) {
		for (int j=0; j<boardSize.width; j++) {

			objectCorners.push_back(cv::Point3f(i, j, 0.0f));
		}
    }

    // 2D Image points:
    cv::Mat image; // to contain chessboard image
    int successes = 0;
    // for all viewpoints
    for (int i=0; i<filelist.size(); i++) {

        // Open the image
        image = cv::imread(filelist[i],0);

        // Get the chessboard corners
        bool found = cv::findChessboardCorners(
                        image, boardSize, imageCorners);

        // Get subpixel accuracy on the corners
        cv::cornerSubPix(image, imageCorners, 
                  cv::Size(11,11), 
                  cv::Size(-1,-1), 
			cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                          cv::TermCriteria::EPS, 
             30,		// max number of iterations 
             0.001));     // min accuracy

          // If we have a good board, add it to our data
		  if (imageCorners.size() == boardSize.area()) {
			// Add image and scene points from one view
            addPoints(imageCorners, objectCorners);
            successes++;
          }

        //Draw the corners
        cv::drawChessboardCorners(image, boardSize, imageCorners, found);
        cv::imshow("Corners on Chessboard", image);
        cv::waitKey(100);
    }

	return successes;
}

// Clear scene points and corresponding image points
void CameraCalibrator::clearPoints()
{
    // 2D image points from one view
    imagePoints.clear();          
    // corresponding 3D scene points
    objectPoints.clear();
}

// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners)
{
	// 2D image points from one view
	imagePoints.push_back(imageCorners);          
	// corresponding 3D scene points
	objectPoints.push_back(objectCorners);
}

// Calibrate the camera
// returns the re-projection error
double CameraCalibrator::calibrate(const cv::Size &imageSize)
{
	// undistorter must be reinitialized
	mustInitUndistort= true;

	//Output rotations and translations
    std::vector<cv::Mat> rvecs, tvecs;

	// start calibration
	return 
     calibrateCamera(objectPoints, // the 3D points
		            imagePoints,  // the image points
					imageSize,    // image size
					cameraMatrix, // output camera matrix
					distCoeffs,   // output distortion matrix
					rvecs, tvecs, // Rs, Ts 
					flag);        // set options
//					,CV_CALIB_USE_INTRINSIC_GUESS);

}

// remove distortion in an image (after calibration)
cv::Mat CameraCalibrator::remap(const cv::Mat &image)
{
	cv::Mat undistorted;

	if (mustInitUndistort) { // called once per calibration
    
		cv::initUndistortRectifyMap(
			cameraMatrix,  // computed camera matrix
            distCoeffs,    // computed distortion matrix
            cv::Mat(),     // optional rectification (none) 
			cv::Mat(),     // camera matrix to generate undistorted
			cv::Size(640,480),
//            image.size(),  // size of undistorted
            CV_32FC1,      // type of output map
            map1, map2);   // the x and y mapping functions

		mustInitUndistort= false;
	}

	// Apply mapping functions
    cv::remap(image, undistorted, map1, map2, 
		cv::INTER_LINEAR); // interpolation type

	return undistorted;
}


// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled)
{
    // Set the flag used in cv::calibrateCamera()
    flag = 0;
    if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}

// Extract extrinsics between two cameras
void CameraCalibrator::extractExtrinsics()
{
    std::cout << "objectPoints[0] == objectPoints[1] ? "<< (objectPoints[0] == objectPoints[1]) << std::endl;
//     std::cout << "#objectPoints: " << objectPoints.size() << "; #imagePoints: " << imagePoints.size() << std::endl;
    
    // Solve pose for camera 1
    cv::Mat rvec1, tvec1;
    cv::solvePnP(objectPoints[0], imagePoints[0], cameraMatrix, distCoeffs, rvec1, tvec1, false);
    std::cout << "rvec1 = "<< std::endl << " "  << rvec1 << std::endl << std::endl;
    const cv::Mat qwt1 = quatFromRVec(rvec1);
    std::cout << "qwt1 = "<< std::endl << " "  << qwt1 << std::endl << std::endl;
    std::cout << "tvec1 = "<< std::endl << " "  << tvec1 << std::endl << std::endl;
    
    // Solve pose for camera 2
    cv::Mat rvec2, tvec2;
    cv::solvePnP(objectPoints[1], imagePoints[1], cameraMatrix, distCoeffs, rvec2, tvec2, false);
    std::cout << "rvec2 = "<< std::endl << " "  << rvec2 << std::endl << std::endl;
    const cv::Mat qwt2 = quatFromRVec(rvec2);
    std::cout << "qwt2 = "<< std::endl << " "  << qwt2 << std::endl << std::endl;
    std::cout << "tvec2 = "<< std::endl << " "  << tvec2 << std::endl << std::endl;
    
    // Compute relative pose of camera 2 w.r.t. camera 1
//     extrinsic_rvec = rvec2 - rvec1; // WRONG!
    const cv::Mat extrinsic_qwt = multQuat(qwt2, invQuat(qwt1));
    std::cout << "extrinsic_qwt = "<< std::endl << " "  << extrinsic_qwt << std::endl << std::endl;
    extrinsic_rvec = rvecFromQuat(extrinsic_qwt);
    extrinsic_tvec = tvec2 - tvec1;
}
