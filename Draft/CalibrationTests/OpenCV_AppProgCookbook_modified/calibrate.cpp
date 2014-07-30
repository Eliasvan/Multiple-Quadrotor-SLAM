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

/*
 * Compile with:
 *  g++ calibrate.cpp CameraCalibrator.cpp Quaternions.cpp -o calibrate -lopencv_calib3d -lopencv_highgui -lopencv_imgproc -lopencv_core
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "CameraCalibrator.h"

int main()
{

	cv::namedWindow("Image");
	cv::Mat image;
	std::vector<std::string> filelist;

	// generate list of chessboard image filename
	for (int i=1; i<=20; i++) {

		std::stringstream str;
		str << "chessboards/chessboard" << std::setw(2) << std::setfill('0') << i << ".jpg";
		std::cout << str.str() << std::endl;

		filelist.push_back(str.str());
		image= cv::imread(str.str(),0);
		cv::imshow("Image",image);
	
		 cv::waitKey(100);
	}

	// Create calibrator object
    CameraCalibrator cameraCalibrator;
	// add the corners from the chessboard
	cv::Size boardSize(8,6);
	cameraCalibrator.addChessboardPoints(
		filelist,	// filenames of chessboard image
		boardSize);	// size of chessboard
		// calibrate the camera
//     	cameraCalibrator.setCalibrationFlag(true,true);
	double reproj_error = cameraCalibrator.calibrate(image.size());
    std::cout << "Reprojection error: " << reproj_error << std::endl;

    // Image Undistortion
    image = cv::imread(filelist[6]);
	cv::Mat uImage= cameraCalibrator.remap(image);

	// display camera matrix
	cv::Mat cameraMatrix= cameraCalibrator.getCameraMatrix();
	std::cout << "Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
// 	std::cout << cameraMatrix.at<double>(0,0) << " " << cameraMatrix.at<double>(0,1) << " " << cameraMatrix.at<double>(0,2) << std::endl;
// 	std::cout << cameraMatrix.at<double>(1,0) << " " << cameraMatrix.at<double>(1,1) << " " << cameraMatrix.at<double>(1,2) << std::endl;
// 	std::cout << cameraMatrix.at<double>(2,0) << " " << cameraMatrix.at<double>(2,1) << " " << cameraMatrix.at<double>(2,2) << std::endl;
    std::cout << " "  << cameraMatrix << std::endl << std::endl;

    cv::Mat distCoeffs= cameraCalibrator.getDistCoeffs();
    std::cout << "Camera distortion: " << distCoeffs.rows << "x" << distCoeffs.cols << std::endl;
//     std::cout << distCoeffs.at<double>(0,0) << " " << distCoeffs.at<double>(0,1) << " " << distCoeffs.at<double>(0,2) << " " << distCoeffs.at<double>(0,3) << " " << distCoeffs.at<double>(0,4) /*<< " " << distCoeffs.at<double>(0,5) << " " << distCoeffs.at<double>(0,6) << " " << distCoeffs.at<double>(0,7)*/ << std::endl;
    std::cout << " "  << distCoeffs << std::endl << std::endl;

    imshow("Original Image", image);
    imshow("Undistorted Image", uImage);
    cv::waitKey();
    
    
    filelist.clear();
    // generate list of chessboard extrinsic image filename
    for (int i=1; i<=2; i++) {

        std::stringstream str;
        str << "chessboards_extrinsic/chessboard" << std::setw(2) << std::setfill('0') << i << ".jpg";
        std::cout << str.str() << std::endl;

        filelist.push_back(str.str());
        image= cv::imread(str.str(),0);
        cv::imshow("Image",image);
    
         cv::waitKey(100);
    }

    // add the corners from the chessboard
    boardSize = cv::Size(8,6);
    cameraCalibrator.clearPoints();
    cameraCalibrator.addChessboardPoints(
        filelist,   // filenames of chessboard image
        boardSize); // size of chessboard
    // extract extrinsics
    cameraCalibrator.extractExtrinsics();
    
    cv::Mat extrinsic_rvec= cameraCalibrator.getExtrinsicRVec();
    std::cout << "Camera2 relative to Camera1: rotation: " << extrinsic_rvec.rows << "x" << extrinsic_rvec.cols << std::endl;
//     std::cout << extrinsic_rvec.at<double>(0,0) << " " << extrinsic_rvec.at<double>(1,0) << " " << extrinsic_rvec.at<double>(2,0) << std::endl;
    std::cout << " "  << extrinsic_rvec << std::endl << std::endl;
    
    cv::Mat extrinsic_tvec= cameraCalibrator.getExtrinsicTVec();
    std::cout << "Camera2 relative to Camera1: translation: " << extrinsic_tvec.rows << "x" << extrinsic_tvec.cols << std::endl;
//     std::cout << extrinsic_tvec.at<double>(0,0) << " " << extrinsic_tvec.at<double>(1,0) << " " << extrinsic_tvec.at<double>(2,0) << std::endl;
    std::cout << " "  << extrinsic_tvec << std::endl << std::endl;

    
	return 0;
}