#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Code originates from:
        http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""
from math import degrees
import numpy as np
import quaternions as qwts
import glob
import cv2
import cv2_helpers as cvh
from cv2_helpers import rgb, format3DVector



def prepare_object_points(boardSize):
    """
    Prepare object points, like (0,0,0), (0,1,0), (0,2,0) ... ,(5,7,0).
    """
    objp = np.zeros((np.prod(boardSize), 3), np.float32)
    objp[:,:] = np.array([ map(float, [i, j, 0])
                            for i in range(boardSize[1])
                            for j in range(boardSize[0]) ])
    
    return objp


def calibrate_camera(images, objp, boardSize):
    # Arrays to store object points and image points from all the images.
    objectPoints = []    # 3d point in real world space
    imagePoints = []    # 2d points in image plane

    test_image = cv2.imread(images[0])
    imageSize = (test_image.shape[1], test_image.shape[0])

    # Read images
    for fname in images:
        img = cv2.imread(fname)
        ret, corners = cvh.extractChessboardFeatures(img, boardSize)

        # If chessboard corners are found, add object points and image points
        if ret == True:
            objectPoints.append(objp)
            imagePoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(
                    img, boardSize, corners, ret )
            cv2.imshow('img', img)
            cv2.waitKey(100)

    # Calibration
    reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints, imagePoints, imageSize )
    print "cameraMatrix:\n", cameraMatrix
    print "distCoeffs:\n", distCoeffs
    print "reproj_error:", reproj_error
    
    return reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs, \
            objectPoints, imagePoints, imageSize


def undistort_image(img, cameraMatrix, distCoeffs, imageSize):
    # Refine cameraMatrix, and calculate ReqionOfInterest
    cameraMatrix_new, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix, distCoeffs, imageSize,
            1 )    # all source image pixels retained in undistorted image

    # undistort
    mapX, mapY = cv2.initUndistortRectifyMap(
            cameraMatrix, distCoeffs,
            None,    # optional rectification transformation
            cameraMatrix_new, imageSize,
            5 )    # type of the first output map (CV_32FC1)
    img_undistorted = cv2.remap(
            img, mapX, mapY, cv2.INTER_LINEAR )

    # crop the image
    x,y, w,h = roi
    img_undistorted = img_undistorted[y:y+h, x:x+w]

    cv2.imshow("Original Image", img);
    cv2.imshow("Undistorted Image", img_undistorted)
    
    return img_undistorted, roi


def reprojection_error(cameraMatrix, distCoeffs, rvecs, tvecs, objectPoints, imagePoints, boardSize):
    # Re-projection Error
    mean_error = np.zeros((1, 2))
    square_error = np.zeros((1, 2))
    n_images = len(imagePoints)

    for i in xrange(n_images):
        imgp_reproj, jacob = cv2.projectPoints(
                objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs )
        error = imgp_reproj.reshape(-1, 2) - imagePoints[i]
        mean_error += abs(error).sum(axis=0) / np.prod(boardSize)
        square_error += (error**2).sum(axis=0) / np.prod(boardSize)

    mean_error = cv2.norm(mean_error / n_images)
    square_error = np.sqrt(square_error.sum() / n_images)

    print "mean absolute error:", mean_error
    print "square error:", square_error
    
    return mean_error, square_error


def realtime_pose_estimation(device_id, filename_base_extrinsics, cameraMatrix, distCoeffs, objp, boardSize):
    """
    View chessboard with axis-system drawn on-top,
    with additional information about the pose of
    the CAMERA w.r.t. the WORLD (= chessboard).
    
    A 
    """
    axis_system_objp = np.array([ [0., 0., 0.],   # Origin (black)
                                  [4., 0., 0.],   # X-axis (red)
                                  [0., 4., 0.],   # Y-axis (green)
                                  [0., 0., 4.] ]) # Z-axis (blue)
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = .5
    mlt = cvh.MultilineText()
    cap = cv2.VideoCapture(device_id)

    imageNr = 0    # keyframe image id
    rvec_prev = np.zeros((3, 1))
    rvec = None
    tvec_prev = np.zeros((3, 1))
    tvec = None

    # Loop until 'Q' or ESC pressed
    last_key_pressed = 0
    while not last_key_pressed in (ord('q'), 27):
        ret_, img = cap.read()
        ret, corners = cvh.extractChessboardFeatures(img, boardSize)

        # If valid features found, solve for 'rvec' and 'tvec'
        if ret == True:
            ret, rvec, tvec = cv2.solvePnP(    # TODO: use Ransac version for other types of features
                    objp, corners, cameraMatrix, distCoeffs )

            # Project axis-system
            imgp_reproj, jacob = cv2.projectPoints(
                    axis_system_objp, rvec, tvec, cameraMatrix, distCoeffs )
            rounding = np.vectorize(lambda x: int(round(x)))
            origin, xAxis, yAxis, zAxis = rounding(imgp_reproj.reshape(-1, 2)) # round to nearest int
            
            # OpenCV's 'rvec' and 'tvec' seem to be defined as follows:
            #   'rvec': rotation transformation: "CAMERA axis-system -> WORLD axis-system"
            #   'tvec': translation of "CAMERA -> WORLD", defined in the "CAMERA axis-system"
            rvec *= -1    # convert to: "WORLD axis-system -> CAMERA axis-system"
            tvec = cv2.Rodrigues(rvec)[0].dot(tvec)    # bring to "WORLD axis-system", ...
            tvec *= -1    # ... and change direction to "WORLD -> CAMERA"
            
            # Calculate pose relative to last keyframe
            rvec_rel = -qwts.delta_rvec(-rvec, -rvec_prev)    # calculate the inverse of the rotation between subsequent "CAMERA -> WORLD" rotations
            tvec_rel = tvec - tvec_prev
            
            # Extract axis and angle, to enhance representation
            rvec_axis, rvec_angle = qwts.axis_and_angle_from_rvec(rvec)
            rvec_rel_axis, rvec_rel_angle = qwts.axis_and_angle_from_rvec(rvec_rel)
            
            # Draw axis-system
            cvh.line(img, origin, xAxis, rgb(255,0,0), thickness=2, lineType=cv2.CV_AA)
            cvh.line(img, origin, yAxis, rgb(0,255,0), thickness=2, lineType=cv2.CV_AA)
            cvh.line(img, origin, zAxis, rgb(0,0,255), thickness=2, lineType=cv2.CV_AA)
            cvh.circle(img, origin, 4, rgb(0,0,0), thickness=-1)    # filled circle, radius 4
            cvh.circle(img, origin, 5, rgb(255,255,255), thickness=2)    # white 'O', radius 5
            
            # Draw pose information
            texts = []
            texts.append("Current pose:")
            texts.append("    rvec: %s * %.1fdeg" % (format3DVector(rvec_axis), degrees(rvec_angle)))
            texts.append("    tvec: %s" % format3DVector(tvec))
            texts.append("Relative to previous pose:")
            texts.append("    rvec: %s * %.1fdeg" % (format3DVector(rvec_rel_axis), degrees(rvec_rel_angle)))
            texts.append("    tvec: %s" % format3DVector(tvec_rel))
            
            mlt.text(texts[0], fontFace, fontScale*1.5, rgb(150,0,0), thickness=2)
            mlt.text(texts[1], fontFace, fontScale, rgb(255,0,0))
            mlt.text(texts[2], fontFace, fontScale, rgb(255,0,0))
            mlt.text(texts[3], fontFace, fontScale*1.5, rgb(150,0,0), thickness=2)
            mlt.text(texts[4], fontFace, fontScale, rgb(255,0,0))
            mlt.text(texts[5], fontFace, fontScale, rgb(255,0,0))
            mlt.putText(img, (img.shape[1], img.shape[0]))    # put text in bottom-right corner

        # Show Image
        cv2.imshow("Image (with axis-system)", img)
        mlt.clear()
        
        # Save keyframe image when SPACE is pressed
        last_key_pressed = cv2.waitKey(1) & 0xFF
        if last_key_pressed == ord(' ') and ret:
            filename = filename_base_extrinsics + str(imageNr) # TODO: replace 'chessboards_extrinsic/chessboard' by parameter
            cv2.imwrite(filename + ".jpg", img)    # write image to jpg-file
            textTotal = '\n'.join(texts)
            open(filename + ".txt", 'w').write(textTotal)    # write data to txt-file
            
            print "Saved keyframe image+data to", filename, ":"
            print textTotal
            
            imageNr += 1
            rvec_prev = rvec
            tvec_prev = tvec



def main():
    boardSize = (8, 6)
    filename_base_chessboards = "chessboards/chessboard*.jpg"
    filename_distorted = "chessboards/chessboard07.jpg"
    filename_base_extrinsics = "chessboards_extrinsic/chessboard"
    device_id = 1    # webcam

    print "Choose between:"
    print "    1: prepare_object_points"
    print "    2: calibrate_camera"
    print "    3: undistort_image"
    print "    4: reprojection_error"
    print "    5: realtime_pose_estimation"
    inp = ""
    while inp.lower() != "q":
        inp = raw_input("\n: ")
        
        if inp == "1":
            boardSize_inp = raw_input("boardSize [%s]: " % repr(boardSize))
            print    # add new-line
            if boardSize_inp:
                exec "boardSize = " + boardSize_inp
            
            objp = prepare_object_points(boardSize)
        
        elif inp == "2":
            filename_base_chessboards_inp = raw_input("filename_base_chessboards [%s]: " % repr(filename_base_chessboards))
            print    # add new-line
            if filename_base_chessboards_inp:
                filename_base_chessboards = filename_base_chessboards_inp
            images = sorted(glob.glob(filename_base_chessboards))
            
            reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs, objectPoints, imagePoints, imageSize = \
                    calibrate_camera(images, objp, boardSize)
            cv2.destroyAllWindows()
        
        elif inp == "3":
            filename_distorted_inp = raw_input("filename_distorted [%s]: " % repr(filename_distorted))
            print    # add new-line
            if filename_distorted_inp:
                filename_distorted = filename_distorted_inp
            img = cv2.imread(filename_distorted)
            
            undistort_image(img, cameraMatrix, distCoeffs, imageSize)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        elif inp == "4":
            reprojection_error(cameraMatrix, distCoeffs, rvecs, tvecs, objectPoints, imagePoints, boardSize)
        
        elif inp == "5":
            filename_base_extrinsics_inp = raw_input("filename_base_extrinsics [%s]: " % repr(filename_base_extrinsics))
            print    # add new-line
            if filename_base_extrinsics_inp:
                filename_base_extrinsics = filename_base_extrinsics_inp
            
            realtime_pose_estimation(device_id, filename_base_extrinsics, cameraMatrix, distCoeffs, objp, boardSize)
            cv2.destroyAllWindows()



main()
