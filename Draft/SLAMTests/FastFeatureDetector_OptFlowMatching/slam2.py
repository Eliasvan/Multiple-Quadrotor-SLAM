#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import glob
import cv2
import cv2_helpers as cvh
from cv2_helpers import rgb



def prepare_object_points(boardSize):
    """
    Prepare object points, like (0,0,0), (0,1,0), (0,2,0) ... ,(5,7,0).
    """
    objp = np.zeros((np.prod(boardSize), 3), np.float32)
    objp[:,:] = np.array([ map(float, [i, j, 0])
                            for i in range(boardSize[1])
                            for j in range(boardSize[0]) ])
    
    return objp


def load_camera_intrinsics(filename):
    from numpy import array
    cameraMatrix, distCoeffs, imageSize = \
            eval(open(filename, 'r').read())
    return cameraMatrix, distCoeffs, imageSize


def keypoint_mask(points):
    """Returns a mask that covers the keypoints with False, using 'keypoint_coverage_radius' as radius."""
    mask_img = np.ones((imageSize[1], imageSize[0]), dtype=np.uint8)
    for p in points:
        cvh.circle(mask_img, p, keypoint_coverage_radius, False, thickness=-1)
    print "countZero:", cv2.countNonZero(mask_img), "(total):", mask_img.size
    cv2.imshow("img", mask_img*255)
    cv2.waitKey()
    return mask_img


def keyframe_test(points1, points2):
    """Returns True if the two images can be taken as keyframes."""
    homography, mask = cv2.findHomography(points1, points2)
    w, u, vt = cv2.SVDecomp(homography, flags=cv2.SVD_NO_UV)
    w = w.reshape((-1))
    print "w[0]/w[2]:", w[0]/w[2]
    return w[0]/w[2] > homography_condition_threshold



def handle_new_frame(base_points,    # includes image-points of both triangulated as not-yet triangl points of last keyframe
                     prev_points,    # includes image-points of last frame
                     prev_img, prev_img_gray,
                     new_img, new_img_gray,
                     all_idxs, triangl_idxs, nontriangl_idxs,    # indices of points in base_points
                     all_idxs_tmp,    # list of idxs of points in base_points, matches prev_points to base_points
                     objp):    # triangulated 3D points
    
    # Calculate OF (Optical Flow), and filter outliers based on OF error
    new_points, status_OF, err_OF = cv2.calcOpticalFlowPyrLK(prev_img_gray, new_img_gray, prev_points)
    err_OF = err_OF.reshape(-1)
    new_points, new_to_prev_idxs = zip(*[ (p, i) for i, p in enumerate(new_points) if status_OF[i] and err_OF[i] < max_OF_error ])
    new_points = np.array(new_points)
    
    # If there is too much OF error in the entire image, simply reject the frame
    lost_tracks_ratio = (len(prev_points) - len(new_points)) / float(len(prev_points))
    print "# points lost because of excessive OF error / # points before: ", len(prev_points) - len(new_points), "/", len(prev_points), "=", lost_tracks_ratio
    if lost_tracks_ratio > max_lost_tracks_ratio:
        new_points = prev_points
        return False, base_points, new_points, all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, objp, None, None
    
    # Save matches by idxs
    all_idxs_tmp = all_idxs_tmp[list(new_to_prev_idxs)]
    base_points_remaining = base_points[all_idxs_tmp]
    
    # Check whether we got a new keyframe
    is_keyframe = keyframe_test(base_points_remaining, new_points)
    print "is_keyframe:", is_keyframe
    cv2.imshow("img", cvh.drawKeypointsAndMotion(new_img, base_points_remaining, new_points, rgb(0,0,255)))
    cv2.waitKey()
    if is_keyframe:
        # TODO
        # Do pose estimation on all points via fundamentalMatrixRansac() and essential matrix,
        # then triangulation of not-yet triangulated points (see nontriangl_idxs),
        # Then do solvePnpRansac() on all points for a refined pose estimation, and to eliminate outliers (only include 'inliers').
        # If ratio of 'inliers' vs input is too low, reject frame.
        rvec, tvec = None, None    # TODO
        
        # Check whether we should 
        mask_img = keypoint_mask(new_points)
        print "coverage:", 1 - cv2.countNonZero(mask_img)/float(mask_img.size), "vs min_keypoint_coverage:", min_keypoint_coverage    # unused
        new_base_points = new_points
        to_add = target_amount_keypoints - len(new_points)
        if to_add > 0:
            print "to_add:", to_add
            ps_extra = cv2.goodFeaturesToTrack(new_img_gray, to_add, corner_quality_level, corner_min_dist, None, mask_img).reshape((-1, 2))
            cv2.imshow("img", cv2.drawKeypoints(new_img, [cv2.KeyPoint(p[0],p[1], 7.) for p in ps_extra], color=rgb(0,0,255)))
            cv2.waitKey()
            new_base_points = np.concatenate((new_points, ps_extra))
            print "added:", len(ps_extra)
        
        # Rebase indices to current keyframe
        all_idxs = np.arange(len(new_base_points))
        triangl_idxs = set(all_idxs[:len(new_points)])
        nontriangl_idxs = set(all_idxs[len(new_points):])
        all_idxs_tmp = np.array(all_idxs)
        
        # Now this frame becomes the base (= keyframe)
        base_points = new_base_points
        new_points = new_base_points
    
    else:
        # TODO
        # Just do solvePnp() on current frame's (new_points) already triangulated points to get its pose estimation.
        # If ratio of 'inliers' vs input is too low, reject frame.
        
        rvec, tvec = None, None    # TODO
        #filtered_triangl_idxs = np.array(tuple( triangl_idxs & set(all_idxs_tmp) ))
        #ret, rvec, tvec = cv2.solvePnP(
                #objp[filtered_triangl_idxs], base_points[filtered_triangl_idxs], cameraMatrix, distCoeffs )
        #print "rvec: \n%s" % rvec
        #print "tvec: \n%s" % tvec
    
    
    return True, base_points, new_points, all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, objp, rvec, tvec


def main():
    global cameraMatrix, distCoeffs, imageSize
    global max_OF_error, max_lost_tracks_ratio
    global keypoint_coverage_radius, min_keypoint_coverage
    global target_amount_keypoints, corner_quality_level, corner_min_dist
    global homography_condition_threshold
    
    # Initially known data
    boardSize = (8, 6)
    objp = prepare_object_points(boardSize)
    
    cameraMatrix, distCoeffs, imageSize = \
            load_camera_intrinsics("camera_intrinsics.txt")
    
    
    ### Tweaking parameters ###
    # OF calculation
    max_OF_error = 12.
    max_lost_tracks_ratio = 0.5
    # keypoint_coverage
    keypoint_coverage_radius = int(max_OF_error)
    min_keypoint_coverage = 0.2
    # goodFeaturesToTrack
    target_amount_keypoints = int(round((imageSize[0]*imageSize[1]) / (math.pi * keypoint_coverage_radius**2)))    # target is entire image full
    print "target_amount_keypoints:", target_amount_keypoints
    corner_quality_level = 0.01
    corner_min_dist = keypoint_coverage_radius
    # keyframe_test
    homography_condition_threshold = 500    # defined as ratio between max and min singular values
    
    
    # Initiate 2d 3d arrays
    objectPoints = []
    imagePoints = []
    
    
    # Select working (or 'testing') set
    from glob import glob
    images = sorted(glob(os.path.join("captures2", "*.jpeg")))
    
    imgs = []
    imgs_gray = []
    
    rvecs = []
    tvecs = []
    
    # Start frame requires special treatment
    imgs.append(cv2.imread(images[0]))
    imgs_gray.append(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY))
    ret, new_points = cvh.extractChessboardFeatures(imgs[0], boardSize)
    base_points = new_points
    all_idxs = np.arange(len(base_points))
    triangl_idxs = set(all_idxs)
    nontriangl_idxs = set([])
    all_idxs_tmp = np.array(all_idxs)
    ret, rvec, tvec = cv2.solvePnP(    # assume first frame is a proper frame with chessboard fully in-sight
            objp, new_points, cameraMatrix, distCoeffs )
    rvecs.append(rvec)
    tvecs.append(tvec)
    
    for i in range(1, len(images)):
        # Frame[i-1] -> Frame[i]
        print "\nFrame[%s] -> Frame[%s]" % (i-1, i)
        imgs.append(cv2.imread(images[i]))
        imgs_gray.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2GRAY))
        ret, base_points, new_points, all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, objp, rvec, tvec = \
                handle_new_frame(base_points, new_points, imgs[-2], imgs_gray[-2], imgs[-1], imgs_gray[-1], all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, objp)
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:    # frame rejected
            del imgs[-1]
            del imgs_gray[-1]


if __name__ == "__main__":
    main()
