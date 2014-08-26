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


def keypoint_coverage(points):
    """Returns the coverage ratio of keypoints, using 'keypoint_coverage_radius' as radius."""
    mask_img = np.zeros((imageSize[1], imageSize[0]), dtype=np.uint8)
    for p in points:
        cvh.circle(mask_img, p, keypoint_coverage_radius, True, thickness=-1)
    print "countZero:", cv2.countNonZero(mask_img), "(total):", mask_img.size
    cv2.imshow("img", mask_img*255)
    cv2.waitKey()
    return cv2.countNonZero(mask_img) / float(mask_img.size)


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
                     all_idxs_tmp):    # list of idxs of points in base_points, matches prev_points to base_points
    
    # Calculate OF (Optical Flow), and filter outliers based on OF error
    new_points, status_OF, err_OF = cv2.calcOpticalFlowPyrLK(prev_img_gray, new_img_gray, prev_points)
    err_OF = err_OF.reshape(-1)
    new_points, new_to_prev_idxs = zip(*[ (p, i) for i, p in enumerate(new_points) if status_OF[i] and err_OF[i] < max_OF_error ])
    new_points = np.array(new_points)
    
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
        # Do pose estimation on already-triangulated points (solvePnp),
        # and triangulation of not-yet triangulated points.
        
        # Check whether we should 
        coverage = keypoint_coverage(new_points)
        print "coverage:", coverage, "vs min_keypoint_coverage:", min_keypoint_coverage
        new_base_points = new_points
        if coverage < min_keypoint_coverage:
            to_add = target_amount_keypoints - len(new_points)
            if to_add > 0:
                print "to_add:", to_add
                ps_extra = cv2.goodFeaturesToTrack(new_img_gray, to_add, corner_quality_level, corner_min_dist).reshape((-1, 2))
                cv2.imshow("img", cv2.drawKeypoints(new_img, [cv2.KeyPoint(p[0],p[1], 7.) for p in ps_extra], color=rgb(0,0,255)))
                cv2.waitKey()
                new_base_points = np.concatenate((new_points, ps_extra))
        
        # Rebase indices to current keyframe
        all_idxs = np.arange(len(new_base_points))
        triangl_idxs = set(all_idxs[:len(new_points)])
        nontriangl_idxs = set(all_idxs[len(new_points):])
        all_idxs_tmp = np.array(all_idxs)
        
        # Now this frame becomes the base (= keyframe)
        base_points = new_base_points
        new_points = new_base_points
    
    return base_points, new_points, all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp


def main():
    global imageSize
    global max_OF_error
    global keypoint_coverage_radius, min_amount_disjunct_keypoints, min_keypoint_coverage
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
    # keypoint_coverage
    keypoint_coverage_radius = int(max_OF_error)
    min_amount_disjunct_keypoints = 120
    min_keypoint_coverage = (math.pi * keypoint_coverage_radius**2 * min_amount_disjunct_keypoints) / (imageSize[0]*imageSize[1])
    # goodFeaturesToTrack
    target_amount_keypoints = 3 * min_amount_disjunct_keypoints
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
    
    # Start frame requires special treatment
    imgs.append(cv2.imread(images[0]))
    imgs_gray.append(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY))
    ret, new_points = cvh.extractChessboardFeatures(imgs[0], boardSize)
    base_points = new_points
    all_idxs = np.arange(len(base_points))
    triangl_idxs = set(all_idxs)
    nontriangl_idxs = set([])
    all_idxs_tmp = np.array(all_idxs)
    
    for i in range(1, len(images)):
        # Frame[i-1] -> Frame[i]
        print "Frame[%s] -> Frame[%s]" % (i-1, i)
        imgs.append(cv2.imread(images[i]))
        imgs_gray.append(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY))
        base_points, new_points, all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp = \
                handle_new_frame(base_points, new_points, imgs[i-1], imgs_gray[i-1], imgs[i], imgs_gray[i], all_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp)


if __name__ == "__main__":
    main()
