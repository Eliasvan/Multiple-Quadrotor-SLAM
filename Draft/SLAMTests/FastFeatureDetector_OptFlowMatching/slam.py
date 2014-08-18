#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
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



def main():
    # Tweaking parameters: optimized for FAST features
    max_OF_error = 12.
    max_radius_OF_to_FAST = 2.
    max_dist_ratio = 0.7
    # Tweaking parameters: optimized for chessboard features
    max_OF_error = 12.
    max_radius_OF_to_FAST = 4.
    max_dist_ratio = 1.
    
    boardSize = (8, 6)
    objp = prepare_object_points(boardSize)
    
    cameraMatrix, distCoeffs, imageSize = \
            load_camera_intrinsics("camera_intrinsics.txt")
    
    objectPoints = []
    imagePoints = []
    
    from glob import glob
    images = sorted(glob(os.path.join("captures", "*.jpeg")))
    
    left_img = cv2.imread(images[2])
    right_img = cv2.imread(images[3])
    
    
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector()
    # Initiate BFMatcher object with default values
    matcher = cvh.BFMatcher()
    
    # Detect left (key)points (TODO: make this general, part of the main loop)
    ret, left_points = cvh.extractChessboardFeatures(left_img, boardSize)
    if not ret:
        raise Exception("No chessboard features detected.")
    #left_keypoints = fast.detect(left_img)
    #left_points = np.array([kp.pt for kp in left_keypoints], dtype=np.float32)
    triangl_idxs = set(range(len(left_points)))
    
    # Detect right (key)points
    right_keypoints = fast.detect(right_img)
    right_FAST_points = np.array([kp.pt for kp in right_keypoints], dtype=np.float32)
    
    # Calculate optical flow (= 'OF') field from left to right
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    right_OF_points, status, err = cv2.calcOpticalFlowPyrLK(
            left_gray, right_gray,
            left_points )    # points to start from
    
    # Filter out the OF points with high error
    right_OF_points, right_OF_to_left_idxs = \
            zip(*[ (p, i) for i, p in enumerate(right_OF_points)
                            if status[i] and    # only include correct OF-points
                               err[i] < max_OF_error ])    # error should be low enough
    right_OF_points = np.array(right_OF_points)
    
    # Visualize right_OF_points
    print "Visualize right_OF_points"
    cv2.imshow("img", cv2.drawKeypoints(
            right_img,
            [cv2.KeyPoint(p[0],p[1], 7.) for p in right_OF_points],
            color=rgb(0,0,255) ))    # blue markers with size 7
    cv2.waitKey()
    
    # Visualize right_FAST_points
    print "Visualize right_FAST_points"
    cv2.imshow("img", cv2.drawKeypoints(
            right_img,
            [cv2.KeyPoint(p[0],p[1], 7.) for p in right_FAST_points],
            color=rgb(0,0,255) ))    # blue markers with size 7
    cv2.waitKey()
    
    # Align right_OF_points with right_FAST_points by matching them
    matches_twoNN = matcher.radiusMatch(
            right_OF_points,    # query points
            right_FAST_points,    # train points
            max_radius_OF_to_FAST )
    
    # Filter out ambiguous matches by a ratio-test, and prevent duplicates
    best_dist_matches_by_trainIdx = {}    # duplicate prevention: trainIdx -> match_best_dist
    for query_matches in matches_twoNN:
        # Ratio test
        #if len(query_matches) > 1:
            #print query_matches[0].distance / query_matches[1].distance
        #print len(query_matches)
        if not ( len(query_matches) == 1 or    # only one match, probably a good one
                (len(query_matches) > 1 and    # if more than one, first two shouldn't lie too close
                 query_matches[0].distance / query_matches[1].distance < max_dist_ratio) ):
            continue
            
        # Relink match to use 'left_point' indices
        match = cv2.DMatch(
                right_OF_to_left_idxs[query_matches[0].queryIdx],    # queryIdx: left_points
                query_matches[0].trainIdx,    # trainIdx: right_FAST_points
                query_matches[0].distance )
        
        # Duplicate prevention
        #if match.trainIdx in best_dist_matches_by_trainIdx:
            #print "duplicate!"
        if (not match.trainIdx in best_dist_matches_by_trainIdx or    # no duplicate found
                match.distance < best_dist_matches_by_trainIdx[match.trainIdx].distance):    # replace duplicate if inferior
            best_dist_matches_by_trainIdx[match.trainIdx] = match
    
    # Partition matches to make a distinction between previously triangulated points and non-triangl.
    # memory preallocation
    matches_left_triangl_to_right_FAST = [None] * min(len(best_dist_matches_by_trainIdx), len(triangl_idxs))
    matches_left_non_triangl_to_right_FAST = [None] * (len(best_dist_matches_by_trainIdx) - len(triangl_idxs))
    i = j = 0
    for trainIdx in best_dist_matches_by_trainIdx:
        match = best_dist_matches_by_trainIdx[trainIdx]
        if best_dist_matches_by_trainIdx[trainIdx].queryIdx in triangl_idxs:
            matches_left_triangl_to_right_FAST[i] = match
            i += 1
        else:
            matches_left_non_triangl_to_right_FAST[j] = match
            j += 1
    # and all matches together
    matches_left_to_right_FAST = matches_left_triangl_to_right_FAST + matches_left_non_triangl_to_right_FAST
    
    # Visualize filtered left_points (previously triangulated)
    print "Visualize filtered left_points (previously triangulated)"
    cv2.imshow("img", cv2.drawKeypoints(
            left_img,
            [cv2.KeyPoint(left_points[m.queryIdx][0],left_points[m.queryIdx][1], 7.) for m in matches_left_triangl_to_right_FAST],
            color=rgb(0,0,255) ))    # blue markers with size 7
    cv2.waitKey()
    
    # Visualize filtered right_FAST_points (previously triangulated)
    print "Visualize filtered right_FAST_points (previously triangulated)"
    cv2.imshow("img", cv2.drawKeypoints(
            right_img,
            [cv2.KeyPoint(right_FAST_points[m.trainIdx][0],right_FAST_points[m.trainIdx][1], 7.) for m in matches_left_triangl_to_right_FAST],
            color=rgb(0,0,255) ))    # blue markers with size 7
    cv2.waitKey()
    
    # Refine chessboard features
    ps = np.array([right_FAST_points[m.trainIdx] for m in matches_left_triangl_to_right_FAST])
    cv2.cornerSubPix(
            right_gray, ps,
            (11,11),    # window
            (-1,-1),    # deadzone
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )    # termination criteria
    ps = ps.reshape(-1, 2)
    
    # Visualize filtered+subpixelCorners right_FAST_points (previously triangulated)
    print "Visualize filtered+subpixelCorners right_FAST_points (previously triangulated)"
    cv2.imshow("img", cv2.drawKeypoints(
            right_img,
            [cv2.KeyPoint(p[0],p[1], 7.) for p in ps],
            color=rgb(0,0,255) ))    # blue markers with size 7
    cv2.waitKey()
    
    # Visualize filtered right_FAST_points (not yet triangulated)
    print "Visualize filtered right_FAST_points (not yet triangulated)"
    cv2.imshow("img", cv2.drawKeypoints(
            right_img,
            [cv2.KeyPoint(right_FAST_points[m.trainIdx][0],right_FAST_points[m.trainIdx][1], 7.) for m in matches_left_non_triangl_to_right_FAST],
            color=rgb(0,0,255) ))    # blue markers with size 7
    cv2.waitKey()
    


if __name__ == "__main__":
    main()
