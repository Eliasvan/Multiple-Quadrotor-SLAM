#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import cv2

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "python_libs"))
import cv2_helpers as cvh
import calibration_tools

blue = cvh.rgb(0, 0, 255)



# Tweaking parameters
max_OF_error = 12.
max_radius_OF_to_FAST = {}
max_dist_ratio = {}
allow_chessboard_matcher_and_refiner = True

# optimized for chessboard features
max_radius_OF_to_FAST["chessboard"] = 4.    # FAST detects points *around* instead of *on* corners
max_dist_ratio["chessboard"] = 1.    # disable ratio test

# optimized for FAST features
max_radius_OF_to_FAST["FAST"] = 2.
max_dist_ratio["FAST"] = 0.7


# Initiate FAST and BFMatcher object with default values
fast = cv2.FastFeatureDetector()
matcher = cvh.BFMatcher()


def main():
    # Initially known data
    boardSize = (8, 6)
    objp = calibration_tools.grid_objp(boardSize)
    
    cameraMatrix, distCoeffs, imageSize = calibration_tools.load_camera_intrinsics(
            os.path.join("..", "..", "datasets", "webcam", "camera_intrinsics.txt") )
    
    
    # Initiate 2d 3d arrays
    objectPoints = []
    imagePoints = []
    
    
    # Select working (or 'testing') set
    from glob import glob
    images = sorted(glob(os.path.join("..", "..", "datasets", "webcam", "captures2", "*.jpeg")))
    
    
    def main_loop(left_points,
                  left_gray,
                  left_img, right_img,
                  triangl_idxs, chessboard_idxs):
        # Detect right (key)points
        right_keypoints = fast.detect(right_img)
        right_FAST_points = np.array([kp.pt for kp in right_keypoints], dtype=np.float32)
        
        # Visualize right_FAST_points
        print "Visualize right_FAST_points"
        cv2.imshow("img", cv2.drawKeypoints(
                right_img,
                [cv2.KeyPoint(p[0],p[1], 7.) for p in right_FAST_points],
                color=blue ))    # blue markers with size 7
        cv2.waitKey()
        
        # Calculate optical flow (= 'OF') field from left to right
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        right_OF_points, status_OF, err_OF = cv2.calcOpticalFlowPyrLK(
                left_gray, right_gray,
                left_points )    # points to start from
        err_OF = err_OF.reshape(-1)
        
        def match_OF_based(right_OF_points, right_FAST_points,
                        err_OF, status_OF,
                        max_radius_OF_to_FAST, max_dist_ratio,
                        left_point_idxs = None):    # if not None, left_point_idxs specifies mask
            # Filter out the OF points with high error
            right_OF_points, right_OF_to_left_idxs = \
                    zip(*[ (p, i) for i, p in enumerate(right_OF_points)
                                    if status_OF[i] and    # only include correct OF-points
                                    err_OF[i] < max_OF_error and    # error should be low enough
                                    (left_point_idxs == None or i in left_point_idxs) ])    # apply mask
            right_OF_points = np.array(right_OF_points)
            
            # Visualize right_OF_points
            print "Visualize right_OF_points"
            cv2.imshow("img", cv2.drawKeypoints(
                    right_img,
                    [cv2.KeyPoint(p[0],p[1], 7.) for p in right_OF_points],
                    color=blue ))    # blue markers with size 7
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
                if (not match.trainIdx in best_dist_matches_by_trainIdx or    # no duplicate found
                        err_OF[match.queryIdx] <    # replace duplicate if inferior, based on err_OF
                            err_OF[best_dist_matches_by_trainIdx[match.trainIdx].queryIdx]):
                    best_dist_matches_by_trainIdx[match.trainIdx] = match
            
            return best_dist_matches_by_trainIdx
        
        # Match between FAST -> FAST features
        matches_by_trainIdx = match_OF_based(
                right_OF_points, right_FAST_points, err_OF, status_OF,
                max_radius_OF_to_FAST["FAST"],
                max_dist_ratio["FAST"] )
        
        if allow_chessboard_matcher_and_refiner and chessboard_idxs:
            # Match between chessboard -> chessboard features
            matches_by_trainIdx_chessboard = match_OF_based(
                    right_OF_points, right_FAST_points, err_OF, status_OF,
                    max_radius_OF_to_FAST["chessboard"],
                    max_dist_ratio["chessboard"],
                    chessboard_idxs )    # set mask
            
            # Overwrite FAST -> FAST feature matches by chessboard -> chessboard feature matches
            matches_by_trainIdx.update(matches_by_trainIdx_chessboard)
            
            # Refine chessboard features
            chessboard_corners_idxs = list(matches_by_trainIdx_chessboard)
            chessboard_corners = right_FAST_points[chessboard_corners_idxs]
            cv2.cornerSubPix(
                    right_gray, chessboard_corners,
                    (11,11),    # window
                    (-1,-1),    # deadzone
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )    # termination criteria
            right_FAST_points[chessboard_corners_idxs] = chessboard_corners
            
            # Update chessboard_idxs
            chessboard_idxs = set(matches_by_trainIdx_chessboard)
            print "Amount of chessboard features tracked in the new image:", len(chessboard_idxs)
        
        
        # Calculate average filtered OF vector
        trainIdxs = list(matches_by_trainIdx)
        queryIdxs = [matches_by_trainIdx[trainIdx].queryIdx for trainIdx in trainIdxs]
        mean_OF_vector = (right_FAST_points[trainIdxs] - left_points[queryIdxs]).mean(axis=0)
        mean_OF_vector_length = np.linalg.norm(mean_OF_vector)
        print "mean_OF_vector (from LEFT to RIGHT):", mean_OF_vector, ";    mean_OF_vector_length:", mean_OF_vector_length
        
        # Partition matches to make a distinction between previously triangulated points and non-triangl.
        matches_left_triangl_to_right_FAST = []
        matches_left_non_triangl_to_right_FAST = []
        for trainIdx in matches_by_trainIdx:
            match = matches_by_trainIdx[trainIdx]
            if matches_by_trainIdx[trainIdx].queryIdx in triangl_idxs:
                matches_left_triangl_to_right_FAST.append(match)
            else:
                matches_left_non_triangl_to_right_FAST.append(match)
        # and all matches together
        matches_left_to_right_FAST = matches_left_triangl_to_right_FAST + matches_left_non_triangl_to_right_FAST
        
        # Visualize (previously triangulated) left_points of corresponding outlier-filtered right_FAST_points
        print "Visualize LEFT  points (previously triangulated)"
        cv2.imshow("img", cv2.drawKeypoints(
                left_img,
                [cv2.KeyPoint(left_points[m.queryIdx][0],left_points[m.queryIdx][1], 7.) for m in matches_left_triangl_to_right_FAST],
                color=blue ))    # blue markers with size 7
        cv2.waitKey()
        
        # Visualize (previously triangulated) outlier-filtered right_FAST_points
        print "Visualize RIGHT points (previously triangulated)"
        cv2.imshow("img", cv2.drawKeypoints(
                right_img,
                [cv2.KeyPoint(right_FAST_points[m.trainIdx][0],right_FAST_points[m.trainIdx][1], 7.) for m in matches_left_triangl_to_right_FAST],
                color=blue ))    # blue markers with size 7
        cv2.waitKey()
        
        # Visualize (not yet triangulated) outlier-filtered right_FAST_points
        print "Visualize LEFT  points (not yet triangulated)"
        cv2.imshow("img", cv2.drawKeypoints(
                left_img,
                [cv2.KeyPoint(left_points[m.queryIdx][0],left_points[m.queryIdx][1], 7.) for m in matches_left_non_triangl_to_right_FAST],
                color=blue ))    # blue markers with size 7
        cv2.waitKey()
        
        # Visualize (not yet triangulated) outlier-filtered right_FAST_points
        print "Visualize RIGHT points (not yet triangulated)"
        cv2.imshow("img", cv2.drawKeypoints(
                right_img,
                [cv2.KeyPoint(right_FAST_points[m.trainIdx][0],right_FAST_points[m.trainIdx][1], 7.) for m in matches_left_non_triangl_to_right_FAST],
                color=blue ))    # blue markers with size 7
        cv2.waitKey()
        
        
        # Pose estimation and Triangulation
        # ...
        
        # Update triangl_idxs    TODO: filter using outlier-removal by epipolar constraints
        triangl_idxs = set(matches_by_trainIdx)
        
        
        return right_FAST_points, right_gray, triangl_idxs, chessboard_idxs
    
    
    ###----------------------------- Frame 0 (init) -----------------------------###
    
    print "###---------------------- Frame 0 (init) ----------------------###"
    
    left_img = cv2.imread(images[0])
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.imread(images[1])
    
    # Detect left (key)points
    ret, left_points = cvh.extractChessboardFeatures(left_img, boardSize)
    if not ret:
        raise Exception("No chessboard features detected.")
    
    # Set masks to alter matches' priority
    chessboard_idxs = set(range(len(left_points)))    # all chessboard corners are in sight
    triangl_idxs = set(chessboard_idxs)    # the chessboard feature points are already triangulated
    
    # Invoke main loop
    right_FAST_points, right_gray, triangl_idxs, chessboard_idxs = \
            main_loop(left_points, left_gray, left_img, right_img, triangl_idxs, chessboard_idxs)
    
    
    for i in range(2, 14):
        ###----------------------------- Frame i -----------------------------###
        
        print "###---------------------- Frame %s ----------------------###" % i

        # Update data for new frame
        left_img = right_img
        left_gray = right_gray
        right_img = cv2.imread(images[i])
        
        # Use previous feature points
        left_points = right_FAST_points
        
        # Invoke main loop
        right_FAST_points, right_gray, triangl_idxs, chessboard_idxs = \
                main_loop(left_points, left_gray, left_img, right_img, triangl_idxs, chessboard_idxs)
    

if __name__ == "__main__":
    main()
