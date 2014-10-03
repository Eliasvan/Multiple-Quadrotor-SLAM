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



def handle_new_frame(base_points,    # includes 2D points of both triangulated as not-yet triangl points of last keyframe
                     prev_points,    # includes 2D points of last frame
                     prev_img, prev_img_gray,
                     new_img, new_img_gray,
                     triangl_idxs, nontriangl_idxs,    # indices of 2D points in base_points
                     imgp_to_objp_idxs,    # indices from 2D points in base_points to 3D points in objp
                     all_idxs_tmp,    # list of idxs of 2D points in base_points, matches prev_points to base_points
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
        return False, base_points, new_points, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, None, None
    
    # Save matches by idxs
    new_to_prev_idxs = list(new_to_prev_idxs)
    all_idxs_tmp_old = all_idxs_tmp
    all_idxs_tmp = all_idxs_tmp[new_to_prev_idxs]
    base_points_remaining = base_points[all_idxs_tmp]
    
    # Check whether we got a new keyframe
    is_keyframe = keyframe_test(base_points_remaining, new_points)
    print "is_keyframe:", is_keyframe
    cv2.imshow("img", cvh.drawKeypointsAndMotion(new_img, base_points_remaining, new_points, rgb(0,0,255)))
    cv2.waitKey()
    if is_keyframe:
        # Rebase indices to current keyframe
        #triangl_idxs &= set(all_idxs_tmp)
        #triangl_idxs = set([set(all_idxs_tmp).index(i) for i in (triangl_idxs & set(all_idxs_tmp))])
        triangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in triangl_idxs)
        #nontriangl_idxs &= set(all_idxs_tmp)
        #nontriangl_idxs = set([set(all_idxs_tmp).index(i) for i in (nontriangl_idxs & set(all_idxs_tmp))])
        nontriangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in nontriangl_idxs)
        imgp_to_objp_idxs = imgp_to_objp_idxs[all_idxs_tmp]
        all_idxs_tmp = np.arange(len(all_idxs_tmp))
        
        # TODO
        # Do pose estimation on all points via fundamentalMatrixRansac() and essential matrix,
        # then triangulation of not-yet triangulated points (see nontriangl_idxs),
        # Then do solvePnpRansac() on all (triangulated) points for a refined pose estimation,
        # and to eliminate outliers (only include 'inliers').
        # If ratio of 'inliers' vs input is too low, reject frame (and undo rebasing of indices).
        rvec, tvec = None, None    # TODO
        # use imgp_to_objp_idxs[np.array(tuple( triangl_idxs ))] and
        #     base_points_remaining[np.array(tuple( nontriangl_idxs ))]
        # to do the calcs, and output nontriangl_idxs_done:
        #     calc it starting from (nontriangl_idxs) array
        nontriangl_idxs_done = set(nontriangl_idxs)    # TODO: replace this with the successfully triangulated 3D points
        objp_done = np.zeros((len(nontriangl_idxs_done), 3))    # TODO
        imgp_to_objp_idxs[np.array(tuple(nontriangl_idxs_done), dtype=int)] = np.arange(len(objp), len(objp) + len(objp_done))
        objp = np.concatenate((objp, objp_done))
        triangl_idxs |= nontriangl_idxs_done
        #nontriangl_idxs -= nontriangl_idxs_done
        nontriangl_idxs.clear()    # the remaining nontriangl_idxs that failed to be triangulated are outliers, forget about them
        
        # Rebase indices and points to inliers
        all_idxs = triangl_idxs | nontriangl_idxs
        all_idxs_tmp = np.array([i for i in range(len(all_idxs_tmp)) if i in all_idxs])
        imgp_to_objp_idxs = imgp_to_objp_idxs[all_idxs_tmp]
        new_points = new_points[all_idxs_tmp]
        
        # Rebase indices to current keyframe
        triangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in triangl_idxs)
        nontriangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in nontriangl_idxs)
        all_idxs_tmp = np.arange(len(all_idxs_tmp))
        
        # Check whether we should add new 2d points
        mask_img = keypoint_mask(new_points)
        print "coverage:", 1 - cv2.countNonZero(mask_img)/float(mask_img.size), "vs min_keypoint_coverage:", min_keypoint_coverage    # unused
        to_add = target_amount_keypoints - len(new_points)
        if to_add > 0:
            print "to_add:", to_add
            ps_extra = cv2.goodFeaturesToTrack(new_img_gray, to_add, corner_quality_level, corner_min_dist, None, mask_img).reshape((-1, 2))
            cv2.imshow("img", cv2.drawKeypoints(new_img, [cv2.KeyPoint(p[0],p[1], 7.) for p in ps_extra], color=rgb(0,0,255)))
            cv2.waitKey()
            if len(ps_extra):
                new_base_points = np.concatenate((new_points, ps_extra))
                extra_idxs = np.arange(len(new_points), len(new_base_points))
                nontriangl_idxs |= set(extra_idxs)
                imgp_to_objp_idxs = np.concatenate((imgp_to_objp_idxs, np.array([-1] * len(ps_extra))))    # add '-1' idxs, because not-yet-triangl
                all_idxs_tmp = np.concatenate((all_idxs_tmp, extra_idxs))
            print "added:", len(ps_extra)
        else:
            new_base_points = new_points
        
        # Now this frame becomes the base (= keyframe)
        base_points = new_base_points
        new_points = new_base_points
    
    else:
        # TODO
        # Just do solvePnp() on current frame's (new_points) already triangulated points to get its pose estimation.
        # If ratio of 'inliers' vs input is too low, reject frame.
        
        filtered_triangl_idxs = np.array(tuple(triangl_idxs & set(all_idxs_tmp)), dtype=int)
        filtered_triangl_points = base_points[filtered_triangl_idxs]
        filtered_triangl_objp = objp[imgp_to_objp_idxs[filtered_triangl_idxs]]
        ret, rvec, tvec = cv2.solvePnP(    # TODO: useExtrinsicGuess
                filtered_triangl_objp, filtered_triangl_points, cameraMatrix, distCoeffs )
        print "rvec: \n%s" % rvec
        print "tvec: \n%s" % tvec
        
        imgp_reproj, jacob = cv2.projectPoints(
                filtered_triangl_objp, rvec, tvec, cameraMatrix, distCoeffs )
        reproj_error = (
                    ((imgp_reproj.reshape(-1, 2) - filtered_triangl_points)**2).sum(axis=0) / 
                    np.prod(boardSize)
                ).sum() / 2
        print "solvePnP reproj_error:", reproj_error
        if reproj_error > max_solvePnP_reproj_error:    # reject frame
            new_points = prev_points
            return False, base_points, new_points, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp_old, objp, None, None
    
    
    return True, base_points, new_points, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, rvec, tvec


def main():
    global boardSize
    global cameraMatrix, distCoeffs, imageSize
    global max_OF_error, max_lost_tracks_ratio
    global keypoint_coverage_radius, min_keypoint_coverage
    global target_amount_keypoints, corner_quality_level, corner_min_dist
    global homography_condition_threshold
    global max_solvePnP_reproj_error, max_fundMat_reproj_error
    
    # Initially known data
    boardSize = (8, 6)
    objp = prepare_object_points(boardSize)    # 3D points
    
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
    # reprojection error
    max_solvePnP_reproj_error = 0.5
    max_fundMat_reproj_error = 2.0
    
    
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
    base_points = new_points    # 2D points
    all_idxs_tmp = np.arange(len(base_points))
    triangl_idxs = set(all_idxs_tmp)
    nontriangl_idxs = set()
    imgp_to_objp_idxs = np.array(tuple(triangl_idxs), dtype=int)
    ret, rvec, tvec = cv2.solvePnP(    # assume first frame is a proper frame with chessboard fully in-sight
            objp, new_points, cameraMatrix, distCoeffs )
    rvecs.append(rvec)
    tvecs.append(tvec)
    
    for i in range(1, len(images)):
        # Frame[i-1] -> Frame[i]
        print "\nFrame[%s] -> Frame[%s]" % (i-1, i)
        imgs.append(cv2.imread(images[i]))
        imgs_gray.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2GRAY))
        ret, base_points, new_points, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, rvec, tvec = \
                handle_new_frame(base_points, new_points, imgs[-2], imgs_gray[-2], imgs[-1], imgs_gray[-1], triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp)
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:    # frame rejected
            del imgs[-1]
            del imgs_gray[-1]


if __name__ == "__main__":
    main()
