#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import glob
import cv2
import cv2_helpers as cvh
from cv2_helpers import rgb
import transforms as trfm



def prepare_object_points(boardSize):
    """
    Prepare object points, like (0,0,0), (0,1,0), (0,2,0) ... ,(5,7,0).
    """
    objp = np.zeros((np.prod(boardSize), 3), dtype=np.float32)
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


# Initialize consts or tmp vars to be used in linear_LS_triangulation()
linear_LS_triangulation_c = -np.eye(2, 3)
linear_LS_triangulation_A = np.zeros((4, 3))
linear_LS_triangulation_b = np.zeros((4, 1))

def linear_LS_triangulation(u, P, u1, P1):
    """
    Linear Least Squares based triangulation.
    TODO: flip rows and columns to increase performance (improve for cache)
    
    (u, P) is the reference pair containing homogenous image coordinates (x, y) and the corresponding camera matrix.
    (u1, P1) is the second pair.
    
    u and u1 are matrices: amount of points equals #columns and should be equal for u and u1.
    """
    global linear_LS_triangulation_A, linear_LS_triangulation_b
    
    # Create array of triangulated points
    x = np.zeros((3, u.shape[1]))
    
    # Initialize C matrices
    C = np.array(linear_LS_triangulation_c)
    C1 = np.array(linear_LS_triangulation_c)
    
    for i in range(u.shape[1]):
        # Build C matrices, to visualize calculation structure
        C[:, 2] = u[:, i]
        C1[:, 2] = u1[:, i]
        
        # Build A matrix
        linear_LS_triangulation_A[0:2, :] = C.dot(P[0:3, 0:3])    # C * R
        linear_LS_triangulation_A[2:4, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        
        # Build b vector
        linear_LS_triangulation_b[0:2, :] = C.dot(P[0:3, 3:4])    # C * t
        linear_LS_triangulation_b[2:4, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        linear_LS_triangulation_b *= -1
        
        # Solve for x vector
        cv2.solve(linear_LS_triangulation_A, linear_LS_triangulation_b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return np.array(x, dtype=np.float32)    # solvePnPRansac() seems to dislike float64...


def reprojection_error(objp, imgp, rvec, tvec, cameraMatrix, distCoeffs):
    imgp_reproj, jacob = cv2.projectPoints(
            objp, rvec, tvec, cameraMatrix, distCoeffs )
    return np.sqrt(((imgp_reproj.reshape(-1, 2) - imgp)**2).sum() / float(len(imgp))), imgp_reproj


def draw_axis_system(img, rvec, tvec, cameraMatrix, distCoeffs):
    axis_system_objp = np.array([ [0., 0., 0.],      # Origin (black)
                                  [4., 0., 0.],      # X-axis (red)
                                  [0., 4., 0.],      # Y-axis (green)
                                  [0., 0., 4.] ])    # Z-axis (blue)
    rounding = np.vectorize(lambda x: int(round(x)))
    imgp_reproj, jacob = cv2.projectPoints(
            axis_system_objp, rvec, tvec, cameraMatrix, distCoeffs )
    origin, xAxis, yAxis, zAxis = rounding(imgp_reproj.reshape(-1, 2)) # round to nearest int
    cvh.line(img, origin, xAxis, rgb(255,0,0), thickness=2, lineType=cv2.CV_AA)
    cvh.line(img, origin, yAxis, rgb(0,255,0), thickness=2, lineType=cv2.CV_AA)
    cvh.line(img, origin, zAxis, rgb(0,0,255), thickness=2, lineType=cv2.CV_AA)
    cvh.circle(img, origin, 4, rgb(0,0,0), thickness=-1)    # filled circle, radius 4
    cvh.circle(img, origin, 5, rgb(255,255,255), thickness=2)    # white 'O', radius 5
    return img

def check_triangulation_input(base_img, new_img,
                              imgp0, imgp1,
                              rvec_keyfr, tvec_keyfr, rvec, tvec,
                              cameraMatrix, distCoeffs):
    img = cvh.drawKeypointsAndMotion(base_img, imgp1, imgp0, rgb(0,0,255))
    draw_axis_system(img, rvec_keyfr, tvec_keyfr, cameraMatrix, distCoeffs)
    print "base_img check"
    cv2.imshow("img", img)
    cv2.waitKey()
    
    img = cvh.drawKeypointsAndMotion(new_img, imgp0, imgp1, rgb(0,0,255))
    draw_axis_system(img, rvec, tvec, cameraMatrix, distCoeffs)
    print "new_img check"
    cv2.imshow("img", img)
    cv2.waitKey()


### Attempt to simplify re-indexing

def idxs_get_new_points_by_idxs(selection_idxs,
                                new_points, all_idxs_tmp):
    """Get selection of new_points corresponding with selection_idxs (of which elements correspond with idxs in base_points).
    type(selection_idxs) must be "set".
    """
    new_points_sel = np.array([p for p, idx in zip(new_points, all_idxs_tmp) if idx in selection_idxs])
    return new_points_sel

def idxs_update_by_idxs(preserve_idxs,
                        triangl_idxs, nontriangl_idxs, all_idxs_tmp):
    """Only preserve preserve_idxs (elements correspond with idxs in base_points).
    type(preserve_idxs) must be "set".
    To update new_points, first use idxs_get_new_points_by_idxs().
    """
    triangl_idxs &= preserve_idxs
    nontriangl_idxs &= preserve_idxs
    #print "first approach (fastest):"
    #print np.array(sorted(preserve_idxs), dtype=int)
    #print "second approach:"
    #print np.array(tuple(set(tuple(preserve_idxs))), dtype=int)
    #print "third approach:"
    #print np.array([i for i in all_idxs_tmp if i in preserve_idxs])
    all_idxs_tmp = np.array(sorted(preserve_idxs), dtype=int)
    return triangl_idxs, nontriangl_idxs, all_idxs_tmp

def idxs_add_objp(objp_extra, triangl_idxs_extra,
                  objp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp):
    """Add new object-points objp_extra, they correspond with elements in base_points of which their idx is in triangl_idxs_extra.
    type(triangl_idxs_extra) must be "set".
    """
    # NOTICE: see approaches explained in idxs_update_by_idxs() to compare indexing methods
    imgp_to_objp_idxs[np.array(sorted(triangl_idxs_extra), dtype=int)] = np.arange(len(objp), len(objp) + len(objp_extra))
    triangl_idxs |= triangl_idxs_extra
    nontriangl_idxs -= triangl_idxs_extra
    objp = np.concatenate((objp, objp_extra))
    return objp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs

def idxs_rebase_and_add_imgp(imgp_extra,
                             base_points, new_points, objp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp):
    """Rebase all idxs to new_points, instead of to base_points; also add new image-points imgp_extra."""
    imgp_to_objp_idxs = imgp_to_objp_idxs[all_idxs_tmp]
    triangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in triangl_idxs)
    nontriangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in nontriangl_idxs)
    all_idxs_tmp = np.arange(len(all_idxs_tmp))
    
    extra_idxs = np.arange(len(new_points), len(new_points) + len(imgp_extra))
    new_points = np.concatenate((new_points, imgp_extra))
    nontriangl_idxs |= set(extra_idxs)
    imgp_to_objp_idxs = np.concatenate((imgp_to_objp_idxs, -np.ones((len(imgp_extra)), dtype=int)))    # add '-1' idxs, because not-yet-triangl
    all_idxs_tmp = np.concatenate((all_idxs_tmp, extra_idxs))
    base_points = new_points
    return base_points, new_points, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp

def handle_new_frame(base_points,    # includes 2D points of both triangulated as not-yet triangl points of last keyframe
                     prev_points,    # includes 2D points of last frame
                     prev_img, prev_img_gray,
                     new_img, new_img_gray,
                     triangl_idxs, nontriangl_idxs,    # indices of 2D points in base_points
                     imgp_to_objp_idxs,    # indices from 2D points in base_points to 3D points in objp
                     all_idxs_tmp,    # list of idxs of 2D points in base_points, matches prev_points to base_points
                     objp,    # triangulated 3D points
                     rvec_keyfr, tvec_keyfr,    # rvec and tvec of last keyframe
                     base_img):    # used for debug
    
    # Save initial indexing state
    triangl_idxs_old = set(triangl_idxs)
    nontriangl_idxs_old = set(nontriangl_idxs)
    all_idxs_tmp_old = np.array(all_idxs_tmp)
    
    # Calculate OF (Optical Flow), and filter outliers based on OF error
    new_points, status_OF, err_OF = cv2.calcOpticalFlowPyrLK(prev_img_gray, new_img_gray, prev_points)
    new_to_prev_idxs = np.where(np.logical_and((status_OF.reshape(-1) == 1), (err_OF.reshape(-1) < max_OF_error)))[0]
    
    # If there is too much OF error in the entire image, simply reject the frame
    lost_tracks_ratio = (len(prev_points) - len(new_to_prev_idxs)) / float(len(prev_points))
    print "# points lost because of excessive OF error / # points before: ", len(prev_points) - len(new_to_prev_idxs), "/", len(prev_points), "=", lost_tracks_ratio
    if lost_tracks_ratio > max_lost_tracks_ratio:
        return False, base_points, prev_points, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, None, None, rvec_keyfr, tvec_keyfr, base_img
    
    # Save matches by idxs
    preserve_idxs = set(all_idxs_tmp[new_to_prev_idxs])
    new_points = new_points[new_to_prev_idxs]
    triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
            preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
    #cv2.cornerSubPix(    # TODO: activate this secret weapon
                #new_img_gray, new_points,
                #(corner_min_dist,corner_min_dist),    # window
                #(-1,-1),    # deadzone
                #(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )    # termination criteria
        ##corners = corners.reshape(-1, 2)
    
    # Do solvePnpRansac() on current frame's (new_points) already triangulated points, then solvePnp() on inliers, to get its pose estimation.
    # If ratio of 'inliers' vs input is too low, reject frame.
    triangl_idxs_array = np.array(sorted(triangl_idxs))
    filtered_triangl_points = idxs_get_new_points_by_idxs(triangl_idxs, new_points, all_idxs_tmp)
    filtered_triangl_objp = objp[imgp_to_objp_idxs[triangl_idxs_array]]
    print "Doing solvePnP() on", filtered_triangl_objp.shape[0], "points"
    rvec, tvec, inliers = cv2.solvePnPRansac(
            filtered_triangl_objp, filtered_triangl_points, cameraMatrix, distCoeffs )
    print "rvec: \n%s" % rvec
    print "tvec: \n%s" % tvec
    if inliers == None:    # reject frame
        return False, base_points, prev_points, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, None, None, rvec_keyfr, tvec_keyfr, base_img
    solvePnP_outlier_ratio = (len(triangl_idxs) - len(inliers)) / float(len(triangl_idxs))
    print "solvePnP_outlier_ratio:", solvePnP_outlier_ratio
    if solvePnP_outlier_ratio > max_solvePnP_outlier_ratio:
        return False, base_points, prev_points, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, None, None, rvec_keyfr, tvec_keyfr, base_img
    reproj_error, imgp_reproj1 = reprojection_error(filtered_triangl_objp, filtered_triangl_points, rvec, tvec, cameraMatrix, distCoeffs)    # TODO: remove
    print "solvePnP reproj_error:", reproj_error
    i3 = np.array(new_img)
    for imgppr, imgppp in zip(filtered_triangl_points, imgp_reproj1): cvh.line(i3, imgppr.T, imgppp.T, rgb(255,0,0))
    cv2.imshow("img", i3)
    cv2.waitKey()
    
    triangl_idxs_array = triangl_idxs_array[inliers.reshape(-1)]
    filtered_triangl_points, filtered_triangl_objp = filtered_triangl_points[inliers], filtered_triangl_objp[inliers]
    preserve_idxs = set(triangl_idxs_array) | nontriangl_idxs    # only preserve inliers
    new_points = idxs_get_new_points_by_idxs(preserve_idxs, new_points, all_idxs_tmp)
    triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
            preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
    ret, rvec, tvec = cv2.solvePnP(
            filtered_triangl_objp, filtered_triangl_points, cameraMatrix, distCoeffs )
    print "rvec refined: \n%s" % rvec
    print "tvec refined: \n%s" % tvec
    reproj_error, imgp_reproj2 = reprojection_error(filtered_triangl_objp.reshape(-1, 3), filtered_triangl_points.reshape(-1, 2), rvec, tvec, cameraMatrix, distCoeffs)
    print "solvePnP refined reproj_error:", reproj_error
    
    # Verify poses by reprojection error
    if reproj_error > max_solvePnP_reproj_error:    # reject frame
        return False, base_points, prev_points, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, None, None, rvec_keyfr, tvec_keyfr, base_img
    i0 = draw_axis_system(np.array(new_img), rvec, tvec, cameraMatrix, distCoeffs)
    for imgppp in imgp_reproj1: cvh.circle(i0, imgppp.T, 2, rgb(255,0,0), thickness=-1)
    for imgppp in imgp_reproj2: cvh.circle(i0, imgppp.T, 2, rgb(0,255,255), thickness=-1)
    print "cur img check"
    cv2.imshow("img", i0)
    cv2.waitKey()    
    
    # Check whether we got a new keyframe
    base_points_remaining = base_points[all_idxs_tmp]
    is_keyframe = keyframe_test(base_points_remaining, new_points)
    print "is_keyframe:", is_keyframe
    cv2.imshow("img", cvh.drawKeypointsAndMotion(new_img, base_points_remaining, new_points, rgb(0,0,255)))
    cv2.waitKey()
    if is_keyframe:
        # First do triangulation of not-yet triangulated points (see nontriangl_idxs),
        # then do solvePnPRansac() on all already- and just- triangulated-points to eliminate outliers and only preserve union of 'inliers' and already-triangulated-points,
        # (if ratio of 'inliers' vs input is too low, reject frame (and undo rebasing of indices)) (TODO)
        # then do solvePnP() on all preserved points to refine pose estimation,
        # then re-triangulate with refined pose estimation.
        if nontriangl_idxs:
            nontriangl_idxs_array = np.array(sorted(nontriangl_idxs))
            imgp0 = base_points[nontriangl_idxs_array]
            imgp1 = idxs_get_new_points_by_idxs(nontriangl_idxs, new_points, all_idxs_tmp)
            check_triangulation_input(base_img, new_img, imgp0, imgp1, rvec_keyfr, tvec_keyfr, rvec, tvec, cameraMatrix, distCoeffs)
            
            imgpnrm0 = cv2.undistortPoints(np.array([imgp0]), cameraMatrix, distCoeffs)[0]
            imgpnrm1 = cv2.undistortPoints(np.array([imgp1]), cameraMatrix, distCoeffs)[0]
            objp_done = linear_LS_triangulation(
                    imgpnrm0.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec_keyfr), tvec_keyfr),
                    imgpnrm1.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec), tvec) )
            objp_done = objp_done.T
            print "triangl_reproj_error 0:", reprojection_error(objp_done, imgp0, rvec_keyfr, tvec_keyfr, cameraMatrix, distCoeffs)[0]
            print "triangl_reproj_error 1:", reprojection_error(objp_done, imgp1, rvec, tvec, cameraMatrix, distCoeffs)[0]
            
            objp_to_check = np.concatenate((filtered_triangl_objp.reshape(-1, 3), objp_done))
            imgp_to_check0 = np.concatenate((base_points[triangl_idxs_array], imgp0))
            imgp_to_check1 = np.concatenate((filtered_triangl_points.reshape(-1, 2), imgp1))
            rvec_, tvec_, inliers0 = cv2.solvePnPRansac(
                    objp_to_check, imgp_to_check0, cameraMatrix, distCoeffs, np.array(rvec_keyfr), np.array(tvec_keyfr), solvePnP_use_extrinsic_guess )
            print "total triangl_reproj_error 0:", reprojection_error(objp_to_check, imgp_to_check0, rvec_, tvec_, cameraMatrix, distCoeffs)[0]
            rvec_, tvec_, inliers1 = cv2.solvePnPRansac(
                    objp_to_check, imgp_to_check1, cameraMatrix, distCoeffs, np.array(rvec), np.array(tvec), solvePnP_use_extrinsic_guess )
            print "total triangl_reproj_error 1:", reprojection_error(objp_to_check, imgp_to_check1, rvec_, tvec_, cameraMatrix, distCoeffs)[0]
            
            inliers = set(inliers0.reshape(-1)) & set(inliers1.reshape(-1))
            inliers_objp_done = np.array(sorted(inliers & set(range(len(triangl_idxs), len(objp_to_check))))) - len(triangl_idxs)    # only include just-triangulated points
            inliers |= set(range(len(triangl_idxs)))    # force already-triangulated points to be reported as inliers
            inliers = np.array(sorted(inliers))
            print "# triangl outliers eliminated:", len(objp_to_check) - len(inliers), "/", len(objp_to_check)
            preserve_idxs = set(np.concatenate((triangl_idxs_array, nontriangl_idxs_array))[inliers])
            filtered_triangl_objp, filtered_triangl_points = objp_to_check[inliers], imgp_to_check1[inliers]
            ret, rvec, tvec = cv2.solvePnP(
                    filtered_triangl_objp, filtered_triangl_points, cameraMatrix, distCoeffs, rvec, tvec, solvePnP_use_extrinsic_guess )
            print "total triangl_reproj_error 1 refined:", reprojection_error(filtered_triangl_objp, filtered_triangl_points, rvec, tvec, cameraMatrix, distCoeffs)[0]
            #objp_done = objp_done[inliers_objp_done]
            
            imgpnrm0, imgpnrm1 = imgpnrm0[inliers_objp_done], imgpnrm1[inliers_objp_done]
            objp_done = linear_LS_triangulation(
                    imgpnrm0.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec_keyfr), tvec_keyfr),
                    imgpnrm1.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec), tvec) )
            objp_done = objp_done.T
            print "triangl_reproj_error 0 refined:", reprojection_error(objp_done, imgp0[inliers_objp_done], rvec_keyfr, tvec_keyfr, cameraMatrix, distCoeffs)[0]
            print "triangl_reproj_error 1 refined:", reprojection_error(objp_done, imgp1[inliers_objp_done], rvec, tvec, cameraMatrix, distCoeffs)[0]
            
            new_points = filtered_triangl_points
            triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
                    preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
            objp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs = idxs_add_objp(
                    objp_done, preserve_idxs - triangl_idxs, objp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
            #i4 = np.array(new_img)
            #objp_test = objp[imgp_to_objp_idxs[np.array(sorted(triangl_idxs))]]
            #imgp_test = idxs_get_new_points_by_idxs(triangl_idxs, new_points, all_idxs_tmp)
            ##print imgp_test - filtered_triangl_points
            #reproj_error, imgp_reproj4 = reprojection_error(objp_test, imgp_test, rvec, tvec, cameraMatrix, distCoeffs)
            #print "checking both 2", reproj_error
            #for imgppr, imgppp in zip(imgp_test, imgp_reproj4): cvh.line(i4, imgppr.T, imgppp.T, rgb(255,0,0))
            #cv2.imshow("img", i4)
            #cv2.waitKey()
        
        # Check whether we should add new 2d points
        mask_img = keypoint_mask(new_points)
        print "coverage:", 1 - cv2.countNonZero(mask_img)/float(mask_img.size), "vs min_keypoint_coverage:", min_keypoint_coverage    # unused
        to_add = target_amount_keypoints - len(new_points)
        if to_add > 0:
            print "to_add:", to_add
            ps_extra = cv2.goodFeaturesToTrack(new_img_gray, to_add, corner_quality_level, corner_min_dist, None, mask_img).reshape((-1, 2))
            cv2.imshow("img", cv2.drawKeypoints(new_img, [cv2.KeyPoint(p[0],p[1], 7.) for p in ps_extra], color=rgb(0,0,255)))
            cv2.waitKey()
            print "added:", len(ps_extra)
        else:
            ps_extra = zeros((0, 2))
            print "adding zero new points"
        base_points, new_points, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_rebase_and_add_imgp(
                ps_extra, base_points, new_points, objp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
        
        # Now this frame becomes the base (= keyframe)
        rvec_keyfr = rvec
        tvec_keyfr = tvec
        base_img = new_img
    
    return True, base_points, new_points, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, rvec, tvec, rvec_keyfr, tvec_keyfr, base_img


def main():
    global boardSize
    global cameraMatrix, distCoeffs, imageSize
    global max_OF_error, max_lost_tracks_ratio
    global keypoint_coverage_radius, min_keypoint_coverage
    global target_amount_keypoints, corner_quality_level, corner_min_dist
    global homography_condition_threshold
    global max_solvePnP_reproj_error, max_fundMat_reproj_error
    global max_solvePnP_outlier_ratio, solvePnP_use_extrinsic_guess
    
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
    max_solvePnP_reproj_error = 2.#0.5    # TODO: revert to a lower number
    max_fundMat_reproj_error = 2.0
    # solvePnp
    max_solvePnP_outlier_ratio = 0.3
    solvePnP_use_extrinsic_guess = False    # TODO: set to True and see whether the 3D results are better (reprojection seems to be worse...)
    
    
    # Select working (or 'testing') set
    from glob import glob
    images = sorted(glob(os.path.join("captures2", "*.jpeg")))
    
    imgs = []
    imgs_gray = []
    
    rvecs = []
    tvecs = []
    
    # Start frame requires special treatment
    imgs.append(cv2.imread(images[0]))
    base_img = imgs[0]
    imgs_gray.append(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY))
    ret, new_points = cvh.extractChessboardFeatures(imgs[0], boardSize)
    if not ret:
        print "First image must contain the entire chessboard!"
        return
    base_points = new_points    # 2D points
    all_idxs_tmp = np.arange(len(base_points))
    triangl_idxs = set(all_idxs_tmp)
    nontriangl_idxs = set()
    imgp_to_objp_idxs = np.array(sorted(triangl_idxs), dtype=int)
    ret, rvec, tvec = cv2.solvePnP(    # assume first frame is a proper frame with chessboard fully in-sight
            objp, new_points, cameraMatrix, distCoeffs )
    print "solvePnP reproj_error init:", reprojection_error(objp, new_points, rvec, tvec, cameraMatrix, distCoeffs)[0]
    rvecs.append(rvec)
    tvecs.append(tvec)
    rvec_keyfr = rvec
    tvec_keyfr = tvec
    
    for i in range(1, len(images)):
        # Frame[i-1] -> Frame[i]
        print "\nFrame[%s] -> Frame[%s]" % (i-1, i)
        imgs.append(cv2.imread(images[i]))
        imgs_gray.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2GRAY))
        ret, base_points, new_points, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, rvec, tvec, rvec_keyfr, tvec_keyfr, base_img = \
                handle_new_frame(base_points, new_points, imgs[-2], imgs_gray[-2], imgs[-1], imgs_gray[-1], triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, rvec_keyfr, tvec_keyfr, base_img)
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:    # frame rejected
            del imgs[-1]
            del imgs_gray[-1]


if __name__ == "__main__":
    main()
