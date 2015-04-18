#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from math import pi, ceil
import numpy as np
import glob
import cv2

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "python_libs"))
from cv2_helpers import rgb, line, circle, putText, drawKeypointsAndMotion, drawAxisSystem, drawCamera, \
                        Rodrigues, extractChessboardFeatures
import transforms as trfm
import calibration_tools
from calibration_tools import reprojection_error
import triangulation
from triangulation import iterative_LS_triangulation
triangulation.set_triangl_output_dtype(np.float32)    # solvePnPRansac() seems to dislike float64...
import color_tools
from color_tools import sample_colors
import dataset_tools

fontFace = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.3



def keypoint_mask(points):
    """Returns a mask that covers the keypoints with False, using 'keypoint_coverage_radius' as radius."""
    mask_img = np.ones((imageSize[1], imageSize[0]), dtype=np.uint8)
    for p in points:
        circle(mask_img, p, keypoint_coverage_radius, False, thickness=-1)
    # <DEBUG: visualize mask>    TODO: remove
    print "countZero:", cv2.countNonZero(mask_img), "(total):", mask_img.size
    cv2.imshow("img", mask_img*255)
    cv2.waitKey()
    # </DEBUG>
    return mask_img


def keyframe_test(points1, points2):
    """Returns True if the two images can be taken as keyframes."""
    homography, mask = cv2.findHomography(points1, points2)
    w, u, vt = cv2.SVDecomp(homography, flags=cv2.SVD_NO_UV)
    w = w.reshape((-1))
    print "w[0]/w[2]:", w[0]/w[2]
    return w[0]/w[2] > homography_condition_threshold


def check_triangulation_input(base_img, new_img,
                              imgp0, imgp1,
                              rvec_keyfr, tvec_keyfr, rvec, tvec,
                              cameraMatrix, distCoeffs):
    img = drawKeypointsAndMotion(base_img, imgp1, imgp0, rgb(0,0,255))
    drawAxisSystem(img, cameraMatrix, distCoeffs, rvec_keyfr, tvec_keyfr)
    print "base_img check"
    cv2.imshow("img", img)
    cv2.waitKey()
    
    img = drawKeypointsAndMotion(new_img, imgp0, imgp1, rgb(0,0,255))
    drawAxisSystem(img, cameraMatrix, distCoeffs, rvec, tvec)
    print "new_img check"
    cv2.imshow("img", img)
    cv2.waitKey()


class Composite2DPainter:
    
    def __init__(self, img_title, imageSize):
        self.img_title = img_title
        
        self.img = np.empty((imageSize[1], imageSize[0], 3), dtype=np.uint8)
        self.image_box = \
                [ [(             0,              0), (imageSize[0]-1,              0)], 
                  [(imageSize[0]-1,              0), (imageSize[0]-1, imageSize[1]-1)], 
                  [(imageSize[0]-1, imageSize[1]-1), (             0, imageSize[1]-1)],
                  [(             0, imageSize[1]-1), (             0,              0)] ]

    def draw(self, img, rvec, tvec, status,
            cameraMatrix, distCoeffs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, new_imgp, imgp_to_objp_idxs, objp, objp_groups, group_id, color_palette, color_palette_size):    # TODO: move these variables to another class
        """
        Draw 2D composite view.
        
        'status' can have the following values:
            0: bad frame
            1: good frame but not a keyframe
            2: good frame and also a keyframe
        """
        
        # Start drawing on copy of current image
        self.img[:, :, :] = img
        
        if status:
            # Draw world axis-system
            drawAxisSystem(self.img, cameraMatrix, distCoeffs, rvec, tvec)
            
            # Draw each already-triangulated point as a dot with text indicating depth w.r.t. the camera
            imgp = idxs_get_new_imgp_by_idxs(triangl_idxs, new_imgp, all_idxs_tmp)
            text_imgp = np.array(imgp)
            text_imgp += [(-15, 10)]    # adjust for text-position
            objp_idxs = imgp_to_objp_idxs[np.array(sorted(triangl_idxs))]
            groups = objp_groups[objp_idxs]
            P = trfm.P_from_R_and_t(Rodrigues(rvec), tvec)
            objp_depth = trfm.projection_depth(objp[objp_idxs], P)
            objp_colors = color_palette[groups % color_palette_size]    # set color by group id
            for ip, ipt, opd, color in zip(imgp, text_imgp, objp_depth, objp_colors):
                circle(self.img, ip, 2, color, thickness=-1)    # draw triangulated point
                putText(self.img, "%.3f" % opd, ipt, fontFace, fontScale, color)    # draw depths
            
            # Draw each to-be-triangulated point as a cross
            nontriangl_imgp = idxs_get_new_imgp_by_idxs(nontriangl_idxs, new_imgp, all_idxs_tmp).astype(int)
            color = color_palette[group_id % color_palette_size]
            for p in nontriangl_imgp:
                line(self.img, (p[0]-2, p[1]), (p[0]+2, p[1]), color)
                line(self.img, (p[0], p[1]-2), (p[0], p[1]+2), color)
        
        else:
            # Draw red corner around image: it's a bad frame
            for (p1, p2) in self.image_box:
                line(self.img, p1, p2, rgb(255,0,0), thickness=4)
        
        # Display image
        cv2.imshow(self.img_title, self.img)
        cv2.waitKey()

class Composite3DPainter:
        
    key_bindings = {
            "MoveLeft"      : [0x51],               # LEFT key
            "MoveRight"     : [0x53],               # RIGHT key
            "MoveUp"        : [0x52],               # UP key
            "MoveDown"      : [0x54],               # DOWN key
            "ZoomOut"       : [0x55, ord('-')],     # PAGEUP or "-" key
            "ZoomIn"        : [0x56, ord('=')],     # PAGEDOWN or "=" key
            "RotateNegZ"    : [0x50, ord('[')],     # HOME or "[" key
            "RotatePosZ"    : [0x57, ord(']')],     # END or "]" key
            "SwitchColors"  : [ord('c')],           # "c" key
            "SaveResults"   : [0x0D] }              # ENTER key
    
    def __init__(self, img_title, P_view, imageSize_view):
        self.img_title = img_title
        self.P = P_view
        
        self.img = np.empty((imageSize_view[1], imageSize_view[0], 3), dtype=np.uint8)
        self.K = np.eye(3)    # camera intrinsics matrix
        self.K[0, 0] = self.K[1, 1] = min(imageSize_view)    # set cam scaling
        self.K[0:2, 2] = np.array(imageSize_view) / 2.    # set cam principal point
        self.cams_pos = np.empty((0, 3))    # caching of cam trajectory
        self.cams_pos_keyfr = np.empty((0, 3))    # caching of cam trajectory
        self.color_mode = 1    # 0 means BGR colors, 1 means objp_group colors
        
        self.save_results_flag = False    # TODO: move this to other class
    
    def draw(self, rvec, tvec, status,
             triangl_idxs, imgp_to_objp_idxs, objp, objp_colors, objp_groups, color_palette, color_palette_size, neg_fy):    # TODO: move these variables to another class
        """
        Draw 3D composite view.
        Navigate using the following keys:
            LEFT/RIGHT    UP/DOWN    PAGEUP/PAGEDOWN    HOME/END
        
        'status' can have the following values:
            0: bad frame
            1: good frame but not a keyframe
            2: good frame and also a keyframe
        """
        
        # Calculate current camera's axis-system expressed in world coordinates and cache center
        if status:
            P_cam = trfm.P_from_rvec_and_tvec(rvec, tvec)
            cam_axes = P_cam[0:3, 0:3]
            M_cam = trfm.P_inv(P_cam)
            cam_origin = M_cam[0:3, 3:4].T
            
            self.cams_pos = np.concatenate((self.cams_pos, cam_origin))    # cache cam_origin
            if status == 2:    # frame is a keyframe
                self.cams_pos_keyfr = np.concatenate((self.cams_pos_keyfr, cam_origin))    # cache cam_origin
        
        while True:
            # Fill with dark gray background color
            self.img.fill(56)
            
            # Draw world axis system
            if trfm.projection_depth(np.array([[0,0,4]]), self.P)[0] > 0:    # only draw axis-system if its Z-axis is entirely in sight
                drawAxisSystem(self.img, self.K, None, Rodrigues(self.P[0:3, 0:3]), self.P[0:3, 3])
            
            # Draw 3D points
            objp_proj, objp_visible = trfm.project_points(objp, self.K, self.img.shape, self.P)
            objp_visible = set(np.where(objp_visible)[0])
            current_idxs = set(imgp_to_objp_idxs[np.array(tuple(triangl_idxs))]) & objp_visible
            done_idxs = np.array(tuple(objp_visible - current_idxs), dtype=int)
            current_idxs = np.array(tuple(current_idxs), dtype=int)
            if self.color_mode == 0:
                colors = objp_colors[current_idxs].astype(float)
            elif self.color_mode == 1:
                colors = color_palette[objp_groups[current_idxs] % color_palette_size]
            for opp, color in zip(objp_proj[current_idxs], colors):
                circle(self.img, opp[0:2], 4, color, thickness=-1)    # draw point, big radius
            if self.color_mode == 0:
                colors = objp_colors[done_idxs].astype(float)
            elif self.color_mode == 1:
                colors = color_palette[objp_groups[done_idxs] % color_palette_size]
            for opp, color in zip(objp_proj[done_idxs], colors):
                circle(self.img, opp[0:2], 2, color, thickness=-1)    # draw point, small radius
            
            # Draw camera trajectory
            cams_pos_proj, cams_pos_visible = trfm.project_points(self.cams_pos, self.K, self.img.shape, self.P)
            cams_pos_proj = cams_pos_proj[np.where(cams_pos_visible)[0]]
            color = rgb(0,0,0)
            for p1, p2 in zip(cams_pos_proj[:-1], cams_pos_proj[1:]):
                line(self.img, p1, p2, color, thickness=2)    # interconnect trajectory points
            cams_pos_keyfr_proj, cams_pos_keyfr_visible = trfm.project_points(self.cams_pos_keyfr, self.K, self.img.shape, self.P)
            cams_pos_keyfr_proj = cams_pos_keyfr_proj[np.where(cams_pos_keyfr_visible)[0]]
            color = rgb(255,255,255)
            for p in cams_pos_proj:
                circle(self.img, p, 1, color, thickness=-1)    # draw trajectory points
            for p in cams_pos_keyfr_proj:
                circle(self.img, p, 3, color)    # highlight keyframe trajectory points
            
            # Draw current camera axis system
            if status:
                drawCamera(self.img, cam_origin, cam_axes, self.K, self.P, neg_fy)
            else:
                last_cam_origin, last_cam_visible = trfm.project_points(self.cams_pos[-1:], self.K, self.img.shape, self.P)
                if last_cam_visible[0]:    # only draw if in sight
                    putText(self.img, '?', last_cam_origin[0] - (11, 11), fontFace, fontScale * 4, rgb(255,0,0))    # draw '?' because it's a bad frame
            
            # Display image
            cv2.imshow(self.img_title, self.img)
            
            # Handle key
            key = cv2.waitKey() & 0xFF
            if not self.handle_key(key):
                break
    
    def handle_key(self, key):
        """
        Handles keys and performs the associated action.
        Return True if a redraw is needed, otherwise False.
        """
        
        # Detect keybindings
        action = None
        for act in Composite3DPainter.key_bindings:
            if key in Composite3DPainter.key_bindings[act]:
                action = act
                break
        
        # Translate view by keyboard
        do_transform = True
        delta_t_view = np.zeros((3))
        if action in ("MoveLeft", "MoveRight"):
            delta_t_view[0] = 2 * (action == "MoveLeft") - 1    # MoveLeft -> increase cam X pos
        elif action in ("MoveUp", "MoveDown"):
            delta_t_view[1] = 2 * (action == "MoveUp") - 1    # MoveUp -> increase cam Y pos
        elif action in ("ZoomOut", "ZoomIn"):
            delta_t_view[2] = 2 * (action == "ZoomOut") - 1    # ZoomOut -> increase cam Z pos
        elif action in ("RotateNegZ", "RotatePosZ"):
            delta_z_rot = 2 * (action == "RotateNegZ") - 1    # RotateNegZ -> counter-clockwise rotate around cam Z axis
            self.P[0:3, 0:4] = Rodrigues((0, 0, delta_z_rot * pi/36)).dot(self.P[0:3, 0:4])    # by steps of 5 degrees
        else:
            do_transform = False
        if do_transform:
            self.P[0:3, 3] += delta_t_view * 0.3
            return True
        
        # Change color mode
        if action == "SwitchColors":
            self.color_mode = (self.color_mode + 1) % 2
            return True
        
        # Set flag to save results    # TODO: move this to other class
        if action == "SaveResults":
            self.save_results_flag = True
        
        return False

### Attempt to simplify re-indexing

def idxs_get_new_imgp_by_idxs(selection_idxs,
                                new_imgp, all_idxs_tmp):
    """Get selection of new_imgp corresponding with selection_idxs (of which elements correspond with idxs in base_imgp).
    type(selection_idxs) must be "set".
    """
    new_imgp_sel = np.array([p for p, idx in zip(new_imgp, all_idxs_tmp) if idx in selection_idxs])
    return new_imgp_sel

def idxs_update_by_idxs(preserve_idxs,
                        triangl_idxs, nontriangl_idxs, all_idxs_tmp):
    """Only preserve preserve_idxs (elements correspond with idxs in base_imgp).
    type(preserve_idxs) must be "set".
    To update new_imgp, first use idxs_get_new_imgp_by_idxs().
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

def idxs_add_objp(objp_extra_container, triangl_idxs_extra,
                  objp_container, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp):
    """Add new object-points objp_extra, they correspond with elements in base_imgp of which their idx is in triangl_idxs_extra.
    With:
        objp_container == (objp, objp_colors, objp_groups)
        objp_extra_container == (objp_extra, objp_colors_extra, objp_groups_extra)
    
    type(triangl_idxs_extra) must be "set".
    """
    # NOTICE: see approaches explained in idxs_update_by_idxs() to compare indexing methods
    (objp, objp_colors, objp_groups) = objp_container
    (objp_extra, objp_colors_extra, objp_groups_extra) = objp_extra_container
    imgp_to_objp_idxs[np.array(sorted(triangl_idxs_extra), dtype=int)] = np.arange(len(objp), len(objp) + len(objp_extra))
    triangl_idxs |= triangl_idxs_extra
    nontriangl_idxs -= triangl_idxs_extra
    objp = np.concatenate((objp, objp_extra))
    objp_colors = np.concatenate((objp_colors, objp_colors_extra))
    objp_groups = np.concatenate((objp_groups, objp_groups_extra))
    return (objp, objp_colors, objp_groups), imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs

def idxs_rebase_and_add_imgp(imgp_extra,
                             base_imgp, new_imgp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp):
    """Rebase all idxs to new_imgp, instead of to base_imgp; also add new image-points imgp_extra."""
    imgp_to_objp_idxs = imgp_to_objp_idxs[all_idxs_tmp]
    triangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in triangl_idxs)
    nontriangl_idxs = set(i for i, idx in enumerate(all_idxs_tmp) if idx in nontriangl_idxs)
    all_idxs_tmp = np.arange(len(all_idxs_tmp))
    
    extra_idxs = np.arange(len(new_imgp), len(new_imgp) + len(imgp_extra))
    new_imgp = np.concatenate((new_imgp, imgp_extra))
    nontriangl_idxs |= set(extra_idxs)
    imgp_to_objp_idxs = np.concatenate((imgp_to_objp_idxs, -np.ones((len(imgp_extra)), dtype=int)))    # add '-1' idxs, because not-yet-triangl
    all_idxs_tmp = np.concatenate((all_idxs_tmp, extra_idxs))
    base_imgp = new_imgp
    return base_imgp, new_imgp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp

def handle_new_frame(base_imgp,    # includes 2D points of both triangulated as not-yet triangl points of last keyframe
                     prev_imgp,    # includes 2D points of last frame
                     base_img,    # used for color extraction and debug
                     prev_img, prev_img_gray,
                     new_img, new_img_gray,
                     triangl_idxs, nontriangl_idxs,    # indices of 2D points in base_imgp
                     imgp_to_objp_idxs,    # indices from 2D points in base_imgp to 3D points in objp
                     all_idxs_tmp,    # list of idxs of 2D points in base_imgp, matches prev_imgp to base_imgp
                     objp,    # triangulated 3D points
                     objp_colors,    # BGR values of triangulated 3D points
                     objp_groups, group_id,    # corresponding group ids of triangulated 3D points, and current group id
                     rvec_keyfr, tvec_keyfr):    # rvec and tvec of last keyframe
    
    # Save initial indexing state
    triangl_idxs_old = set(triangl_idxs)
    nontriangl_idxs_old = set(nontriangl_idxs)
    all_idxs_tmp_old = np.array(all_idxs_tmp)
    
    # Calculate OF (Optical Flow), and filter outliers based on OF error
    new_imgp, status_OF, err_OF = cv2.calcOpticalFlowPyrLK(prev_img_gray, new_img_gray, prev_imgp)    # WARNING: OpenCV can output corrupted values in 'status_OF': "RuntimeWarning: invalid value encountered in less"
    new_to_prev_idxs = np.where(np.logical_and((status_OF.reshape(-1) == 1), (err_OF.reshape(-1) < max_OF_error)))[0]
    
    # If there is too much OF error in the entire image, simply reject the frame
    lost_tracks_ratio = (len(prev_imgp) - len(new_to_prev_idxs)) / float(len(prev_imgp))
    print "# points lost because of excessive OF error / # points before: ", len(prev_imgp) - len(new_to_prev_idxs), "/", len(prev_imgp), "=", lost_tracks_ratio
    if lost_tracks_ratio > max_lost_tracks_ratio:    # reject frame
        print "REJECTED: I lost track of all points!\n"
        #brisk = cv2.BRISK()#ORB()
        #prev_keyp, prev_descr = brisk.compute(prev_img_gray, [cv2.KeyPoint(p[0], p[1], keypoint_coverage_radius) for p in prev_imgp])
        #new_imgp = cv2.goodFeaturesToTrack(new_img_gray, len(prev_imgp), corner_quality_level, corner_min_dist).reshape((-1, 2))
        #new_keyp, new_descr = brisk.compute(new_img_gray, [cv2.KeyPoint(p[0], p[1], keypoint_coverage_radius) for p in new_imgp])
        ##FLANN_INDEX_KDTREE = 1    # BUG: this enum is missing in the Python OpenCV binding
        ##index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #FLANN_INDEX_LSH = 6
        #index_params= dict(algorithm = FLANN_INDEX_LSH,
                   #table_number = 6, # 12
                   #key_size = 12,     # 20
                   #multi_probe_level = 1) #2
        #search_params = dict(checks=50)
        #flann = cv2.FlannBasedMatcher(index_params, search_params)
        #matches = flann.knnMatch(prev_descr, new_descr, k=2)
        ## store all the good matches as per Lowe's ratio test.
        #print [i for i in matches]
        #kp1, kp2 = zip(*[(prev_keyp[m.queryIdx].pt, new_keyp[m.trainIdx].pt) for m, n in matches if m.distance < 0.7*n.distance])
        ##flann = cv2.flann_Index(new_descr, index_params)
        ##idx2, dist = flann.knnSearch(prev_descr, 2, params = {}) # bug: need to provide empty dict
        ##mask = dist[:,0] / dist[:,1] < 0.6
        ##idx1 = np.arange(len(prev_descr))
        ##matches = np.int32( zip(idx1, idx2[:,0]) )[mask]
        ##kp1, kp2 = zip(*[(prev_keyp[m.queryIdx].pt, new_keyp[m.trainIdx].pt) for m, n in matches])
        #print "Trying to recover from lost tracks, using flann matcher..."
        #while True:
            #cv2.imgshow("img", drawKeypointsAndMotion(new_img, kp1, kp2, rgb(0,0,255)))
            #cv2.waitKey()
        return False, base_imgp, prev_imgp, base_img, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_colors, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr
    
    # Save matches by idxs
    preserve_idxs = set(all_idxs_tmp[new_to_prev_idxs])
    triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
            preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
    if len(triangl_idxs) < 8:    # solvePnP uses 8-point algorithm
        print "REJECTED: I lost track of too many already-triangulated points, so we can't do solvePnP() anymore...\n"
        return False, base_imgp, prev_imgp, base_img, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_colors, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr
    new_imgp = new_imgp[new_to_prev_idxs]
    #cv2.cornerSubPix(    # TODO: activate this secret weapon    <-- hmm, actually seems to make it worse
                #new_img_gray, new_imgp,
                #(corner_min_dist,corner_min_dist),    # window
                #(-1,-1),    # deadzone
                #(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )    # termination criteria
        ##corners = corners.reshape(-1, 2)
    
    # Do solvePnPRansac() on current frame's (new_imgp) already triangulated points, ...
    triangl_idxs_array = np.array(sorted(triangl_idxs))    # select already-triangulated point-indices
    filtered_triangl_imgp = idxs_get_new_imgp_by_idxs(triangl_idxs, new_imgp, all_idxs_tmp)    # collect corresponding image-points
    filtered_triangl_objp = objp[imgp_to_objp_idxs[triangl_idxs_array]]    # collect corresponding object-points
    print "Doing solvePnP() on", filtered_triangl_objp.shape[0], "points"
    rvec_, tvec_, inliers = cv2.solvePnPRansac(    # perform solvePnPRansac() to identify outliers, force to obey max_solvePnP_outlier_ratio
            filtered_triangl_objp, filtered_triangl_imgp, cameraMatrix, distCoeffs, minInliersCount=int(ceil((1 - max_solvePnP_outlier_ratio) * len(triangl_idxs))), reprojectionError=max_solvePnP_reproj_error )
    
    # ... if ratio of 'inliers' vs input is too low, reject frame, ...
    if inliers == None:    # inliers is empty => reject frame
        print "REJECTED: No inliers based on solvePnP()!\n"
        return False, base_imgp, prev_imgp, base_img, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_colors, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr
    inliers = inliers.reshape(-1)
    solvePnP_outlier_ratio = (len(triangl_idxs) - len(inliers)) / float(len(triangl_idxs))
    print "solvePnP_outlier_ratio:", solvePnP_outlier_ratio
    if solvePnP_outlier_ratio > max_solvePnP_outlier_ratio or len(inliers) < 8:    # reject frame
        if solvePnP_outlier_ratio > max_solvePnP_outlier_ratio:
            print "REJECTED: Not enough inliers (ratio) based on solvePnP()!\n"
        else:
            print "REJECTED: Not enough inliers (absolute) based on solvePnP() to perform (non-RANSAC) solvePnP()!\n"
        return False, base_imgp, prev_imgp, base_img, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_colors, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr
    
    # <DEBUG: visualize reprojection error>    TODO: remove
    reproj_error, imgp_reproj1 = reprojection_error(filtered_triangl_objp, filtered_triangl_imgp, cameraMatrix, distCoeffs, rvec_, tvec_)
    print "solvePnP reproj_error:", reproj_error
    i3 = np.array(new_img)
    try:
        for imgppr, imgppp in zip(filtered_triangl_imgp, imgp_reproj1): line(i3, imgppr.T, imgppp.T, rgb(255,0,0))
    except OverflowError: print "WARNING: OverflowError!"
    cv2.imshow("img", i3)
    cv2.waitKey()
    # </DEBUG>
    
    # ... then do solvePnP() on inliers, to get the current frame's pose estimation, ...
    triangl_idxs_array = triangl_idxs_array[inliers]    # select inliers among all already-triangulated point-indices
    filtered_triangl_imgp, filtered_triangl_objp = filtered_triangl_imgp[inliers], filtered_triangl_objp[inliers]
    preserve_idxs = set(triangl_idxs_array) | nontriangl_idxs
    new_imgp = idxs_get_new_imgp_by_idxs(preserve_idxs, new_imgp, all_idxs_tmp)
    triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(    # update indices to only preserve inliers
            preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
    ret, rvec, tvec = cv2.solvePnP(    # perform solvePnP() to estimate the pose
            filtered_triangl_objp, filtered_triangl_imgp, cameraMatrix, distCoeffs, rvec_, tvec_, useExtrinsicGuess=True )
    
    # .. finally do a check on the average reprojection error, and reject frame if too high.
    reproj_error, imgp_reproj = reprojection_error(filtered_triangl_objp, filtered_triangl_imgp, cameraMatrix, distCoeffs, rvec, tvec)
    print "solvePnP refined reproj_error:", reproj_error
    if reproj_error > max_solvePnP_reproj_error:    # reject frame
        print "REJECTED: Too high reprojection error based on pose estimate of solvePnP()!\n"
        return False, base_imgp, prev_imgp, base_img, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_colors, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr
    
    # <DEBUG: verify poses by reprojection error>    TODO: remove
    i0 = drawAxisSystem(np.array(new_img), cameraMatrix, distCoeffs, rvec, tvec)
    try:
        for imgppp in imgp_reproj1: circle(i0, imgppp.T, 2, rgb(255,0,0), thickness=-1)
    except OverflowError: print "WARNING: OverflowError!"
    for imgppp in imgp_reproj : circle(i0, imgppp.T, 2, rgb(0,255,255), thickness=-1)
    print "cur img check"
    cv2.imshow("img", i0)
    cv2.waitKey()
    # </DEBUG>
    
    # <DEBUG: verify OpticalFlow motion on preserved inliers>    TODO: remove
    cv2.imshow("img", drawKeypointsAndMotion(new_img, base_imgp[all_idxs_tmp], new_imgp, rgb(0,0,255)))
    cv2.waitKey()
    # </DEBUG>
    
    # Check whether we got a new keyframe
    is_keyframe = keyframe_test(base_imgp[all_idxs_tmp], new_imgp)
    print "is_keyframe:", is_keyframe
    if is_keyframe:
        # If some points are not yet triangulated, do it now:
        if nontriangl_idxs:
            
            # First do triangulation of not-yet triangulated points using initial pose estimation, ...
            nontriangl_idxs_array = np.array(sorted(nontriangl_idxs))    # select not-yet-triangulated point-indices
            imgp0 = base_imgp[nontriangl_idxs_array]    # collect corresponding image-points of last keyframe
            imgp1 = idxs_get_new_imgp_by_idxs(nontriangl_idxs, new_imgp, all_idxs_tmp)    # collect corresponding image-points of current frame
            # <DEBUG: check sanity of input to triangulation function>    TODO: remove
            check_triangulation_input(base_img, new_img, imgp0, imgp1, rvec_keyfr, tvec_keyfr, rvec, tvec, cameraMatrix, distCoeffs)
            # </DEBUG>
            imgpnrm0 = cv2.undistortPoints(np.array([imgp0]), cameraMatrix, distCoeffs)[0]    # undistort and normalize to homogenous coordinates
            imgpnrm1 = cv2.undistortPoints(np.array([imgp1]), cameraMatrix, distCoeffs)[0]
            objp_done, objp_done_status = iterative_LS_triangulation(    # triangulate
                    imgpnrm0, trfm.P_from_R_and_t(Rodrigues(rvec_keyfr), tvec_keyfr),    # data from last keyframe
                    imgpnrm1, trfm.P_from_R_and_t(Rodrigues(rvec), tvec) )               # data from current frame
            print "objp_done_status:", objp_done_status
            inliers_objp_done = np.where(objp_done_status == 1)[0]
            
            # <DEBUG: check reprojection error of the new freshly triangulated points, based on both pose estimates of keyframe and current cam>    TODO: remove
            print "triangl_reproj_error 0:", reprojection_error(objp_done, imgp0, cameraMatrix, distCoeffs, rvec_keyfr, tvec_keyfr)[0]
            print "triangl_reproj_error 1:", reprojection_error(objp_done, imgp1, cameraMatrix, distCoeffs, rvec, tvec)[0]
            # </DEBUG>
            
            # ... filter out outliers based on Iterative-LS triangulation convergence, and whether points are in front of all cameras, ...
            objp_done = objp_done[inliers_objp_done]
            imgp1 = imgp1[inliers_objp_done]
            imgpnrm0 = imgpnrm0[inliers_objp_done]
            imgpnrm1 = imgpnrm1[inliers_objp_done]
            filtered_triangl_objp_tmp = np.concatenate((filtered_triangl_objp, objp_done))    # collect all desired object-points
            filtered_triangl_imgp_tmp = np.concatenate((filtered_triangl_imgp, imgp1))    # collect corresponding image-points of current frame
            nontriangl_idxs_array = nontriangl_idxs_array[inliers_objp_done]
            
            # ... then do solvePnP() on all preserved points ('inliers') to refine pose estimation, ...
            ret, rvec, tvec = cv2.solvePnP(    # perform solvePnP(), we start from the initial pose estimation
                    filtered_triangl_objp_tmp, filtered_triangl_imgp_tmp, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess=True )
            print "total triangl_reproj_error 1 refined:", reprojection_error(filtered_triangl_objp_tmp, filtered_triangl_imgp_tmp, cameraMatrix, distCoeffs, rvec, tvec)[0]    # TODO: remove
            
            # ... then do re-triangulation of 'inliers_objp_done' using refined pose estimation.
            objp_done, objp_done_status = iterative_LS_triangulation(    # triangulate
                    imgpnrm0, trfm.P_from_R_and_t(Rodrigues(rvec_keyfr), tvec_keyfr),    # data from last keyframe
                    imgpnrm1, trfm.P_from_R_and_t(Rodrigues(rvec), tvec) )               # data from current frame
            print "objp_done_status refined:", objp_done_status
            
            ## Uncomment this to re-filter triangulation output, otherwise ...
            #inliers_objp_done = np.where(objp_done_status == 1)[0]
            #objp_done = objp_done[inliers_objp_done]
            #imgp1 = imgp1[inliers_objp_done]
            #imgpnrm0 = imgpnrm0[inliers_objp_done]
            #imgpnrm1 = imgpnrm1[inliers_objp_done]
            ##filtered_triangl_objp = np.concatenate((filtered_triangl_objp, objp_done))    # collect all desired object-points
            #filtered_triangl_imgp = np.concatenate((filtered_triangl_imgp, imgp1))    # collect corresponding image-points of current frame
            #nontriangl_idxs_array = nontriangl_idxs_array[inliers_objp_done]
            
            # ... uncomment this.
            filtered_triangl_imgp = filtered_triangl_imgp_tmp
            
            # Preserve all good indices
            preserve_idxs = triangl_idxs | set(nontriangl_idxs_array)
            
            # <DEBUG: check reprojection error of the new freshly (refined) triangulated points, based on both pose estimates of keyframe and current cam>    TODO: remove
            if len(inliers_objp_done):
                imgp0 = imgp0[inliers_objp_done]
                print "triangl_reproj_error 0 refined:", reprojection_error(objp_done, imgp0, cameraMatrix, distCoeffs, rvec_keyfr, tvec_keyfr)[0]
                print "triangl_reproj_error 1 refined:", reprojection_error(objp_done, imgp1, cameraMatrix, distCoeffs, rvec, tvec)[0]
            # </DEBUG>
            
            # Update image-points and indices, and store the newly triangulated object-points and assign them the current group id
            new_imgp = filtered_triangl_imgp
            triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
                    preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
            objp_colors_done = sample_colors(base_img, base_imgp[nontriangl_idxs_array])    # use colors of base-image, they don't have OF drift
            objp_groups_done = np.empty((len(objp_done)), dtype=int); objp_groups_done.fill(group_id)    # assign to current 'group_id'
            (objp, objp_colors, objp_groups), imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs = idxs_add_objp(
                    (objp_done, objp_colors_done, objp_groups_done), preserve_idxs - triangl_idxs, (objp, objp_colors, objp_groups), imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
            
            ## <DEBUG: check intermediate outlier filtering>    TODO: remove
            #i4 = np.array(new_img)
            #objp_test = objp[imgp_to_objp_idxs[np.array(sorted(triangl_idxs))]]
            #imgp_test = idxs_get_new_imgp_by_idxs(triangl_idxs, new_imgp, all_idxs_tmp)
            ##print imgp_test - filtered_triangl_imgp
            #reproj_error, imgp_reproj4 = reprojection_error(objp_test, imgp_test, cameraMatrix, distCoeffs, rvec, tvec)
            #print "checking both 2", reproj_error
            #for imgppr, imgppp in zip(imgp_test, imgp_reproj4): line(i4, imgppr.T, imgppp.T, rgb(255,0,0))
            #cv2.imshow("img", i4)
            #cv2.waitKey()
            ## </DEBUG>
        
        # Check whether we should add new image-points
        mask_img = keypoint_mask(new_imgp)    # generate mask that covers all image-points (with a certain radius)
        print "coverage:", 1 - cv2.countNonZero(mask_img)/float(mask_img.size)    # TODO: remove: unused
        to_add = target_amount_keypoints - len(new_imgp)    # limit the amount of to-be-added image-points
        
        # Add new image-points
        if to_add > 0:
            print "to_add:", to_add
            imgp_extra = cv2.goodFeaturesToTrack(new_img_gray, to_add, corner_quality_level, corner_min_dist, None, mask_img).reshape((-1, 2))
            print "added:", len(imgp_extra)
            group_id += 1    # create a new group to assign the new batch of points to, later on
        else:
            imgp_extra = zeros((0, 2))
            print "adding zero new points"
        base_imgp, new_imgp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_rebase_and_add_imgp(    # update indices to include new image-points
                imgp_extra, base_imgp, new_imgp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
        
        # <DEBUG: visualize newly added points>    TODO: remove
        cv2.imshow("img", cv2.drawKeypoints(new_img, [cv2.KeyPoint(p[0],p[1], 7.) for p in imgp_extra], color=rgb(0,0,255)))
        cv2.waitKey()
        # </DEBUG>
        
        # Now this frame becomes the base (= keyframe)
        rvec_keyfr = rvec
        tvec_keyfr = tvec
        base_img = new_img
    
    # Successfully return
    return True + int(is_keyframe), base_imgp, new_imgp, base_img, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, objp_colors, objp_groups, group_id, rvec, tvec, rvec_keyfr, tvec_keyfr


def parse_cmd_args():
    import argparse
    
    def join_path(path_list):
        """Convenience function for creating OS-indep relative paths."""
        return os.path.relpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *path_list))
    
    # Class to generate an example command-line usage-message
    class ExampleUsage:
        
        def __init__(self, img_dir, calib_file,
                     init_chessboard_size=None, init_files=None, traj_out_file=None, map_out_file=None):
            """
            "init_chessboard_size" : a (init_chessboard_size_x, init_chessboard_size_y) tuple,
                                     or None if inferred from init-files, see next
            "init_files" : a (init_points_file (.pcd), init_pose_file (.txt)) tuple,
                           or None if inferred from chessboard
            """
            self.img_dir, self.calib_file, self.init_chessboard_size, self.init_files, self.traj_out_file, self.map_out_file = \
                    img_dir, calib_file, init_chessboard_size, init_files, traj_out_file, map_out_file
        
        def generate(self):
            out = "\t %s %s %s " % (sys.argv[0], join_path(self.img_dir), join_path(self.calib_file))
            if self.init_chessboard_size:
                out += "-sx %s -sy %s " % tuple(self.init_chessboard_size)
            if self.init_files:
                out += "-o %s -p %s " % tuple(map(join_path, self.init_files))
            if self.traj_out_file:
                out += "-t %s " % join_path(self.traj_out_file)
            if self.map_out_file:
                out += "-m %s " % join_path(self.map_out_file)
            return out
    
    # Define examples
    example_usages = []
    example_usages.append(ExampleUsage(    # example of using handshot C170 webcam footage
            ("..", "..", "datasets", "webcam", "captures2"),
            ("..", "..", "datasets", "webcam", "camera_intrinsics.txt"),
            init_chessboard_size=(8, 6) ))
    example_usages.append(ExampleUsage(    # example of using flying ARDrone2 footage
            ("..", "..", "..", "ARDrone2_tests", "flying_front", "lowres", "drone0"),
            ("..", "..", "..", "ARDrone2_tests", "camera_calibration", "live_video", "camera_intrinsics_front.txt"),
            init_chessboard_size=(8, 6) ))
    example_usages.append(ExampleUsage(    # example of using the ICL_NUIM living-room dataset (4th trajectory)
            ("..", "..", "datasets", "ICL_NUIM", "living_room_traj3n_frei_png", "rgb"),
            ("..", "..", "datasets", "ICL_NUIM", "camera_intrinsics.txt"),
            init_files=(
                    ("..", "..", "datasets", "ICL_NUIM", "living_room_traj3n_frei_png", "init_points.pcd"),
                    ("..", "..", "datasets", "ICL_NUIM", "living_room_traj3n_frei_png", "init_pose.txt") ),
            traj_out_file=
            ("..", "..", "datasets", "ICL_NUIM", "living_room_traj3n_frei_png", "traj_out-slam2.txt"),
            map_out_file=
            ("..", "..", "datasets", "ICL_NUIM", "living_room_traj3n_frei_png", "map_out-slam2.pcd") ))
    
    # Create parser object and help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
            description=
            "SLAM algorithm. \n\n" +
            "Example usages: \n" +
            "%s \n" % '\n'.join(map(ExampleUsage.generate, example_usages)) )
    
    parser.add_argument("img_dir",
                        help="path to the directory of the input images")
    parser.add_argument("calib_file",
                        help="path to the camera intrinsics calibration file")
    
    parser.add_argument("-sx", "--init-chessboard-size-x", dest="init_chessboard_size_x",
                        type=int,
                        help="X resolution of the chessboard in the initial frame")
    parser.add_argument("-sy", "--init-chessboard-size-y", dest="init_chessboard_size_y",
                        type=int,
                        help="Y resolution of the chessboard in the initial frame")
    
    parser.add_argument("-o", "--init-objp-file", dest="init_objp_file",
                        help="path to the PCD-file (pointcloud) containing the initial 3D points")
    parser.add_argument("-p", "--init-pose-file", dest="init_pose_file",
                        help="path to the ASCII file containing the 4x4 matrix of the initial camera pose")
    
    parser.add_argument("-t", "--traj-out-file", dest="traj_out_file",
                        help="filepath of the output camera trajectory, in TUM format")
    parser.add_argument("-m", "--map-out-file", dest="map_out_file",
                        help="filepath of the output 3D map, in PCD format (pointcloud)")
    
    # Parse arguments
    args = parser.parse_args()
    img_dir, calib_file, init_chessboard_size_x, init_chessboard_size_y, init_objp_file, init_pose_file, traj_out_file, map_out_file = \
            args.img_dir, args.calib_file, args.init_chessboard_size_x, args.init_chessboard_size_y, args.init_objp_file, args.init_pose_file, args.traj_out_file, args.map_out_file
    
    # A chessboard will be used in this case
    if (init_chessboard_size_x or init_chessboard_size_y):
        if (init_chessboard_size_x and not init_chessboard_size_y) or \
                (not init_chessboard_size_x and init_chessboard_size_y):
            raise AttributeError("The --init-chessboard-size-x and --init-chessboard-size-y arguments "
                                "should always be used together.")
        if not (2 <= init_chessboard_size_x < 128) or \
                not (2 <= init_chessboard_size_y < 128):
            raise AttributeError("The --init-chessboard-size-x and --init-chessboard-size-y arguments "
                                 "should be in the range from 2 to 128.")
        init_chessboard_size = (init_chessboard_size_x, init_chessboard_size_y)
        init_files = None
    
    # Otherwise the initialization files will be used
    else:
        if not (init_objp_file and init_pose_file):
            raise AttributeError("Both --init-objp-file and --init-pose-file arguments "
                                 "should be used if no --init-chessboard-size arguments are given.")
        init_chessboard_size = None
        init_files = (init_objp_file, init_pose_file)
    
    return img_dir, calib_file, init_chessboard_size, init_files, traj_out_file, map_out_file


def main():
    global cameraMatrix, distCoeffs, imageSize
    global max_OF_error, max_lost_tracks_ratio
    global keypoint_coverage_radius#, min_keypoint_coverage
    global target_amount_keypoints, corner_quality_level, corner_min_dist
    global homography_condition_threshold
    global max_solvePnP_reproj_error, max_2nd_solvePnP_reproj_error, max_fundMat_reproj_error
    global max_solvePnP_outlier_ratio, max_2nd_solvePnP_outlier_ratio, solvePnP_use_extrinsic_guess
    
    # Parse command-line arguments
    img_dir, calib_file, init_chessboard_size, init_files, traj_out_file, map_out_file = parse_cmd_args()
    
    # Load camera intrinsics
    cameraMatrix, distCoeffs, imageSize = calibration_tools.load_camera_intrinsics(calib_file)
    neg_fy = (cameraMatrix[1, 1] < 0)
    
    # Select working (or 'testing') set
    images = [image for image in os.listdir(img_dir)
              if os.path.splitext(image)[1] in (".png", ".jpg", ".jpeg", ".tiff")]
    # Correctly sort "2.png" vs "10.png" by adding leading zeros until filename-lengths are equal
    max_img_filename_length = max(map(len, images))
    images.sort(key=(lambda img_file: '0' * (max_img_filename_length - len(img_file)) + img_file))
    images = [os.path.join(img_dir, image) for image in images]
    
    # Load pre-defined initialization points, needed for datasets without chessboard in the beginning
    if not init_chessboard_size:
        init_objp_file, init_pose_file = init_files
        P_init = np.loadtxt(init_pose_file)
        predef_objp, _, _ = dataset_tools.load_3D_points_from_pcd_file(init_objp_file)
        predef_imgp, predef_imgp_visible = trfm.project_points(
                predef_objp, cameraMatrix, [imageSize[1], imageSize[0]], P_init, round=False )    # keep high accuracy, no rounding
        predef_imgp = predef_imgp[np.where(predef_imgp_visible)[0]]
    
    
    # Create color palette, used to identify 3D point group ids
    color_palette, color_palette_size = color_tools.color_palette(2, 3, 4)
    
    # Setup some visualization helpers
    composite2D_painter = Composite2DPainter("composite 2D", imageSize)
    composite3D_painter = Composite3DPainter(
            "composite 3D", trfm.P_from_R_and_t(Rodrigues((pi, 0., 0.)), np.array([[0., 0., 40.]]).T), (1280, 720) )
    
    
    ### Tweaking parameters ###
    # OF calculation
    max_OF_error = 12.
    max_lost_tracks_ratio = 0.5
    # keypoint_coverage
    keypoint_coverage_radius = int(max_OF_error)
    #min_keypoint_coverage = 0.2
    # goodFeaturesToTrack
    target_amount_keypoints = int(round((imageSize[0] * imageSize[1]) / (pi * keypoint_coverage_radius**2)))    # target is entire image full
    print "target_amount_keypoints:", target_amount_keypoints
    corner_quality_level = 0.01
    corner_min_dist = keypoint_coverage_radius
    # keyframe_test
    homography_condition_threshold = 500    # defined as ratio between max and min singular values
    # reprojection error
    max_solvePnP_reproj_error = 2.#0.5    # TODO: revert to a lower number
    max_2nd_solvePnP_reproj_error = max_solvePnP_reproj_error / 2    # be more strict in a 2nd iteration, used after 1st pass of triangulation
    max_fundMat_reproj_error = 2.0
    # solvePnP
    max_solvePnP_outlier_ratio = 0.33
    max_2nd_solvePnP_outlier_ratio = 1.    # used in 2nd iteration, after 1st pass of triangulation
    solvePnP_use_extrinsic_guess = False    # TODO: set to True and see whether the 3D results are better
    
    
    # Init
    
    imgs = []
    imgs_gray = []
    
    objp = []    # 3D points
    objp_colors = []    # 3D point BGR color, measured at pixel of first frame of the newly added point
    objp_groups = []    # 3D point group ids, each new batch of detected points is put in a separate group
    group_id = 0    # current 3D point group id
    
    rvecs, rvecs_keyfr = [], []
    tvecs, tvecs_keyfr = [], []
    
    
    # Start frame requires special treatment
    
    # Start frame : read image and detect 2D points ...
    imgs.append(cv2.imread(images[0]))
    base_img = imgs[0]
    imgs_gray.append(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY))
    
    # ... in case of chessboard
    if init_chessboard_size:
        ret, new_imgp = extractChessboardFeatures(imgs[0], init_chessboard_size)
        if not ret:
            print "First image must contain the entire chessboard!"
            return
    
    # ... in case of pre-defined points
    else:
        new_imgp = predef_imgp.astype(np.float32)
    
    # Start frame : define a priori 3D points ...
    objp_colors = sample_colors(imgs[0], new_imgp)
    objp_groups = np.zeros(len(new_imgp), dtype=np.int)
    group_id += 1
    
    # ... in case of chessboard
    if init_chessboard_size:
        objp = calibration_tools.grid_objp(init_chessboard_size)
    
    # ... in case of pre-defined points
    else:
        objp = np.array(predef_objp)
    
    # Start frame : setup linking data-structures
    base_imgp = new_imgp    # 2D points
    all_idxs_tmp = np.arange(len(base_imgp))
    triangl_idxs = set(all_idxs_tmp)
    nontriangl_idxs = set()
    imgp_to_objp_idxs = np.array(sorted(triangl_idxs), dtype=int)
    
    # Start frame : get absolute pose
    ret, rvec, tvec = cv2.solvePnP(    # assume first frame is a proper frame with chessboard fully in-sight
            objp, new_imgp, cameraMatrix, distCoeffs )
    print "solvePnP reproj_error init:", reprojection_error(objp, new_imgp, cameraMatrix, distCoeffs, rvec, tvec)[0]
    rvecs.append(rvec)
    tvecs.append(tvec)
    rvec_keyfr = rvec
    tvec_keyfr = tvec
    rvecs_keyfr.append(rvec_keyfr)
    tvecs_keyfr.append(tvec_keyfr)
    
    # Start frame : add other points
    mask_img = keypoint_mask(new_imgp)
    to_add = target_amount_keypoints - len(new_imgp)
    imgp_extra = cv2.goodFeaturesToTrack(imgs_gray[0], to_add, corner_quality_level, corner_min_dist, None, mask_img).reshape((-1, 2))
    cv2.imshow("img", cv2.drawKeypoints(imgs[0], [cv2.KeyPoint(p[0],p[1], 7.) for p in imgp_extra], color=rgb(0,0,255)))
    cv2.waitKey()
    print "added:", len(imgp_extra)
    base_imgp, new_imgp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_rebase_and_add_imgp(
            imgp_extra, base_imgp, new_imgp, imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
    ret = 2    # indicate keyframe
        
    # Draw 3D points info of current frame
    print "Drawing composite 2D image"
    composite2D_painter.draw(imgs[0], rvec, tvec, ret,
                             cameraMatrix, distCoeffs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, new_imgp, imgp_to_objp_idxs, objp, objp_groups, group_id, color_palette, color_palette_size)
    
    # Draw 3D points info of all frames
    print "Drawing composite 3D image    (keys: LEFT/RIGHT/UP/DOWN/PAGEUP/PAGEDOWN/HOME/END)"
    #print "                             (  or:   A    D    W   S     -       =      [   ] )"
    print "                              (  or:                       -       =      [   ] )"
    composite3D_painter.draw(rvec, tvec, ret,
                             triangl_idxs, imgp_to_objp_idxs, objp, objp_colors, objp_groups, color_palette, color_palette_size, neg_fy)
    
    for i in range(1, len(images)):
        # Frame[i-1] -> Frame[i]
        print "\nFrame[%s] -> Frame[%s]" % (i-1, i)
        print "    processing '", images[i], "':"
        cur_img = cv2.imread(images[i])
        imgs.append(cur_img)
        imgs_gray.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2GRAY))
        ret, base_imgp, new_imgp, base_img, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, objp_colors, objp_groups, group_id, rvec, tvec, rvec_keyfr, tvec_keyfr = \
                handle_new_frame(base_imgp, new_imgp, base_img, imgs[-2], imgs_gray[-2], imgs[-1], imgs_gray[-1], triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, objp_colors, objp_groups, group_id, rvec_keyfr, tvec_keyfr)
        
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
            if ret == 2:    # frame is a keyframe
                rvecs_keyfr.append(rvec_keyfr)
                tvecs_keyfr.append(tvec_keyfr)
        else:    # frame rejected
            rvecs.append(None)
            tvecs.append(None)
            del imgs[-1]
            del imgs_gray[-1]
        
        # Draw 3D points info of current frame
        print "Drawing composite 2D image"
        composite2D_painter.draw(cur_img, rvec, tvec, ret,
                                cameraMatrix, distCoeffs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, new_imgp, imgp_to_objp_idxs, objp, objp_groups, group_id, color_palette, color_palette_size)
        
        # Draw 3D points info of all frames
        print "Drawing composite 3D image    (view keys: LEFT/RIGHT/UP/DOWN/PAGEUP/PAGEDOWN/HOME/END/C)"
        #print "                             (       or:   A    D    W   S     -       =      [   ]   )"
        print "                              (       or:                       -       =      [   ]   )"
        print "                              (take snapshot of results:           ENTER               )"
        composite3D_painter.draw(rvec, tvec, ret,
                                triangl_idxs, imgp_to_objp_idxs, objp, objp_colors, objp_groups, color_palette, color_palette_size, neg_fy)
        
        if composite3D_painter.save_results_flag or not (i % 30):    # save results once every 30 frames
            composite3D_painter.save_results_flag = False
            
            # Save trajectory
            if traj_out_file:
                print "Saving trajectory..."
                Ps = []
                for rvec, tvec in zip(rvecs, tvecs):
                    if rvec == None == tvec:
                        Ps.append(None)    # bad frame
                    else:
                        Ps.append(trfm.P_from_rvec_and_tvec(rvec, tvec))
                timestps, locations, quaternions = dataset_tools.convert_cam_poses_to_cam_trajectory_TUM(Ps)
                dataset_tools.save_cam_trajectory_TUM(traj_out_file, timestps, locations, quaternions)
                print "Done."
            
            # Save map
            if map_out_file:
                print "Saving map pointcloud..."
                
                ## Visualize lifetime by group-number, ...
                #max_lifetime = float(np.max(objp_groups))
                #if max_lifetime > 0:
                    #objp_group_lifetime = objp_groups.reshape(len(objp_groups), 1) / max_lifetime
                #else:
                    #objp_group_lifetime = np.ones((len(objp_groups), 1))
                
                # ... or visualize lifetime by 1 if currently triangulated point, 0 otherwise
                objp_group_lifetime = np.zeros((len(objp_groups), 1))
                objp_group_lifetime[imgp_to_objp_idxs[np.array(tuple(triangl_idxs))]] = 1
                
                # Export colors of the selected color-mode
                if composite3D_painter.color_mode == 0:
                    colors = objp_colors
                elif composite3D_painter.color_mode == 1:
                    colors = color_palette[objp_groups % color_palette_size][:, 0:3]
                
                # Save pointcloud with BGR colors + lifetime as alpha channel
                dataset_tools.save_3D_points_to_pcd_file(
                        map_out_file, objp, np.concatenate((
                        colors,
                        255 * (0.3 + 0.7 * objp_group_lifetime) ), axis=1) )    # lifetime as alpha
                
                print "Done."


if __name__ == "__main__":
    main()
