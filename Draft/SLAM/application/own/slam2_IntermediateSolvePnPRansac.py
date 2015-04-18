#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from math import pi, ceil
import numpy as np
import glob
import cv2

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "python_libs"))
import cv2_helpers as cvh
from cv2_helpers import rgb
import transforms as trfm

fontFace = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.3



def prepare_color_palette(nc_L, nc_a, nc_b):
    """
    Prepare color palette, colors are randomly distributed.
    
    nc_L, nc_a, nc_b give the number of colors in each dimension of Lab color-space.
    Returns the color palette and the total amount of colors.
    """
    L_min, L_max = 99, 230
    a_min, a_max = 26, 230
    b_min, b_max = 26, 230
    
    num_colors = nc_L * nc_a * nc_b
    colors = np.zeros((num_colors, 1, 3), dtype=np.uint8)
    for Li, L in enumerate(np.arange(L_min, L_max + 1, (L_max-L_min) / (nc_L-1))):
        for ai, a in enumerate(np.arange(a_min, a_max + 1, (a_max-a_min) / (nc_a-1))):
            for bi, b in enumerate(np.arange(b_min, b_max + 1, (b_max-b_min) / (nc_b-1))):
                colors[Li*nc_a*nc_b + ai*nc_b + bi, 0, :] = (L, a, b)
    
    color_palette = cv2.cvtColor(colors, cv2.COLOR_LAB2RGB).reshape(num_colors, 3)
    np.random.seed(1)
    color_palette = np.random.permutation(color_palette)
    np.random.seed()
    
    return np.array(map(rgb, *zip(*color_palette))), num_colors


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
    imgp_reproj, jacob = cv2.projectPoints(
            axis_system_objp, rvec, tvec, cameraMatrix, distCoeffs )
    origin, xAxis, yAxis, zAxis = np.rint(imgp_reproj.reshape(-1, 2)).astype(np.int32)    # round to nearest int
    if not (0 <= origin[0] < img.shape[1] and 0 <= origin[1] < img.shape[0]):    # projected origin lies out of the image
        return img    # so don't draw axis-system
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
            draw_axis_system(self.img, rvec, tvec, cameraMatrix, distCoeffs)
            
            # Draw to-be-triangulated point as dots with text indicating depth w.r.t. the camera
            imgp = idxs_get_new_imgp_by_idxs(triangl_idxs, new_imgp, all_idxs_tmp)
            text_imgp = np.array(imgp)
            text_imgp += [(-15, 10)]    # adjust for text-position
            objp_idxs = imgp_to_objp_idxs[np.array(sorted(triangl_idxs))]
            groups = objp_groups[objp_idxs]
            P = trfm.P_from_R_and_t(cvh.Rodrigues(rvec), tvec)
            objp_depth = trfm.projection_depth(objp[objp_idxs], P)
            objp_colors = color_palette[groups % color_palette_size]    # set color by group id
            for ip, ipt, opd, opg, color in zip(imgp, text_imgp, objp_depth, groups, objp_colors):
                cvh.circle(self.img, ip, 2, color, thickness=-1)    # draw triangulated point
                cvh.putText(self.img, "%.3f" % opd, ipt, fontFace, fontScale, color)    # draw depths
            
            # Draw to-be-triangulated point as a cross
            nontriangl_imgp = idxs_get_new_imgp_by_idxs(nontriangl_idxs, new_imgp, all_idxs_tmp).astype(int)
            color = color_palette[group_id % color_palette_size]
            for p in nontriangl_imgp:
                cvh.line(self.img, (p[0]-2, p[1]), (p[0]+2, p[1]), color)
                cvh.line(self.img, (p[0], p[1]-2), (p[0], p[1]+2), color)
        
        else:
            # Draw red corner around image: it's a bad frame
            for (p1, p2) in self.image_box:
                cvh.line(self.img, p1, p2, rgb(255,0,0), thickness=4)
        
        # Display image
        cv2.imshow(self.img_title, self.img)
        cv2.waitKey()

class Composite3DPainter:
    
    def __init__(self, img_title, P_view, imageSize_view):
        self.img_title = img_title
        self.P = P_view
        
        self.img = np.empty((imageSize_view[1], imageSize_view[0], 3), dtype=np.uint8)
        self.K = np.eye(3)    # camera intrinsics matrix
        self.K[0, 0] = self.K[1, 1] = min(imageSize_view)    # set cam scaling
        self.K[0:2, 2] = np.array(imageSize_view) / 2.    # set cam principal point
        self.cams_pos = np.empty((0, 3))    # caching of cam trajectory
        self.cams_pos_keyfr = np.empty((0, 3))    # caching of cam trajectory
    
    def draw(self, rvec, tvec, status,
             triangl_idxs, imgp_to_objp_idxs, objp, objp_groups, color_palette, color_palette_size):    # TODO: move these variables to another class
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
            R_cam = cvh.Rodrigues(rvec)
            cam_axissys_objp = np.empty((4, 3))
            cam_axissys_objp[0, :] = -R_cam.T.dot(tvec).reshape(1, 3)    # cam_origin
            cam_axissys_objp[1:4, :] = cam_axissys_objp[0, :] + R_cam    # cam_x, cam_y, cam_z
            self.cams_pos = np.concatenate((self.cams_pos, cam_axissys_objp[0:1, :]))    # cache cam_origin
            if status == 2:    # frame is a keyframe
                self.cams_pos_keyfr = np.concatenate((self.cams_pos_keyfr, cam_axissys_objp[0:1, :]))    # cache cam_origin
        
        while True:
            # Fill with dark gray background color
            self.img.fill(56)
            
            # Draw world axis system
            if trfm.projection_depth(np.array([[0,0,4]]), self.P)[0] > 0:    # only draw axis-system if its Z-axis is entirely in sight
                draw_axis_system(self.img, cvh.Rodrigues(self.P[0:3, 0:3]), self.P[0:3, 3], self.K, None)
            
            # Draw 3D points
            objp_proj, objp_visible = trfm.project_points(objp, self.K, self.img.shape, self.P)
            objp_visible = set(np.where(objp_visible)[0])
            current_idxs = set(imgp_to_objp_idxs[np.array(tuple(triangl_idxs))]) & objp_visible
            done_idxs = np.array(tuple(objp_visible - current_idxs), dtype=int)
            current_idxs = np.array(tuple(current_idxs), dtype=int)
            groups = objp_groups[current_idxs]
            for opp, opg, color in zip(objp_proj[current_idxs], groups, color_palette[groups % color_palette_size]):
                cvh.circle(self.img, opp[0:2], 4, color, thickness=-1)    # draw point, big radius
            groups = objp_groups[done_idxs]
            for opp, opg, color in zip(objp_proj[done_idxs], groups, color_palette[groups % color_palette_size]):
                cvh.circle(self.img, opp[0:2], 2, color, thickness=-1)    # draw point, small radius
            
            # Draw camera trajectory
            cams_pos_proj, cams_pos_visible = trfm.project_points(self.cams_pos, self.K, self.img.shape, self.P)
            cams_pos_proj = cams_pos_proj[np.where(cams_pos_visible)[0]]
            color = rgb(0,0,0)
            for p1, p2 in zip(cams_pos_proj[:-1], cams_pos_proj[1:]):
                cvh.line(self.img, p1, p2, color, thickness=2)    # interconnect trajectory points
            cams_pos_keyfr_proj, cams_pos_keyfr_visible = trfm.project_points(self.cams_pos_keyfr, self.K, self.img.shape, self.P)
            cams_pos_keyfr_proj = cams_pos_keyfr_proj[np.where(cams_pos_keyfr_visible)[0]]
            color = rgb(255,255,255)
            for p in cams_pos_proj:
                cvh.circle(self.img, p, 1, color, thickness=-1)    # draw trajectory points
            for p in cams_pos_keyfr_proj:
                cvh.circle(self.img, p, 3, color)    # highlight keyframe trajectory points
            
            # Draw current camera axis system
            if status:
                (cam_origin, cam_xAxis, cam_yAxis, cam_zAxis), cam_visible = \
                        trfm.project_points(cam_axissys_objp, self.K, self.img.shape, self.P)
                if cam_visible.sum() == len(cam_visible):    # only draw axis-system if it's entirely in sight
                    cvh.line(self.img, cam_origin, cam_xAxis, rgb(255,0,0), lineType=cv2.CV_AA)
                    cvh.line(self.img, cam_origin, cam_yAxis, rgb(0,255,0), lineType=cv2.CV_AA)
                    cvh.line(self.img, cam_origin, cam_zAxis, rgb(0,0,255), lineType=cv2.CV_AA)
                    cvh.circle(self.img, cam_zAxis, 3, rgb(0,0,255))    # small dot to highlight cam Z axis
            else:
                last_cam_origin, last_cam_visible = trfm.project_points(self.cams_pos[-1:], self.K, self.img.shape, self.P)
                if last_cam_visible[0]:    # only draw if in sight
                    cvh.putText(self.img, '?', last_cam_origin[0] - (11, 11), fontFace, fontScale * 4, rgb(255,0,0))    # draw '?' because it's a bad frame
            
            # Display image
            cv2.imshow(self.img_title, self.img)
            
            # Translate view by keyboard
            key = cv2.waitKey() & 0xFF
            delta_t_view = np.zeros((3))
            if key in (0x51, 0x53):    # LEFT/RIGHT key
                delta_t_view[0] = 2 * (key == 0x51) - 1    # LEFT -> increase cam X pos
            elif key in (0x52, 0x54):    # UP/DOWN key
                delta_t_view[1] = 2 * (key == 0x52) - 1    # UP -> increase cam Y pos
            elif key in (0x55, 0x56):    # PAGEUP/PAGEDOWN key
                delta_t_view[2] = 2 * (key == 0x55) - 1    # PAGEUP -> increase cam Z pos
            elif key in (0x50, 0x57):    # HOME/END key
                delta_z_rot = 2 * (key == 0x50) - 1    # HOME -> counter-clockwise rotate around cam Z axis
                self.P[0:3, 0:4] = cvh.Rodrigues((0, 0, delta_z_rot * pi/36)).dot(self.P[0:3, 0:4])    # by steps of 5 degrees
            else:
                break
            self.P[0:3, 3] += delta_t_view * 0.3

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
        objp_container == (objp, objp_groups)
        objp_extra_container == (objp_extra, objp_groups_extra)
    
    type(triangl_idxs_extra) must be "set".
    """
    # NOTICE: see approaches explained in idxs_update_by_idxs() to compare indexing methods
    (objp, objp_groups) = objp_container
    (objp_extra, objp_groups_extra) = objp_extra_container
    imgp_to_objp_idxs[np.array(sorted(triangl_idxs_extra), dtype=int)] = np.arange(len(objp), len(objp) + len(objp_extra))
    triangl_idxs |= triangl_idxs_extra
    nontriangl_idxs -= triangl_idxs_extra
    objp = np.concatenate((objp, objp_extra))
    objp_groups = np.concatenate((objp_groups, objp_groups_extra))
    return (objp, objp_groups), imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs

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
                     prev_img, prev_img_gray,
                     new_img, new_img_gray,
                     triangl_idxs, nontriangl_idxs,    # indices of 2D points in base_imgp
                     imgp_to_objp_idxs,    # indices from 2D points in base_imgp to 3D points in objp
                     all_idxs_tmp,    # list of idxs of 2D points in base_imgp, matches prev_imgp to base_imgp
                     objp,    # triangulated 3D points
                     objp_groups, group_id,    # corresponding group ids of triangulated 3D points, and current group id
                     rvec_keyfr, tvec_keyfr,    # rvec and tvec of last keyframe
                     base_img):    # used for debug
    
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
        return False, base_imgp, prev_imgp, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr, base_img
    
    # Save matches by idxs
    preserve_idxs = set(all_idxs_tmp[new_to_prev_idxs])
    triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
            preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
    if len(triangl_idxs) < 8:    # solvePnP uses 8-point algorithm
        print "REJECTED: I lost track of too many already-triangulated points, so we can't do solvePnP() anymore...\n"
        return False, base_imgp, prev_imgp, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr, base_img
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
        return False, base_imgp, prev_imgp, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr, base_img
    inliers = inliers.reshape(-1)
    solvePnP_outlier_ratio = (len(triangl_idxs) - len(inliers)) / float(len(triangl_idxs))
    print "solvePnP_outlier_ratio:", solvePnP_outlier_ratio
    if solvePnP_outlier_ratio > max_solvePnP_outlier_ratio or len(inliers) < 8:    # reject frame
        if solvePnP_outlier_ratio > max_solvePnP_outlier_ratio:
            print "REJECTED: Not enough inliers (ratio) based on solvePnP()!\n"
        else:
            print "REJECTED: Not enough inliers (absolute) based on solvePnP() to perform (non-RANSAC) solvePnP()!\n"
        return False, base_imgp, prev_imgp, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr, base_img
    
    # <DEBUG: visualize reprojection error>    TODO: remove
    reproj_error, imgp_reproj1 = reprojection_error(filtered_triangl_objp, filtered_triangl_imgp, rvec_, tvec_, cameraMatrix, distCoeffs)
    print "solvePnP reproj_error:", reproj_error
    i3 = np.array(new_img)
    try:
        for imgppr, imgppp in zip(filtered_triangl_imgp, imgp_reproj1): cvh.line(i3, imgppr.T, imgppp.T, rgb(255,0,0))
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
            filtered_triangl_objp, filtered_triangl_imgp, cameraMatrix, distCoeffs )
    
    # .. finally do a check on the average reprojection error, and reject frame if too high.
    reproj_error, imgp_reproj = reprojection_error(filtered_triangl_objp, filtered_triangl_imgp, rvec, tvec, cameraMatrix, distCoeffs)
    print "solvePnP refined reproj_error:", reproj_error
    if reproj_error > max_solvePnP_reproj_error:    # reject frame
        print "REJECTED: Too high reprojection error based on pose estimate of solvePnP()!\n"
        return False, base_imgp, prev_imgp, triangl_idxs_old, nontriangl_idxs_old, imgp_to_objp_idxs, all_idxs_tmp_old, objp, objp_groups, group_id, None, None, rvec_keyfr, tvec_keyfr, base_img
    
    # <DEBUG: verify poses by reprojection error>    TODO: remove
    i0 = draw_axis_system(np.array(new_img), rvec, tvec, cameraMatrix, distCoeffs)
    try:
        for imgppp in imgp_reproj1: cvh.circle(i0, imgppp.T, 2, rgb(255,0,0), thickness=-1)
    except OverflowError: print "WARNING: OverflowError!"
    for imgppp in imgp_reproj : cvh.circle(i0, imgppp.T, 2, rgb(0,255,255), thickness=-1)
    print "cur img check"
    cv2.imshow("img", i0)
    cv2.waitKey()
    # </DEBUG>
    
    # <DEBUG: verify OpticalFlow motion on preserved inliers>    TODO: remove
    cv2.imshow("img", cvh.drawKeypointsAndMotion(new_img, base_imgp[all_idxs_tmp], new_imgp, rgb(0,0,255)))
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
            objp_done = linear_LS_triangulation(    # triangulate
                    imgpnrm0.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec_keyfr), tvec_keyfr),    # data from last keyframe
                    imgpnrm1.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec), tvec) )               # data from current frame
            objp_done = objp_done.T
            
            # <DEBUG: check reprojection error of the new freshly triangulated points, based on both pose estimates of keyframe and current cam>    TODO: remove
            print "triangl_reproj_error 0:", reprojection_error(objp_done, imgp0, rvec_keyfr, tvec_keyfr, cameraMatrix, distCoeffs)[0]
            print "triangl_reproj_error 1:", reprojection_error(objp_done, imgp1, rvec, tvec, cameraMatrix, distCoeffs)[0]
            # </DEBUG>
            
            filtered_triangl_objp = np.concatenate((filtered_triangl_objp, objp_done))    # collect all desired object-points
            filtered_triangl_imgp = np.concatenate((filtered_triangl_imgp, imgp1))    # collect corresponding image-points of current frame
            
            # ... then do solvePnP() on all preserved points ('inliers') to refine pose estimation, ...
            ret, rvec, tvec = cv2.solvePnP(    # perform solvePnP(), we start from the initial pose estimation
                    filtered_triangl_objp, filtered_triangl_imgp, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess=True )
            print "total triangl_reproj_error 1 refined:", reprojection_error(filtered_triangl_objp, filtered_triangl_imgp, rvec, tvec, cameraMatrix, distCoeffs)[0]    # TODO: remove
            
            # ... then do re-triangulation of 'inliers_objp_done' using refined pose estimation.
            objp_done = linear_LS_triangulation(    # triangulate
                    imgpnrm0.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec_keyfr), tvec_keyfr),    # data from last keyframe
                    imgpnrm1.T, trfm.P_from_R_and_t(cvh.Rodrigues(rvec), tvec) )               # data from current frame
            objp_done = objp_done.T
            
            # <DEBUG: check reprojection error of the new freshly (refined) triangulated points, based on both pose estimates of keyframe and current cam>    TODO: remove
            print "triangl_reproj_error 0 refined:", reprojection_error(objp_done, imgp0, rvec_keyfr, tvec_keyfr, cameraMatrix, distCoeffs)[0]
            print "triangl_reproj_error 1 refined:", reprojection_error(objp_done, imgp1, rvec, tvec, cameraMatrix, distCoeffs)[0]
            # </DEBUG>
            
            # Update image-points and indices, and store the newly triangulated object-points and assign them the current group id
            new_imgp = filtered_triangl_imgp
            triangl_idxs, nontriangl_idxs, all_idxs_tmp = idxs_update_by_idxs(
                    preserve_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
            objp_groups_done = np.empty((len(objp_done)), dtype=int); objp_groups_done.fill(group_id)    # assign to current 'group_id'
            (objp, objp_groups), imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs = idxs_add_objp(
                    (objp_done, objp_groups_done), preserve_idxs - triangl_idxs, (objp, objp_groups), imgp_to_objp_idxs, triangl_idxs, nontriangl_idxs, all_idxs_tmp )
            
            ## <DEBUG: check intermediate outlier filtering>    TODO: remove
            #i4 = np.array(new_img)
            #objp_test = objp[imgp_to_objp_idxs[np.array(sorted(triangl_idxs))]]
            #imgp_test = idxs_get_new_imgp_by_idxs(triangl_idxs, new_imgp, all_idxs_tmp)
            ##print imgp_test - filtered_triangl_imgp
            #reproj_error, imgp_reproj4 = reprojection_error(objp_test, imgp_test, rvec, tvec, cameraMatrix, distCoeffs)
            #print "checking both 2", reproj_error
            #for imgppr, imgppp in zip(imgp_test, imgp_reproj4): cvh.line(i4, imgppr.T, imgppp.T, rgb(255,0,0))
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
    return True + int(is_keyframe), base_imgp, new_imgp, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, objp_groups, group_id, rvec, tvec, rvec_keyfr, tvec_keyfr, base_img


def main():
    global boardSize
    global cameraMatrix, distCoeffs, imageSize
    global max_OF_error, max_lost_tracks_ratio
    global keypoint_coverage_radius#, min_keypoint_coverage
    global target_amount_keypoints, corner_quality_level, corner_min_dist
    global homography_condition_threshold
    global max_solvePnP_reproj_error, max_2nd_solvePnP_reproj_error, max_fundMat_reproj_error
    global max_solvePnP_outlier_ratio, max_2nd_solvePnP_outlier_ratio, solvePnP_use_extrinsic_guess
    
    # Initially known data
    boardSize = (8, 6)
    color_palette, color_palette_size = prepare_color_palette(2, 3, 4)    # used to identify 3D point group ids
    
    cameraMatrix, distCoeffs, imageSize = \
            load_camera_intrinsics(os.path.join("..", "..", "datasets", "webcam", "camera_intrinsics.txt"))
    #cameraMatrix, distCoeffs, imageSize = \
            #load_camera_intrinsics(os.path.join("..", "..", "..", "ARDrone2_tests", "camera_calibration", "live_video", "camera_intrinsics_front.txt"))
    
    # Select working (or 'testing') set
    from glob import glob
    images = sorted(glob(os.path.join("..", "..", "datasets", "webcam", "captures2", "*.jpeg")))
    #images = sorted(glob(os.path.join("..", "..", "..", "ARDrone2_tests", "flying_front", "lowres", "drone0", "*.jpg")))
    
    # Setup some visualization helpers
    composite2D_painter = Composite2DPainter("composite 2D", imageSize)
    composite3D_painter = Composite3DPainter(
            "composite 3D", trfm.P_from_R_and_t(cvh.Rodrigues((pi, 0., 0.)), np.array([[0., 0., 40.]]).T), (1280, 720) )
    
    
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
    max_solvePnP_reproj_error = 4.#0.5    # TODO: revert to a lower number
    max_2nd_solvePnP_reproj_error = max_solvePnP_reproj_error / 2    # be more strict in a 2nd iteration, used after 1st pass of triangulation
    max_fundMat_reproj_error = 2.0
    # solvePnP
    max_solvePnP_outlier_ratio = 0.3
    max_2nd_solvePnP_outlier_ratio = 1.    # used in 2nd iteration, after 1st pass of triangulation
    solvePnP_use_extrinsic_guess = False    # TODO: set to True and see whether the 3D results are better
    
    
    # Init
    
    imgs = []
    imgs_gray = []
    
    objp = []    # 3D points
    objp_groups = []    # 3D point group ids, each new batch of detected points is put in a separate group
    group_id = 0    # current 3D point group id
    
    rvecs, rvecs_keyfr = [], []
    tvecs, tvecs_keyfr = [], []
    
    
    # Start frame requires special treatment
    
    # Start frame : read image and detect 2D points
    imgs.append(cv2.imread(images[0]))
    base_img = imgs[0]
    imgs_gray.append(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY))
    ret, new_imgp = cvh.extractChessboardFeatures(imgs[0], boardSize)
    if not ret:
        print "First image must contain the entire chessboard!"
        return
    
    # Start frame : define a priori 3D points
    objp = prepare_object_points(boardSize)
    objp_groups = np.zeros((objp.shape[0]), dtype=np.int)
    group_id += 1
    
    # Start frame : setup linking data-structures
    base_imgp = new_imgp    # 2D points
    all_idxs_tmp = np.arange(len(base_imgp))
    triangl_idxs = set(all_idxs_tmp)
    nontriangl_idxs = set()
    imgp_to_objp_idxs = np.array(sorted(triangl_idxs), dtype=int)
    
    # Start frame : get absolute pose
    ret, rvec, tvec = cv2.solvePnP(    # assume first frame is a proper frame with chessboard fully in-sight
            objp, new_imgp, cameraMatrix, distCoeffs )
    print "solvePnP reproj_error init:", reprojection_error(objp, new_imgp, rvec, tvec, cameraMatrix, distCoeffs)[0]
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
    composite3D_painter.draw(rvec, tvec, ret,
                             triangl_idxs, imgp_to_objp_idxs, objp, objp_groups, color_palette, color_palette_size)
    
    for i in range(1, len(images)):
        # Frame[i-1] -> Frame[i]
        print "\nFrame[%s] -> Frame[%s]" % (i-1, i)
        print "    processing '", images[i], "':"
        cur_img = cv2.imread(images[i])
        imgs.append(cur_img)
        imgs_gray.append(cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2GRAY))
        ret, base_imgp, new_imgp, triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, objp_groups, group_id, rvec, tvec, rvec_keyfr, tvec_keyfr, base_img = \
                handle_new_frame(base_imgp, new_imgp, imgs[-2], imgs_gray[-2], imgs[-1], imgs_gray[-1], triangl_idxs, nontriangl_idxs, imgp_to_objp_idxs, all_idxs_tmp, objp, objp_groups, group_id, rvec_keyfr, tvec_keyfr, base_img)
        
        if ret:
            rvecs.append(rvec)
            tvecs.append(tvec)
            if ret == 2:    # frame is a keyframe
                rvecs_keyfr.append(rvec_keyfr)
                tvecs_keyfr.append(tvec_keyfr)
        else:    # frame rejected
            del imgs[-1]
            del imgs_gray[-1]
        
        # Draw 3D points info of current frame
        print "Drawing composite 2D image"
        composite2D_painter.draw(cur_img, rvec, tvec, ret,
                                 cameraMatrix, distCoeffs, triangl_idxs, nontriangl_idxs, all_idxs_tmp, new_imgp, imgp_to_objp_idxs, objp, objp_groups, group_id, color_palette, color_palette_size)
        
        # Draw 3D points info of all frames
        print "Drawing composite 3D image    (keys: LEFT/RIGHT/UP/DOWN/PAGEUP/PAGEDOWN)"
        composite3D_painter.draw(rvec, tvec, ret,
                                 triangl_idxs, imgp_to_objp_idxs, objp, objp_groups, color_palette, color_palette_size)


if __name__ == "__main__":
    main()
