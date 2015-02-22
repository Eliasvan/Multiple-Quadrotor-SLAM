#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Code originates from:
        http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    
    View demo video at http://www.youtube.com/watch?v=SX2qodUfDaA
"""
import os
from math import degrees, pi
import numpy as np
import cv2

import sys; sys.path.append("../../PythonLibraries")
import transforms as trfm
import cv2_helpers as cvh
from cv2_helpers import rgb, format3DVector



def prepare_object_points(boardSize):
    """
    Prepare object points, like (0,0,0), (0,1,0), (0,2,0) ... ,(5,7,0).
    boardSize[0] is used as Y-axis, boardSize[1] as X-axis.
    """
    objp = np.zeros((np.prod(boardSize), 3), np.float32)
    objp[:,:] = np.array([ map(float, [i, j, 0])
                            for i in range(boardSize[1])
                            for j in range(boardSize[0]) ])
    
    return objp


def calibrate_camera_interactive(images, objp, boardSize):
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
            cv2.imshow("img", img)
            cv2.waitKey(100)

    # Calibration
    reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints, imagePoints, imageSize )
    distCoeffs = distCoeffs.reshape((-1))    # convert to vector
    
    return reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs, \
            objectPoints, imagePoints, imageSize


def save_camera_intrinsics(filename, cameraMatrix, distCoeffs, imageSize):
    out = """\
    # cameraMatrix, distCoeffs, imageSize =
    
    %s, \\
    \\
    %s, \\
    \\
    %s
    """
    from textwrap import dedent
    out = dedent(out) % (repr(cameraMatrix), repr(distCoeffs), repr(imageSize))
    open(filename, 'w').write(out)

def load_camera_intrinsics(filename):
    from numpy import array
    cameraMatrix, distCoeffs, imageSize = \
            eval(open(filename, 'r').read())
    return cameraMatrix, distCoeffs, imageSize


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
    
    return img_undistorted, roi


def reprojection_error(cameraMatrix, distCoeffs, rvecs, tvecs, objectPoints, imagePoints):
    mean_error = np.zeros((1, 2))
    square_error = np.zeros((1, 2))
    n_images = len(imagePoints)

    for i in xrange(n_images):
        imgp_reproj, jacob = cv2.projectPoints(
                objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs )
        error = imgp_reproj.reshape(-1, 2) - imagePoints[i]
        mean_error += abs(error).sum(axis=0) / len(imagePoints[i])
        square_error += (error**2).sum(axis=0) / len(imagePoints[i])

    mean_error = cv2.norm(mean_error / n_images)
    square_error = np.sqrt(square_error.sum() / n_images)
    
    return mean_error, square_error


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)
iterative_LS_triangulation_tolerance = 1.e-6

def iterative_LS_triangulation(u, P, u1, P1):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    
    (u, P) is the reference pair containing homogenous image coordinates (x, y) and the corresponding camera matrix.
    (u1, P1) is the second pair.
    
    Additionally returns a status-vector to indicate outliers:
        True:  inlier
        False: outlier
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
    
    u and u1 are matrices: amount of points equals #rows and should be equal for u and u1.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.empty((4, len(u))); x[3, :].fill(1)    # create empty array of homogenous 3D coordinates
    x_status = np.zeros(len(u), dtype=int)    # default: mark every point as an outlier
    
    # Initialize C matrices
    C = np.array(iterative_LS_triangulation_C)
    C1 = np.array(iterative_LS_triangulation_C)
    
    for xi in range(len(u)):
        # Build C matrices, to visualize calculation structure of A and b
        C[:, 2] = u[xi, :]
        C1[:, 2] = u1[xi, :]
        
        # Build A matrix
        A[0:2, :] = C.dot(P[0:3, 0:3])     # C * R
        A[2:4, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        
        # Build b vector
        b[0:2, :] = C.dot(P[0:3, 3:4])    # C * t
        b[2:4, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b *= -1
        
        # Init depths
        d = d1 = 1.
        
        for i in range(10):    # Hartley suggests 10 iterations at most
            # Solve for x vector
            cv2.solve(A, b, x[0:3, xi:xi+1], cv2.DECOMP_SVD)
            
            # Calculate new depths
            d_new = P[2, :].dot(x[:, xi])
            d1_new = P1[2, :].dot(x[:, xi])
            
            # Convergence criterium
            print i, d_new - d, d1_new - d1, (d_new > 0 and d1_new > 0)
            if abs(d_new - d) <= iterative_LS_triangulation_tolerance and \
                    abs(d1_new - d1) <= iterative_LS_triangulation_tolerance:
                x_status[xi] = (d_new > 0 and d1_new > 0)    # points should be in front of both cameras
                if d_new <= 0: x_status[xi] -= 1    # TODO: remove
                if d1_new <= 0: x_status[xi] -= 2    # TODO: remove
                break
            
            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d_new
            A[2:4, :] *= 1 / d1_new
            b[0:2, :] *= 1 / d_new
            b[2:4, :] *= 1 / d1_new
            
            # Update depths
            d = d_new
            d1 = d1_new
    
    return x[0:3, :].T, x_status

# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)

def linear_LS_triangulation(u, P, u1, P1):
    """
    Linear Least Squares based triangulation.
    
    (u, P) is the reference pair containing homogenous image coordinates (x, y) and the corresponding camera matrix.
    (u1, P1) is the second pair.
    
    u and u1 are matrices: amount of points equals #rows and should be equal for u and u1.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u)))
    
    # Initialize C matrices
    C = np.array(linear_LS_triangulation_C)
    C1 = np.array(linear_LS_triangulation_C)
    
    for i in range(len(u)):
        # Build C matrices, to visualize calculation structure of A and b
        C[:, 2] = u[i, :]
        C1[:, 2] = u1[i, :]
        
        # Build A matrix
        A[0:2, :] = C.dot(P[0:3, 0:3])    # C * R
        A[2:4, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        
        # Build b vector
        b[0:2, :] = C.dot(P[0:3, 3:4])    # C * t
        b[2:4, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return x.T

def triangl_pose_est_interactive(img_left, img_right, cameraMatrix, distCoeffs, imageSize, objp, boardSize, nonplanar_left, nonplanar_right):
    """ (TODO: remove debug-prints)
    Triangulation and relative pose estimation will be performed from LEFT to RIGHT image.
    
    Both images have to contain the whole chessboard,
    in order to make a decent estimation of the relative pose based on 'solvePnP'.
    
    Then the user can manually create matches between non-planar objects.
    If the user omits this step, only triangulation of the chessboard corners will be performed,
    and they will be compared to the real 3D points.
    Otherwise the coordinates of the triangulated points of the manually matched points will be printed,
    and a relative pose estimation will be performed (using the essential matrix),
    this pose estimation will be compared with the decent 'solvePnP' estimation.
    In that case you will be asked whether you want to mute the chessboard corners in all calculations,
    note that this requires at least 8 pairs of matches corresponding with non-planar geometry.
    
    During manual matching process,
    switching between LEFT and RIGHT image will be done in a zigzag fashion.
    To stop selecting matches, press SPACE.
    
    Additionally, you can also introduce predefined non-planar geometry,
    this is e.g. useful if you don't want to select non-planar geometry manually.
    """
    
    ### Setup initial state, and calc some very accurate poses to compare against with later
    
    K_left = cameraMatrix
    K_right = cameraMatrix
    K_inv_left = cvh.invert(K_left)
    K_inv_right = cvh.invert(K_right)
    distCoeffs_left = distCoeffs
    distCoeffs_right = distCoeffs
    
    # Extract chessboard features, together they form a set of 'planar' points
    ret_left, planar_left = cvh.extractChessboardFeatures(img_left, boardSize)
    ret_right, planar_right = cvh.extractChessboardFeatures(img_right, boardSize)
    if not ret_left or not ret_right:
        print "Chessboard is not (entirely) in sight, aborting."
        return
    
    # Save exact 2D and 3D points of the chessboard
    num_planar = planar_left.shape[0]
    planar_left_orig, planar_right_orig = planar_left, planar_right
    objp_orig = objp
    
    # Calculate P matrix of left pose, in reality this one is known
    ret, rvec_left, tvec_left = cv2.solvePnP(
            objp, planar_left, K_left, distCoeffs_left )
    P_left = trfm.P_from_R_and_t(cvh.Rodrigues(rvec_left), tvec_left)
    
    # Calculate P matrix of right pose, in reality this one is unknown
    ret, rvec_right, tvec_right = cv2.solvePnP(
            objp, planar_right, K_right, distCoeffs_right )
    P_right = trfm.P_from_R_and_t(cvh.Rodrigues(rvec_right), tvec_right)
    
    # Load previous non-planar inliers, if desired
    if nonplanar_left.size and not raw_input("Type 'no' if you don't want to use previous non-planar inliers? ").strip().lower() == "no":
        nonplanar_left = map(tuple, nonplanar_left)
        nonplanar_right = map(tuple, nonplanar_left)
    else:
        nonplanar_left = []
        nonplanar_right = []
    
    
    ### Create predefined non-planar 3D geometry, to enable automatic selection of non-planar points
    
    if raw_input("Type 'yes' if you want to create and select predefined non-planar geometry? ").strip().lower() == "yes":
        
        # A cube
        cube_coords = np.array([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
                                (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.)])
        cube_coords *= 2    # scale
        cube_edges = np.array([(0, 1), (1, 2), (2, 3), (3, 0),
                               (4, 5), (5, 6), (6, 7), (7, 4),
                               (0, 4), (1, 5), (2, 6), (3, 7)])
        
        # An 8-point circle
        s2 = 1. / np.sqrt(2)
        circle_coords = np.array([( 1.,  0., 0.), ( s2,  s2, 0.),
                                  ( 0.,  1., 0.), (-s2,  s2, 0.),
                                  (-1.,  0., 0.), (-s2, -s2, 0.),
                                  ( 0., -1., 0.), ( s2, -s2, 0.)])
        circle_edges = np.array([(i, (i+1) % 8) for i in range(8)])
        
        # Position 2 cubes and 2 circles in the scene
        cube1 = np.array(cube_coords)
        cube1[:, 1] -= 1
        cube2 = np.array(cube_coords)
        cube2 = cvh.Rodrigues((0., 0., pi/4)).dot(cube2.T).T
        cube2[:, 0] += 4
        cube2[:, 1] += 3
        cube2[:, 2] += 2
        circle1 = np.array(circle_coords)
        circle1 *= 2
        circle1[:, 1] += 5
        circle1[:, 2] += 2
        circle2 = np.array(circle_coords)
        circle2 = cvh.Rodrigues((pi/2, 0., 0.)).dot(circle2.T).T
        circle2[:, 1] += 5
        circle2[:, 2] += 2
        
        # Print output to be used in Blender
        print
        print "Cubes"
        print "edges_cube = \\\n", map(list, cube_edges)
        print "coords_cube1 = \\\n", map(list, cube1)
        print "coords_cube2 = \\\n", map(list, cube2)
        print
        print "Circles"
        print "edges_circle = \\\n", map(list, circle_edges)
        print "coords_circle1 = \\\n", map(list, circle1)
        print "coords_circle2 = \\\n", map(list, circle2)
        print
        
        color = rgb(0, 200, 150)
        for verts, edges in zip([cube1, cube2, circle1, circle2],
                                [cube_edges, cube_edges, circle_edges, circle_edges]):
            out_left = cvh.wireframe3DGeometry(
                    img_left, verts, edges, color, rvec_left, tvec_left, K_left, distCoeffs_left )
            out_right = cvh.wireframe3DGeometry(
                    img_right, verts, edges, color, rvec_right, tvec_right, K_right, distCoeffs_right )
            valid_match_idxs = [i for i, (pl, pr) in enumerate(zip(out_left, out_right))
                                    if 0 <= min(pl[0], pr[0]) <= max(pl[0], pr[0]) < imageSize[0] and 
                                        0 <= min(pl[1], pr[1]) <= max(pl[1], pr[1]) < imageSize[1]
                                ]
            nonplanar_left += map(tuple, out_left[valid_match_idxs])    # concatenate
            nonplanar_right += map(tuple, out_right[valid_match_idxs])    # concatenate
        
        nonplanar_left = np.rint(nonplanar_left).astype(int)
        nonplanar_right = np.rint(nonplanar_right).astype(int)
    
    
    ### User can manually create matches between non-planar objects
    
    class ManualMatcher:
        def __init__(self, window_name, images, points):
            self.window_name = window_name
            self.images = images
            self.points = points
            self.img_idx = 0    # 0: left; 1: right
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.onMouse)
        
        def onMouse(self, event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points[self.img_idx].append((x, y))
            
            elif event == cv2.EVENT_LBUTTONUP:
                # Switch images in a ping-pong fashion
                if len(self.points[0]) != len(self.points[1]):
                    self.img_idx = 1 - self.img_idx
        
        def run(self):
            print "Select your matches. Press SPACE when done."
            while True:
                img = cv2.drawKeypoints(self.images[self.img_idx], [cv2.KeyPoint(p[0],p[1], 7.) for p in self.points[self.img_idx]], color=rgb(0,0,255))
                cv2.imshow(self.window_name, img)
                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '): break    # finish if SPACE is pressed
            
            num_points_diff = len(self.points[0]) - len(self.points[1])
            if num_points_diff:
                del self.points[num_points_diff < 0][-abs(num_points_diff):]
            print "Selected", len(self.points[0]), "pairs of matches."
    
    # Execute the manual matching
    ManualMatcher("Select match-points of non-planar objects", [img_left, img_right], [nonplanar_left, nonplanar_right]).run()
    num_nonplanar = len(nonplanar_left)
    has_nonplanar = (num_nonplanar > 0)
    if 0 < num_nonplanar < 8:
        print "Warning: you've selected less than 8 pairs of matches."
    nonplanar_left = np.array(nonplanar_left).reshape(num_nonplanar, 2)
    nonplanar_right = np.array(nonplanar_right).reshape(num_nonplanar, 2)
    
    mute_chessboard_corners = False
    if num_nonplanar >= 8 and raw_input("Type 'yes' if you want to exclude chessboard corners from calculations? ").strip().lower() == "yes":
        mute_chessboard_corners = True
        num_planar = 0
        planar_left = np.zeros((0, 2))
        planar_right = np.zeros((0, 2))
        objp = np.zeros((0, 3))
        print "Chessboard corners muted."
    
    has_prev_triangl_points = not mute_chessboard_corners    # normally in this example, this should always be True, unless user forces False
    
    
    ### Undistort points => normalized coordinates
    
    allfeatures_left = np.concatenate((planar_left, nonplanar_left))
    allfeatures_right = np.concatenate((planar_right, nonplanar_right))
    
    allfeatures_nrm_left = cv2.undistortPoints(np.array([allfeatures_left]), K_left, distCoeffs_left)[0]
    allfeatures_nrm_right = cv2.undistortPoints(np.array([allfeatures_right]), K_right, distCoeffs_right)[0]
    
    planar_nrm_left, nonplanar_nrm_left = allfeatures_nrm_left[:planar_left.shape[0]], allfeatures_nrm_left[planar_left.shape[0]:]
    planar_nrm_right, nonplanar_nrm_right = allfeatures_nrm_right[:planar_right.shape[0]], allfeatures_nrm_right[planar_right.shape[0]:]
    
    
    ### Calculate relative pose using the essential matrix
    
    # Only do pose estimation if we've got 2D points of non-planar geometry
    if has_nonplanar:
    
        # Determine inliers by calculating the fundamental matrix on all points,
        # except when mute_chessboard_corners is True: only use nonplanar_nrm_left and nonplanar_nrm_right
        F, status = cv2.findFundamentalMat(allfeatures_nrm_left, allfeatures_nrm_right, cv2.FM_RANSAC, 0.006 * np.amax(allfeatures_nrm_left), 0.99)    # threshold from [Snavely07 4.1]
        # OpenCV BUG: "status" matrix is not initialized with zeros in some cases, reproducable with 2 sets of 8 2D points equal to (0., 0.)
        #   maybe because "_mask.create()" is not called:
        #   https://github.com/Itseez/opencv/blob/7e2bb63378dafb90063af40caff20c363c8c9eaf/modules/calib3d/src/ptsetreg.cpp#L185
        # Workaround to test on outliers: use "!= 1", instead of "== 0"
        print status.T
        inlier_idxs = np.where(status == 1)[0]
        print "Removed", allfeatures_nrm_left.shape[0] - inlier_idxs.shape[0], "outlier(s)."
        num_planar = np.where(inlier_idxs < num_planar)[0].shape[0]
        num_nonplanar = inlier_idxs.shape[0] - num_planar
        print "num chessboard inliers:", num_planar
        allfeatures_left, allfeatures_right = allfeatures_left[inlier_idxs], allfeatures_right[inlier_idxs]
        allfeatures_nrm_left, allfeatures_nrm_right = allfeatures_nrm_left[inlier_idxs], allfeatures_nrm_right[inlier_idxs]
        if not mute_chessboard_corners:
            objp = objp[inlier_idxs[:num_planar]]
            planar_left, planar_right = allfeatures_left[:num_planar], allfeatures_right[:num_planar]
            planar_nrm_left, planar_nrm_right = allfeatures_nrm_left[:num_planar], allfeatures_nrm_right[:num_planar]
        nonplanar_nrm_left, nonplanar_nrm_right = allfeatures_nrm_left[num_planar:], allfeatures_nrm_right[num_planar:]
        
        # Calculate first solution of relative pose
        if allfeatures_nrm_left.shape[0] >= 8:
            F, status = cv2.findFundamentalMat(allfeatures_nrm_left, allfeatures_nrm_right, cv2.FM_8POINT)
            print "F:"
            print F
        else:
            print "Error: less than 8 pairs of inliers found, I can't perform the 8-point algorithm."
        #E = (K_right.T) .dot (F) .dot (K_left)    # "Multiple View Geometry in CV" by Hartley&Zisserman (9.12)
        E = F    # K = I because we already normalized the coordinates
        print "Correct determinant of essential matrix?", (abs(cv2.determinant(E)) <= 1e-7)
        w, u, vt = cv2.SVDecomp(E, flags=cv2.SVD_MODIFY_A)
        print w
        if ((w[0] < w[1] and w[0]/w[1]) or (w[1] < w[0] and w[1]/w[0]) or 0) < 0.7:
            print "Essential matrix' 'w' vector deviates too much from expected"
        W = np.array([[0., -1., 0.],    # Hartley&Zisserman (9.13)
                      [1.,  0., 0.],
                      [0.,  0., 1.]])
        R = (u) .dot (W) .dot (vt)    # Hartley&Zisserman result 9.19
        det_R = cv2.determinant(R)
        print "Coherent rotation?", (abs(det_R) - 1 <= 1e-7)
        if det_R - (-1) < 1e-7:    # http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
            # E *= -1:
            vt *= -1    # svd(-E) = u * w * (-v).T
            R *= -1     # => u * W * (-v).T = -R
            print "det(R) == -1, compensated."
        t = u[:, 2:3]    # Hartley&Zisserman result 9.19
        P = trfm.P_from_R_and_t(R, t)
        
        # Select right solution where the (center of the) 3d points are/is in front of both cameras,
        # when has_prev_triangl_points == True, we can do it a lot faster.
        
        test_point = np.ones((4, 1))    # 3D point to test the solutions with
        
        if has_prev_triangl_points:
            print "Using advantage of already triangulated points' position"
            print P
            
            # Select the closest already triangulated cloudpoint idx to the center of the cloud
            center_of_mass = objp.sum(axis=0) / objp.shape[0]
            center_objp_idx = np.argmin(((objp - center_of_mass)**2).sum(axis=1))
            print "center_objp_idx:", center_objp_idx
            
            # Select the corresponding image points
            center_imgp_left = planar_nrm_left[center_objp_idx]
            center_imgp_right = planar_nrm_right[center_objp_idx]
            
            # Select the corresponding 3D point
            test_point[0:3, :] = objp[center_objp_idx].reshape(3, 1)
            print "test_point:"
            print test_point
            test_point = P_left.dot(test_point)    # set the reference axis-system to the one of the left camera, note that are_points_in_front_of_left_camera is automatically True
            
            center_objp_triangl, triangl_status = iterative_LS_triangulation(
                    center_imgp_left.reshape(1, 2), np.eye(4),
                    center_imgp_right.reshape(1, 2), P )
            print "triangl_status:", triangl_status
            
            if (center_objp_triangl) .dot (test_point[0:3, 0:1]) < 0:
                P[0:3, 3:4] *= -1    # do a baseline reversal
                print P, "fixed triangulation inversion"
            
            if P[0:3, :].dot(test_point)[2, 0] < 0:    # are_points_in_front_of_right_camera is False
                P[0:3, 0:3] = (u) .dot (W.T) .dot (vt)    # use the other solution of the twisted pair ...
                P[0:3, 3:4] *= -1    # ... and also do a baseline reversal
                print P, "fixed camera projection inversion"
        
        elif num_nonplanar > 0:
            print "Doing all ambiguity checks since there are no already triangulated points"
            print P
            
            for i in range(4):    # check all 4 solutions
                objp_triangl, triangl_status = iterative_LS_triangulation(
                        nonplanar_nrm_left, np.eye(4),
                        nonplanar_nrm_right, P )
                print "triangl_status:", triangl_status
                center_of_mass = objp_triangl.sum(axis=0) / len(objp_triangl)    # select the center of the triangulated cloudpoints
                test_point[0:3, :] = center_of_mass.reshape(3, 1)
                print "test_point:"
                print trfm.P_inv(P_left) .dot (test_point)
                
                if np.eye(3, 4).dot(test_point)[2, 0] > 0 and P[0:3, :].dot(test_point)[2, 0] > 0:    # are_points_in_front_of_cameras is True
                    break
                
                if i % 2:
                    P[0:3, 0:3] = (u) .dot (W.T) .dot (vt)   # use the other solution of the twisted pair
                    print P, "using the other solution of the twisted pair"
                
                else:
                    P[0:3, 3:4] *= -1    # do a baseline reversal
                    print P, "doing a baseline reversal"
        
        
        are_points_in_front_of_left_camera = (np.eye(3, 4).dot(test_point)[2, 0] > 0)
        are_points_in_front_of_right_camera = (P[0:3, :].dot(test_point)[2, 0] > 0)
        print "are_points_in_front_of_cameras?", are_points_in_front_of_left_camera, are_points_in_front_of_right_camera
        if not (are_points_in_front_of_left_camera and are_points_in_front_of_right_camera):
            print "No valid solution found!"
        
        P_left_result = trfm.P_inv(P).dot(P_right)
        P_right_result = P.dot(P_left)
        
        print "P_left"
        print P_left
        print "P_rel"
        print P
        print "P_right"
        print P_right
        print "=> error:", reprojection_error(
                cameraMatrix, distCoeffs,
                [rvec_left, rvec_right],
                [tvec_left, tvec_right],
                [objp_orig] * 2, [planar_left_orig, planar_right_orig] )[1]
        
        print "P_left_result"
        print P_left_result
        print "P_right_result"
        print P_right_result
        print "=> error:", reprojection_error(    # we use "P_left" instead of "P_left_result" because the latter depends on the unknown "P_right"
                cameraMatrix, distCoeffs,
                [cvh.Rodrigues(P_left[0:3, 0:3]), cvh.Rodrigues(P_right_result[0:3, 0:3])],
                [P_left[0:3, 3], P_right_result[0:3, 3]],
                [objp_orig] * 2, [planar_left_orig, planar_right_orig] )[1]
    
    
    ### Triangulate
    
    objp_result = np.zeros((0, 3))
    
    if has_prev_triangl_points:
        # Do triangulation of all points
        # NOTICE: in a real case, we should only use not yet triangulated points that are in sight
        objp_result, triangl_status = iterative_LS_triangulation(
                allfeatures_nrm_left, P_left,
                allfeatures_nrm_right, P_right )
        print "triangl_status:", triangl_status
    
    elif num_nonplanar > 0:
        # We already did the triangulation during the pose estimation, but we still need to backtransform them from the left camera axis-system
        objp_result = trfm.P_inv(P_left) .dot (np.concatenate((objp_triangl.T, np.ones((1, len(objp_triangl))))))
        objp_result = objp_result[0:3, :].T
        print objp_triangl
    
    print "objp:"
    print objp
    print "=> error:", reprojection_error(
            cameraMatrix, distCoeffs, [rvec_left, rvec_right], [tvec_left, tvec_right],
            [objp_orig] * 2,
            [planar_left_orig, planar_right_orig] )[1]
    
    print "objp_result of chessboard:"
    print objp_result[:num_planar, :]
    if has_nonplanar:
        print "objp_result of non-planar geometry:"
        print objp_result[num_planar:, :]
    if num_planar + num_nonplanar == 0:
        print "=> error: undefined"
    else:
        print "=> error:", reprojection_error(
                cameraMatrix, distCoeffs, [rvec_left, rvec_right], [tvec_left, tvec_right],
                [objp_result] * 2,
                [allfeatures_left, allfeatures_right] )[1]
    
    
    ### Print total combined reprojection error
    
    # We only have both pose estimation and triangulation if we've got 2D points of non-planar geometry
    if has_nonplanar:
        if num_planar + num_nonplanar == 0:
            print "=> error: undefined"
        else:
            print "Total combined error:", reprojection_error(    # we use "P_left" instead of "P_left_result" because the latter depends on the unknown "P_right"
                    cameraMatrix, distCoeffs,
                    [cvh.Rodrigues(P_left[0:3, 0:3]), cvh.Rodrigues(P_right_result[0:3, 0:3])],
                    [P_left[0:3, 3], P_right_result[0:3, 3]],
                    [objp_result] * 2,
                    [allfeatures_left, allfeatures_right] )[1]
    
    
    """
    Further things we can do (in a real case):
        1. solvePnP() on inliers (including previously already triangulated points),
            or should we use solvePnPRansac() (in that case what to do with the outliers)?
        2. re-triangulate on new pose estimation
    """
    
    
    ### Print summary to be used in Blender to visualize
    
    print "Camera poses:"
    if has_nonplanar:
        def print_pose(rvec, tvec):
            ax, an = trfm.axis_and_angle_from_rvec(-rvec)
            print "axis, angle = \\\n", list(ax.reshape(-1)), ",", an    # R
            print "pos = \\\n", list(-cvh.Rodrigues(-rvec).dot(tvec).reshape(-1))    # t
        print "Left"
        print_pose(rvec_left, tvec_left)
        print
        print "Left_result"
        print_pose(cvh.Rodrigues(P_left_result[0:3, 0:3]), P_left_result[0:3, 3:4])
        print
        print "Right"
        print_pose(rvec_right, tvec_right)
        print
        print "Right_result"
        print_pose(cvh.Rodrigues(P_right_result[0:3, 0:3]), P_right_result[0:3, 3:4])
        print
    else:
        print "<skipped: no non-planar objects have been selected>"
        print
    
    print "Points:"
    print "Chessboard"
    print "coords = \\\n", map(list, objp_result[:num_planar, :])
    print 
    if has_nonplanar:
        print "Non-planar geometry"
        print "coords_nonplanar = \\\n", map(list, objp_result[num_planar:, :])
        print
    
    ### Return to remember last manually matched successful non-planar imagepoints
    return nonplanar_left, nonplanar_right


def realtime_pose_estimation(device_id, filename_base_extrinsics, cameraMatrix, distCoeffs, objp, boardSize):
    """
    This interactive demo will track a chessboard in realtime using a webcam,
    and the WORLD axis-system will be drawn on it: [X Y Z] = [red green blue]
    Further on you will see some data in the bottom-right corner,
    this indicates both the pose of the current image w.r.t. the WORLD axis-system,
    as well as the pose of the current image w.r.t. the previous keyframe pose.
    
    To create a new keyframe while running, press SPACE.
    Each time a new keyframe is generated,
    the corresponding image and data (in txt-format) is written to the 'filename_base_extrinsics' folder.
    
    All poses are defined in the WORLD axis-system,
    the rotation notation follows axis-angle representation: '<unit vector> * <magnitude (degrees)>'.
    
    To quit, press ESC.
    """
    cv2.namedWindow("Image (with axis-system)")
    axis_system_objp = np.array([ [0., 0., 0.],      # Origin (black)
                                  [4., 0., 0.],      # X-axis (red)
                                  [0., 4., 0.],      # Y-axis (green)
                                  [0., 0., 4.] ])    # Z-axis (blue)
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.5
    mlt = cvh.MultilineText()
    cap = cv2.VideoCapture(device_id)

    imageNr = 0    # keyframe image id
    rvec_prev = np.zeros((3, 1))
    rvec = None
    tvec_prev = np.zeros((3, 1))
    tvec = None

    # Loop until 'q' or ESC pressed
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
            origin, xAxis, yAxis, zAxis = np.rint(imgp_reproj.reshape(-1, 2)).astype(int)    # round to nearest int
            
            # OpenCV's 'rvec' and 'tvec' seem to be defined as follows:
            #   'rvec': rotation transformation: transforms points from "WORLD axis-system -> CAMERA axis-system"
            #   'tvec': translation of "CAMERA center -> WORLD center", all defined in the "CAMERA axis-system"
            rvec *= -1    # convert to: "CAMERA axis-system -> WORLD axis-system", equivalent to rotation of CAMERA axis-system w.r.t. WORLD axis-system
            tvec = cvh.Rodrigues(rvec).dot(tvec)    # bring to "WORLD axis-system", ...
            tvec *= -1    # ... and change direction to "WORLD center -> CAMERA center"
            
            # Calculate pose relative to last keyframe
            rvec_rel = -trfm.delta_rvec(-rvec, -rvec_prev)    # calculate the inverse of the rotation between subsequent "WORLD -> CAMERA" rotations
            tvec_rel = tvec - tvec_prev
            
            # Extract axis and angle, to enhance representation
            rvec_axis, rvec_angle = trfm.axis_and_angle_from_rvec(rvec)
            rvec_rel_axis, rvec_rel_angle = trfm.axis_and_angle_from_rvec(rvec_rel)
            
            # Draw axis-system
            cvh.line(img, origin, xAxis, rgb(255,0,0), thickness=2, lineType=cv2.CV_AA)
            cvh.line(img, origin, yAxis, rgb(0,255,0), thickness=2, lineType=cv2.CV_AA)
            cvh.line(img, origin, zAxis, rgb(0,0,255), thickness=2, lineType=cv2.CV_AA)
            cvh.circle(img, origin, 4, rgb(0,0,0), thickness=-1)    # filled circle, radius 4
            cvh.circle(img, origin, 5, rgb(255,255,255), thickness=2)    # white 'O', radius 5
            
            # Draw pose information
            texts = []
            texts.append("Current pose:")
            texts.append("    Rvec: %s * %.1fdeg" % (format3DVector(rvec_axis), degrees(rvec_angle)))
            texts.append("    Tvec: %s" % format3DVector(tvec))
            texts.append("Relative to previous pose:")
            texts.append("    Rvec: %s * %.1fdeg" % (format3DVector(rvec_rel_axis), degrees(rvec_rel_angle)))
            texts.append("    Tvec: %s" % format3DVector(tvec_rel))
            
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
            filename = filename_base_extrinsics + str(imageNr)
            cv2.imwrite(filename + ".jpg", img)    # write image to jpg-file
            textTotal = '\n'.join(texts)
            open(filename + ".txt", 'w').write(textTotal)    # write data to txt-file
            
            print "Saved keyframe image+data to", filename, ":"
            print textTotal
            
            imageNr += 1
            rvec_prev = rvec
            tvec_prev = tvec


def calibrate_relative_poses_interactive(image_sets, cameraMatrixs, distCoeffss, imageSizes, boardSizes, board_scales, board_rvecs, board_tvecs):
    """
    Make an estimate of the relative poses (as 4x4 projection matrices) between many cameras.
    Base these relative poses to the first camera.
    
    Each camera should be looking at its own chessboard,
    use different sizes of chessboards if a camera sees chessboard that are not associated with that camera.
    'board_scales' scales the chessboard-units to world-units.
    'board_rvecs' and 'board_tvecs' transform the rescaled local chessboard-coordinates to world-coordinates.
    
    The inverse of the reprojection error is used for weighting.
    """
    num_cams = len(image_sets)
    num_images = len(image_sets[0])
    reproj_error_max = 0
    
    # Preprocess object-points of the different boards
    board_objps = []
    for boardSize, board_scale, board_rvec, board_tvec in zip(
            boardSizes, board_scales, board_rvecs, board_tvecs ):
        objp = np.ones((np.prod(boardSize), 4))
        objp[:, 0:3] = prepare_object_points(boardSize) * board_scale
        objp = objp.dot(trfm.P_from_R_and_t(cvh.Rodrigues(board_rvec), np.array(board_tvec).reshape(3, 1))[0:3, :].T)
        board_objps.append(objp)
    
    # Calculate all absolute poses
    Ps = np.zeros((num_images, num_cams, 4, 4))
    weights = np.zeros((num_images, 1, 1, 1))
    for i, images in enumerate(zip(*image_sets)):
        reproj_error = 0
        for c, (image, cameraMatrix, distCoeffs, imageSize, boardSize, board_objp) in enumerate(zip(
                images, cameraMatrixs, distCoeffss, imageSizes, boardSizes, board_objps )):
            img = cv2.imread(image)
            ret, corners = cvh.extractChessboardFeatures(img, boardSize)
            if not ret:
                print "Error: Image '%s' didn't contain a chessboard of size %s." % (image, boardSize)
                return False, None
            
            # Draw and display the corners
            cv2.drawChessboardCorners(
                    img, boardSize, corners, ret )
            cv2.imshow("img", img)
            cv2.waitKey(100)
            
            ret, rvec, tvec = cv2.solvePnP(board_objp, corners, cameraMatrix, distCoeffs)
            Ps[i, c, :, :] = trfm.P_from_R_and_t(cvh.Rodrigues(rvec), tvec)
            reproj_error = max(reprojection_error(cameraMatrix, distCoeffs, [rvec], [tvec], [board_objp], [corners])[1], reproj_error)    # max: worst case
        reproj_error_max = max(reproj_error, reproj_error_max)
        weights[i] = 1. / reproj_error
    
    # Apply weighting on Ps, and rebase against first camera
    Ps *= weights / weights.sum()
    Ps = Ps.sum(axis=0)
    Pref_inv = trfm.P_inv(Ps[0, :, :])    # use first cam as reference
    return True, [P.dot(Pref_inv) for P in Ps], reproj_error_max



def get_variable(name, func = lambda x: x):
    value = eval(name)
    value_inp = raw_input("%s [%s]: " % (name, repr(value)))
    if value_inp:
        value = func(value_inp)
        exec("global " + name)
        globals()[name] = value

def main():
    global boardSize, filename_base_chessboards, filename_intrinsics, filename_distorted, filename_triangl_pose_est_left, filename_triangl_pose_est_right, filename_base_extrinsics, filenames_extra_chessboards, filenames_extra_intrinsics, extra_boardSizes, extra_board_scales, extra_board_rvecs, extra_board_tvecs, device_id
    boardSize = (8, 6)
    filename_base_chessboards = os.path.join("chessboards", "chessboard*.jpg")    # calibration images of the base camera
    filename_intrinsics = "camera_intrinsics.txt"
    filename_distorted = os.path.join("chessboards", "chessboard07.jpg")    # a randomly chosen image
    #filename_triangl_pose_est_left = os.path.join("chessboards", "chessboard07.jpg")    # a randomly chosen image
    #filename_triangl_pose_est_right = os.path.join("chessboards", "chessboard08.jpg")    # a randomly chosen image
    filename_triangl_pose_est_left = os.path.join("chessboards_and_nonplanar", "image-0001.jpeg")    # a randomly chosen image
    filename_triangl_pose_est_right = os.path.join("chessboards_and_nonplanar", "image-0056.jpeg")    # a randomly chosen image
    filename_base_extrinsics = os.path.join("chessboards_extrinsic", "chessboard")
    filenames_extra_chessboards = (os.path.join("chessboards_front", "front-*.jpg"),    # sets of calibration images of extra cameras
                                   os.path.join("chessboards_bottom", "bottom-*.jpg"))
    filenames_extra_intrinsics = ("camera_intrinsics_front.txt", "camera_intrinsics_bottom.txt")    # intrinsics of extra cameras
    extra_boardSizes = ((8, 6), (8, 6))
    extra_board_scales = (1., 1.)
    extra_board_rvecs = ((0., 0., 0.), (0., -pi/2, 0.))
    extra_board_tvecs = ((0., 0., 0.), (6., 0., (1200.-193.)/27.6+1.))
    device_id = 1    # webcam

    nonplanar_left = nonplanar_right = np.zeros((0, 2))

    help_text = """\
    Choose between: (in order)
        1: prepare_object_points (required)
        2: calibrate_camera_interactive (required for "reprojection_error")
        3: save_camera_intrinsics
        4: load_camera_intrinsics (required)
        5: undistort_image
        6: reprojection_error
        7: triangl_pose_est_interactive
        8: realtime_pose_estimation (recommended)
        9: calibrate_relative_poses_interactive
        q: quit
    
    Info: Sometimes you will be prompted: 'someVariable [defaultValue]: ',
          in that case you can type a new value,
          or simply press ENTER to preserve the default value.
    """
    from textwrap import dedent
    print dedent(help_text)
    
    inp = ""
    while inp.lower() != "q":
        inp = raw_input("\n: ").strip()
        
        if inp == "1":
            get_variable("boardSize", lambda x: eval("(%s)" % x))
            print    # add new-line
            
            objp = prepare_object_points(boardSize)
        
        elif inp == "2":
            get_variable("filename_base_chessboards")
            from glob import glob
            images = sorted(glob(filename_base_chessboards))
            print    # add new-line
            
            reproj_error, cameraMatrix, distCoeffs, rvecs, tvecs, objectPoints, imagePoints, imageSize = \
                    calibrate_camera_interactive(images, objp, boardSize)
            print "cameraMatrix:\n", cameraMatrix
            print "distCoeffs:\n", distCoeffs
            print "reproj_error:", reproj_error
            
            cv2.destroyAllWindows()
        
        elif inp == "3":
            get_variable("filename_intrinsics")
            print    # add new-line
            
            save_camera_intrinsics(filename_intrinsics, cameraMatrix, distCoeffs, imageSize)
        
        elif inp == "4":
            get_variable("filename_intrinsics")
            print    # add new-line
            
            cameraMatrix, distCoeffs, imageSize = \
                    load_camera_intrinsics(filename_intrinsics)
        
        elif inp == "5":
            get_variable("filename_distorted")
            img = cv2.imread(filename_distorted)
            print    # add new-line
            
            img_undistorted, roi = \
                    undistort_image(img, cameraMatrix, distCoeffs, imageSize)
            cv2.imshow("Original Image", img)
            cv2.imshow("Undistorted Image", img_undistorted)
            print "Press any key to continue."
            cv2.waitKey()
            
            cv2.destroyAllWindows()
        
        elif inp == "6":
            mean_error, square_error = \
                    reprojection_error(cameraMatrix, distCoeffs, rvecs, tvecs, objectPoints, imagePoints)
            print "mean absolute error:", mean_error
            print "square error:", square_error
        
        elif inp == "7":
            print triangl_pose_est_interactive.__doc__
            
            get_variable("filename_triangl_pose_est_left")
            img_left = cv2.imread(filename_triangl_pose_est_left)
            get_variable("filename_triangl_pose_est_right")
            img_right = cv2.imread(filename_triangl_pose_est_right)
            print    # add new-line
            
            nonplanar_left, nonplanar_right = \
                    triangl_pose_est_interactive(img_left, img_right, cameraMatrix, distCoeffs, imageSize, objp, boardSize, nonplanar_left, nonplanar_right)
            
            cv2.destroyAllWindows()
        
        elif inp == "8":
            print realtime_pose_estimation.__doc__
            
            get_variable("device_id", int)
            get_variable("filename_base_extrinsics")
            print    # add new-line
            
            realtime_pose_estimation(device_id, filename_base_extrinsics, cameraMatrix, distCoeffs, objp, boardSize)
            
            cv2.destroyAllWindows()
        
        elif inp == "9":
            print calibrate_relative_poses_interactive.__doc__
            
            get_variable("filenames_extra_chessboards", lambda x: eval("(%s)" % x))
            from glob import glob
            image_sets = [sorted(glob(images)) for images in filenames_extra_chessboards]
            get_variable("filenames_extra_intrinsics", lambda x: eval("(%s)" % x))
            cameraMatrixs, distCoeffss, imageSizes = zip(*map(load_camera_intrinsics, filenames_extra_intrinsics))
            get_variable("extra_boardSizes", lambda x: eval("(%s)" % x))
            get_variable("extra_board_scales", lambda x: eval("(%s)" % x))
            get_variable("extra_board_rvecs", lambda x: eval("(%s)" % x))
            get_variable("extra_board_tvecs", lambda x: eval("(%s)" % x))
            print    # add new-line
            
            ret, Ps, reproj_error_max = \
                    calibrate_relative_poses_interactive(image_sets, cameraMatrixs, distCoeffss, imageSizes,
                                                         extra_boardSizes, extra_board_scales, extra_board_rvecs, extra_board_tvecs)
            if ret:
                print "Ps:"
                for P in Ps: print P
                print "reproj_error_max:", reproj_error_max

if __name__ == "__main__":
    main()
