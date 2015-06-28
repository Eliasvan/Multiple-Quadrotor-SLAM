#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function    # Python 3 compatibility

import os
import numpy as np
import cv2

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "python_libs"))
import transforms as trfm
import calibration_tools
import dataset_tools



def join_path(*path_list):
    """Convenience function for creating OS-indep relative paths."""
    return os.path.relpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *path_list))


def main():
    """
    A file with the initial pose and 3D points of the SVO dataset, will be created.
    """
    num_features = 100
    num_iterations = 53
    #corner_quality_level = 0.4805294789864505

    print ("Searching for features ( max", num_features, ")...")
    
    # Load first image
    img = cv2.cvtColor(
            cv2.imread(join_path("sin2_tex2_h1_v8_d", "img", "frame_000002_0.png")),
            cv2.COLOR_BGR2GRAY )
    
    # Find just enough features, iterate using bisection to find the best "corner_quality_level" value
    corner_min_dist = 0.
    lower = 0.
    upper = 1.
    for i in range(num_iterations):
        corner_quality_level = (lower + upper) / 2
        imgp = cv2.goodFeaturesToTrack(img, num_features, corner_quality_level, corner_min_dist)
        if imgp == None or len(imgp) < num_features:
            upper = corner_quality_level
        else:
            lower = corner_quality_level
    corner_quality_level = lower if lower else corner_quality_level
    imgp = cv2.goodFeaturesToTrack(img, num_features, corner_quality_level, corner_min_dist).reshape((-1, 2))
    
    print (len(imgp), "features found, corner_quality_level:", corner_quality_level)
    
    # Load camera intrinsics
    cameraMatrix, distCoeffs, imageSize = calibration_tools.load_camera_intrinsics(
            join_path("camera_intrinsics.txt") )
    
    # Load and save first pose
    timestps, locations, quaternions = dataset_tools.load_cam_trajectory_TUM(
            join_path("sin2_tex2_h1_v8_d", "traj_groundtruth.txt") )
    P = trfm.P_from_pose_TUM(quaternions[0], locations[0])
    np.savetxt(join_path("sin2_tex2_h1_v8_d", "init_pose.txt"), P)
    
    # Generate 3D points, knowing that they lay at the plane z=0:
    # for each 2D point, the following system of equations is solved for X:
    #     A * X = b
    # where:
    #     X = [x, y, s]^T
    #     [x, y, 0] the 3D point's coords
    #     s is an unknown scalefactor of the normalized 2D point
    #     A = [[ 1  0 |    ],
    #          [ 0  1 | Pu ],
    #          [ 0  0 |    ]]
    #     Pu = - R^(-1) * p
    #     p = [u, v, 1]^T the 2D point's homogeneous coords
    #     b = - R^(-1) * t
    #     R, t = 3x3 rotation matrix and 3x1 translation vector of 3x4 pose matrix P
    # The system of equations is solved in bulk for all points using broadcasted backsubstitution
    objp = np.empty((len(imgp), 3)).T
    objp[2:3, :] = 0.    # z = 0
    imgp_nrml = cv2.undistortPoints(np.array([imgp]), cameraMatrix, distCoeffs)[0]
    imgp_nrml = np.concatenate((imgp_nrml, np.ones((len(imgp), 1))), axis=1)    # to homogeneous coords
    P_inv = trfm.P_inv(P)
    Pu = P_inv[0:3, 0:3].dot(imgp_nrml.T)
    scale = -P_inv[2:3, 3:4] / Pu[2:3, :]
    objp[0:2, :] = P_inv[0:2, 3:4] + scale * Pu[0:2, :]    # x and y components
    objp = objp.T
    
    # Save 3D points
    dataset_tools.save_3D_points_to_pcd_file(
            join_path("sin2_tex2_h1_v8_d", "init_points.pcd"),
            objp )
    
    print ("Done.")

if __name__ == "__main__":
    main()
