#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import tan, asin, pi
import numpy as np
from numpy import random
import scipy.io as sio
import cv2

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "python_libs")
import transforms as trfm
import dataset_tools
import triangulation



""" 3D point configurations """

def finite_3D_points(r, x_on=True, y_on=True, z_on=True):
    """
    Generate 3D coordinates of points in a grid with spacing 1, confined in a sphere of radius "r".
    Homogeneous coordinates (4D) of the points are returned.
    
    Disable a dimension by setting "<axis-name>_on" to False.
    """
    rx, ry, rz = r*x_on, r*y_on, r*z_on
    
    points = np.array([ (x, y, z, 1.) for x in range(-rx, rx + 1)
                                      for y in range(-ry, ry + 1)
                                      for z in range(-rz, rz + 1)
                                      if (x*x + y*y + z*z) <= r*r ])
    return points

def infinite_3D_points(r, max_angle, x_on=True, y_on=True):
    """
    Generate 3D coordinates of points in a XY grid with spacing 1, confined in a circle of radius "r",
    transformed to infinite +Z, where the maximum angle from origin to a point equals "max_angle".
    Homogeneous coordinates (4D) of the points are returned.
    
    Disable a dimension by setting "<axis-name>_on" to False.
    """
    rx, ry = r*x_on, r*y_on
    
    z = r / tan(max_angle)
    points = np.array([ (x, y, z, 0.) for x in range(-rx, rx + 1)
                                      for y in range(-ry, ry + 1)
                                      if (x*x + y*y) <= r*r ])
    return points

'''   DEPRECATED
def scene_3D_points(r=1., filename="scene_3D_points.mat"):
    """
    Load 3D coordinates of points from a MATLAB file given by filename "filename".
    Use "r" as geometry scaling factor.
    
    The points are assumed to be confined in a cube with edges of length 2.
    The number of columns of the MATLAB variable "scene_3D_points" should
    equal 3 or 4 (in case of homogeneous coordinates).
    """
    
    points = sio.loadmat(filename)["scene_3D_points"]
    if points.shape[1] == 3:
        points = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    
    points[:, 0:3] *= r
    
    return points
'''

def scene_3D_points(r=1., filename="scene_3D_points.pcd"):
    """
    Load 3D coordinates of points from a PointCloud .pcd-file given by filename "filename".
    Use "r" as geometry scaling factor.
    
    The points are assumed to be confined in a cube with edges of length 2.
    """
    
    points = dataset_tools.load_3D_points_from_pcd_file(filename)[0]    # only load verts, not colors
    points *= r
    
    points = np.concatenate((points, np.ones((len(points), 1))), axis=1)    # to homogeneous coordinates
    
    return points

""" Camera configurations """

class Camera:

    def camera_intrinsics(self, resolution, k1=0.):
        """
        Generate intrinsics matrix and distortion coefficients (4) of a camera
        with resolution given by "resolution": (width, height),
        and with a 2nd order radial distortion coefficient of "k1".
        """
        f = min(resolution)    # focal length
        c = np.array(resolution) / 2.    # principal point
        
        K = np.eye(3)
        K[0, 0] = K[1, 1] = f    # set cam scaling
        K[0:2, 2] = c    # set cam principal point
        
        dist_coeffs = np.array([k1, 0.,     # 2nd and 4th order radial distortion coefficients
                                0., 0.])    # tangential distortion coefficients
        
        self.f, self.c, self.K, self.dist_coeffs = f, c, K, dist_coeffs

    def camera_pose(self, offset, sideways=0., towards=0., angle=0.):
        """
        Generate 3x4 projection matrix of a camera initially
        with center at (0, 0, -"offset") with unit orientation (along +Z; Y-axis down)
        and transformed by a translation (+"sideways", 0, +"towards")
            and by a local rotation of "angle" around Y-axis (to the left).
        """
        rvec = (0, angle, 0)
        cam_center = (sideways, 0, -offset + towards)
        
        R = cv2.Rodrigues(rvec)[0]
        tvec = -R.dot(np.array(cam_center).reshape(3, 1))
        P = trfm.P_from_R_and_t(R, tvec)[0:3, :]
        
        self.P = P

    """ 2D point generation + noise models """

    def project_points(self, points_3D, save_result=True):
        """
        Project the 3D points "points_3D" (in homogeneous coordinates: 4D) on the camera image.
        
        If "save_result" is set to False, the result is returned,
        otherwise it overwrites the camera's list of projected points.
        """
        
        # Do the projection here, OpenCV's "projectPoints()" can't handle 4D
        points_3D = self.P.dot(points_3D.T).T
        
        # Only use OpenCV's "projectPoints()" to apply the intrinsics
        rvec = tvec = (0., 0., 0.)    # unit projection
        points_2D, jacob = cv2.projectPoints(
                points_3D, rvec, tvec, self.K, self.dist_coeffs )
        points_2D = points_2D.reshape(len(points_2D), 2)
        
        if save_result:
            self.points_2D_exact = self.points_2D = points_2D
        else:
            return points_2D
    
    def apply_noise(self, sigma, discretized=False):
        """
        Apply additive zero mean gaussian noise to the projected points on the image.
        """
        
        if sigma:
            points_2D = self.points_2D_exact + random.normal(0, sigma, self.points_2D_exact.shape)
        else:
            points_2D = self.points_2D_exact
        
        if discretized:
            points_2D = np.rint(points_2D)
        
        self.points_2D = points_2D
    
    def normalized_points(self):
        """
        Return the normalized (noisy) image points of this camera.
        """
        if not self.dist_coeffs[0]:    # if no distortion, take a shortcut
            u = np.array(self.points_2D)
            u[:, 0] -= self.c[0]
            u[:, 1] -= self.c[1]
            return u / self.f
        return cv2.undistortPoints(np.array([self.points_2D]), self.K, self.dist_coeffs)[0]



""" Error evaluation functions """

def error_vectors_3D(points_3D_exact, points_3D_calc):
    """
    Return the squared euclidean errors between each point of "points_3D_exact" and "points_3D_calc".
    The 3D points of "points_3D_calc" must not have homogeneous coordinates.
    
    This function only works for finite coordinates.
    """
    
    # Convert "points_3D_exact" from 4D to 3D (assumes weighting of 1.0)
    return points_3D_calc - points_3D_exact[:, 0:3]

def error_vectors_2D(points_3D_calc, cam1, cam2):
    """
    Return the squared euclidean errors between each point of "cam1" and "points_3D_calc".
    """
    if points_3D_calc.shape[1] == 3:    # convert to homogeneous coordinates, if needed
        points_3D_calc = np.concatenate((points_3D_calc, np.ones((len(points_3D_calc), 1))), axis=1)
    
    cam1_points_2D_calc = cam1.project_points(points_3D_calc, False)
    cam2_points_2D_calc = cam2.project_points(points_3D_calc, False)
    
    cam1_errors_2D = cam1_points_2D_calc - cam1.points_2D_exact
    cam2_errors_2D = cam2_points_2D_calc - cam2.points_2D_exact
    
    return cam1_errors_2D, cam2_errors_2D

def error_rms(error_vectors):
    """
    Return the root mean 
       and the root median
    square error of all error_vectors,
    as well as a list of errors.
    """
    if type(error_vectors) == list:
        error_vectors = np.concatenate(error_vectors)
    
    errors = np.sum(error_vectors**2, axis=1)
    
    return np.sqrt(np.mean(errors)), np.sqrt(np.median(errors)), errors

def vector_stat(error_vectors):
    """
    Return the mean vector and covariance matrix of each point in "error_vectors".
    "error_vectors" has dimensions (num_trials, N, d) where
        "num_trials" is the amount of repeated measurements,
        "N" is the amount of different (unrelated) points,
        "d" is the dimension of a point vector.
    """
    num_trials = float(error_vectors.shape[0])
    d = error_vectors.shape[2]
    
    means = np.empty((error_vectors.shape[1], d))
    covars = np.empty((error_vectors.shape[1], d, d))
    one = np.ones((num_trials, 1))
    
    for i in range(error_vectors.shape[1]):
        point_measurements = error_vectors[:, i, :]
        means[i] = one.T.dot(point_measurements) / num_trials
        point_deviations = point_measurements - one.dot(means[i].reshape(1, d))
        covars[i] = point_deviations.T.dot(point_deviations) / num_trials
    
    return means, covars

def robustness_stat(errors, statuses):
    """
    Return the ratio of false positives and false negatives
    based on the real 3D error "err3D",
    and the estimated status "statuses" of the point (positive corresponds with a properly triangulated point).
    
    'properly triangulated' means that a certain threshold for the squared error distance is not exceeded.
    """
    if type(statuses) == list:
        statuses = np.concatenate(statuses)
    
    positives_max = (errors <= robustness_thresh_max)
    positives_min = (errors <= robustness_thresh_min)
    positives_est = (statuses > 0)
    
    false_positives = np.logical_and(positives_max == False, positives_est)
    false_negatives = np.logical_and(positives_min, positives_est == False)
    
    return np.mean(false_positives), np.mean(false_negatives)



""" Default scenario parameters, generators and random seeds """

default_params = {
    "3D_points_source"      : "finite",    # "finite", "infinite", or "scene"
    "3D_points_r"           : 4,
    "3D_points_max_angle"   : pi / 4,
    "3D_points_x_on"        : True,
    "3D_points_y_on"        : True,
    "3D_points_z_on"        : True,
    
    "cam_resolution"        : (640, 480),
    "cam_k1"                : 0.3,
    "cam_pose_offset"       : 40.,
    "cam_noise_sigma"       : 0.8,    # based on (99.7% of the) data of footage with rolling shutter
    "cam_noise_discretized" : True,
    
    "cam1_pose_sideways"    : 0.,
    "cam1_pose_towards"     : 0.,
    "cam1_pose_angle"       : 0.,
    
    "cam2_pose_sideways"    : 5.,
    "cam2_pose_towards"     : 0.,
    "cam2_pose_angle"       : 0.
}

def data_from_parameters(params):
    """
    Return the data of a scenario defined by the given parameters.
    The data consists of the 3D pointcloud, and the 2 cameras.
    """
    
    if params["3D_points_source"] == "finite":
        points_3D = finite_3D_points(
                params["3D_points_r"],
                params["3D_points_x_on"], params["3D_points_y_on"], params["3D_points_z_on"] )
    elif params["3D_points_source"] == "infinite":
        points_3D = infinite_3D_points(
                params["3D_points_r"], params["3D_points_max_angle"],
                params["3D_points_x_on"], params["3D_points_y_on"] )
    elif params["3D_points_source"] == "scene":
        points_3D = scene_3D_points(params["3D_points_r"])
    
    cam1 = Camera()
    cam1.camera_pose(
            params["cam_pose_offset"],
            params["cam1_pose_sideways"], params["cam1_pose_towards"], params["cam1_pose_angle"] )
    
    cam2 = Camera()
    cam2.camera_pose(
            params["cam_pose_offset"],
            params["cam2_pose_sideways"], params["cam2_pose_towards"], params["cam2_pose_angle"] )
    
    for cam in (cam1, cam2):
        cam.camera_intrinsics(params["cam_resolution"], params["cam_k1"])
        cam.project_points(points_3D)
        cam.apply_noise(params["cam_noise_sigma"], params["cam_noise_discretized"])
    
    return points_3D, cam1, cam2

def cam_trajectory(traj_descr, cam_pose_offset, num_poses,
                   from_sideways=0., to_sideways=0., from_towards=0., to_towards=0., from_angle=0., to_angle=0.,
                   angle_by_sideways=False):
    """
    A trajectory description "traj_descr", as well as the camera -Z offset "cam_pose_offset" should be provided.
    Each trajectory/path consists of "num_poses" nodes.
    
    All "from_*" parameters result in a linear interpolation to the corresponding "to_*" parameters.
    See "Camera.camera_pose()" for the documentation of the "*" parameters.
    
    If "angle_by_sideways" is set to True,
        the "to_angle" parameter is determined by the intersection of the XZ circle centered at the origin
        with the plane at X="to_sideways",
        similar for the "from_angle" parameter (with the plane at X="from_sideways".
    """
    
    if angle_by_sideways:
        from_angle = asin(from_sideways / cam_pose_offset)
        to_angle = asin(to_sideways / cam_pose_offset)
        angle_values = np.linspace(from_angle, to_angle, num_poses)
        sideways_values = cam_pose_offset * np.sin(angle_values)
        towards_values = cam_pose_offset * (1 - np.cos(angle_values))
    else:
        sideways_values = np.linspace(from_sideways, to_sideways, num_poses)
        towards_values = np.linspace(from_towards, to_towards, num_poses)
        angle_values = np.linspace(from_angle, to_angle, num_poses)
    
    return { "traj_descr"        : traj_descr,
             "sideways_values"   : sideways_values,
             "towards_values"    : towards_values,
             "angle_values"      : angle_values }

def reset_random(delta_rseed=0):
    """
    Reset the random generator.
    This can be useful to reproduce the same results on different times,
    or to synchronize results on different times.
    
    Use "delta_rseed" to start from another random seed.
    """
    random.seed(rseed + delta_rseed)



""" Test cases """

num_trials = 10    # amount of samples taken for a random variable    # TODO: set this to 100
rseed = 123456789

robustness_thresh_max = 1.**2    # error distance threshold, if exceeded, it is better to discard the point
robustness_thresh_min = 1.**2    # error distance threshold, if not exceeded, it is better to keep the point

triangl_methods = [    # triangulation methods to test
    triangulation.linear_eigen_triangulation,
    triangulation.linear_LS_triangulation,
    triangulation.iterative_LS_triangulation,
    triangulation.polynomial_triangulation
]

num_poses = 40
max_sideways = 12.
max_towards = 12.
trajectories = [    # trajectories of 2nd cam
    # Trajectory 1
    cam_trajectory("From 1st cam, to sideways",
                   default_params["cam_pose_offset"], num_poses, to_sideways=max_sideways),
    # Trajectory 2
    cam_trajectory("From 1st cam, towards the sphere of points",
                   default_params["cam_pose_offset"], num_poses, to_towards=max_towards),
    # Trajectory 3
    cam_trajectory("From last pose of trajectory 1, towards the sphere of points, parallel to trajectory 2",
                   default_params["cam_pose_offset"], num_poses, from_sideways=max_sideways, to_sideways=max_sideways, to_towards=max_towards),
    # Trajectory 4
    cam_trajectory("From 1st cam, describing circle (while facing the sphere of points) until intersecting with trajectory 3",
                   default_params["cam_pose_offset"], num_poses, to_sideways=max_sideways, angle_by_sideways=True),
    # Trajectory 5
    cam_trajectory("From last pose of trajectory 4, describing circle (while facing the sphere of points) until 90 degrees",
                   default_params["cam_pose_offset"], num_poses, from_sideways=max_sideways, to_sideways=default_params["cam_pose_offset"], angle_by_sideways=True)
]

def test_1and2(trajectories, filename="test_1and2.mat"):
    """
    Effect of (2nd) camera configurations. (Test 2)
    & 
    Effect of 3D point positions. (Test 1)
    
    A list of 2nd cam trajectories "trajectories" should be provided,
    of which each item is generated by "cam_trajectory()".
    
    Test results are saved to a MATLAB-file with filename "filename".
    Relevant output variables of Test 2 include:
    - "err3D_mean_summary", "err3D_median_summary" : 3D error dist for each pose of each traj
    - "err2D_mean_summary", "err2D_median_summary" : 2D reproj error dist for each pose of each traj
    - "false_pos_summary", "false_neg_summary" : ratio of false positives/negatives, refering to triangl method's status-values
    - "units" : first three elems describe each dimension of the previously mentioned variables
    Relevant output variables of Test 1 include:
    - "p_err3D_mean_summary", "p_err3D_median_summary" : 3D error dist for each point, only for last pose of traj
    - "p_err3Dv_mean_summary", "p_err3Dv_covar_summary" : 3D error vect's mean vect and covar matrix for each point, only for last pose of traj
    - "units" : all elems except 2nd describe each dimension of the previously mentioned variables
    """
    params = dict(default_params)
    points_3D, cam1, cam2 = data_from_parameters(params)
    num_poses = len(trajectories[0]["sideways_values"])
    
    err3D_mean_summary, err3D_median_summary, err2D_mean_summary, err2D_median_summary, false_pos_summary, false_neg_summary = \
            np.zeros((6, len(trajectories), num_poses, len(triangl_methods)))
    p_err3D_mean_summary, p_err3D_median_summary = \
            np.zeros((2, len(trajectories), len(triangl_methods), len(points_3D)))
    p_err3Dv_mean_summary = np.zeros((len(trajectories), len(triangl_methods), len(points_3D), 3))
    p_err3Dv_covar_summary = np.zeros((len(trajectories), len(triangl_methods), len(points_3D), 3, 3))
    
    is_inside_view = True
    
    for ptci, trajectory in enumerate(trajectories):
        print "Performing trajectory id", ptci, "..."
        
        for pci, (sideways, towards, angle) in enumerate(zip(
                trajectory["sideways_values"], trajectory["towards_values"], trajectory["angle_values"] )):
            cam2.camera_pose(params["cam_pose_offset"], sideways, towards, angle)
            cam2.project_points(points_3D)
            
            errs3D, errs2D, statuses = [], [], []
            for ti in range(len(triangl_methods)):
                errs3D.append([])
                errs2D.append([])
                statuses.append([])
            
            reset_random()
            for trial in range(num_trials):
                cam1.apply_noise(params["cam_noise_sigma"], params["cam_noise_discretized"])
                cam2.apply_noise(params["cam_noise_sigma"], params["cam_noise_discretized"])
                
                #np.set_printoptions(threshold=5)
                #print cam2.points_2D
                #print np.min(cam2.points_2D[:, 0]), np.max(cam2.points_2D[:, 0]), np.min(cam2.points_2D[:, 1]), np.max(cam2.points_2D[:, 1])
                #print all(0 <= cam2.points_2D[:, 0]) and all(cam2.points_2D[:, 0] < params["cam_resolution"][0]) and all(0 <= cam2.points_2D[:, 1]) and all(cam2.points_2D[:, 1] < params["cam_resolution"][1])
                is_inside_view *= (
                        all(0 <= cam2.points_2D[:, 0]) and all(cam2.points_2D[:, 0] < params["cam_resolution"][0]) and 
                        all(0 <= cam2.points_2D[:, 1]) and all(cam2.points_2D[:, 1] < params["cam_resolution"][1]) )
                
                u1 = cam1.normalized_points()
                u2 = cam2.normalized_points()
                
                for ti, triangl_method in enumerate(triangl_methods):
                    start_timer()
                    points_3D_calc, status = triangl_method(u1, cam1.P, u2, cam2.P)
                    stop_timer()
                    errs3D[ti].append(error_vectors_3D(points_3D, points_3D_calc))
                    errs2D[ti] += error_vectors_2D(points_3D_calc, cam1, cam2)
                    statuses[ti].append(status)
            
            for ti in range(len(triangl_methods)):
                err3D_mean_summary[ptci, pci, ti], err3D_median_summary[ptci, pci, ti], errors = \
                        error_rms(errs3D[ti])
                err2D_mean_summary[ptci, pci, ti], err2D_median_summary[ptci, pci, ti], _ = \
                        error_rms(errs2D[ti])
                false_pos_summary[ptci, pci, ti], false_neg_summary[ptci, pci, ti] = \
                        robustness_stat(errors, statuses[ti])
                
                if pci == num_poses - 1:    # last pose
                    errors_partitioned = np.array(errs3D[ti])
                    p_err3D_mean_summary[ptci, ti], p_err3D_median_summary[ptci, ti] = \
                            zip(*map(error_rms, [errors_partitioned[:, i, :] for i in range(len(points_3D))]))[:2]
                    p_err3Dv_mean_summary[ptci, ti], p_err3Dv_covar_summary[ptci, ti] = \
                            vector_stat(errors_partitioned)
    
    if not is_inside_view:
        print "Warning: some points fell out of view."
    
    variables = {
            "err3D_mean_summary"    : err3D_mean_summary,
            "err3D_median_summary"  : err3D_median_summary,
            "err2D_mean_summary"    : err2D_mean_summary,
            "err2D_median_summary"  : err2D_median_summary,
            "false_pos_summary"     : false_pos_summary,
            "false_neg_summary"     : false_neg_summary,
            "p_err3D_mean_summary"  : p_err3D_mean_summary,
            "p_err3D_median_summary": p_err3D_median_summary,
            "p_err3Dv_mean_summary" : p_err3Dv_mean_summary,
            "p_err3Dv_covar_summary": p_err3Dv_covar_summary,
            "units"             : ["trajectory id", "node in a trajectory", "triangulation method", "point index"],
            "trajectories"      : trajectories,
            "triangl_methods"   : [m.func_name for m in triangl_methods],
            "points_3D"             : points_3D,
            "robustness_thresh_max" : robustness_thresh_max,
            "robustness_thresh_min" : robustness_thresh_min,
            "num_trials"            : num_trials,
            "rseed"                 : rseed,
            "default_params"        : default_params,
            "num_poses"             : num_poses,
            "max_sideways"          : max_sideways,
            "max_towards"           : max_towards }
    sio.savemat(filename, variables)

def test_3(trajectories, max_noise_sigma=4., num_noise_tests=40, filename="test_3.mat"):
    """
    Effect of noise (models). (Test 3)
    
    The following image-points perturbations are used:
    - Additive gaussian noise
    - Additive gaussian noise + discretization
    - Additive gaussian noise + discretization + radial distortion
    These tests will be executed for each last pose of "trajectories".
    
    "num_noise_tests" tests will be performed of normal distributed noise
    with sigma varying from 0 to "max_noise_sigma".
    
    Test results are saved to a MATLAB-file with filename "filename".
    Relevant output variables of Test 3 include:
    - "err3D_mean_summary", "err3D_median_summary" : 3D error dist for each pose of each traj
    - "err2D_mean_summary", "err2D_median_summary" : 2D reproj error dist for each pose of each traj
    - "false_pos_summary", "false_neg_summary" : ratio of false positives/negatives, refering to triangl method's status-values
    - "units" : the elems describe each dimension of the previously mentioned variables
    """
    params = dict(default_params)
    points_3D, cam1, cam2 = data_from_parameters(params)
    
    num_noise_types = 3
    err3D_mean_summary, err3D_median_summary, err2D_mean_summary, err2D_median_summary, false_pos_summary, false_neg_summary = \
            np.zeros((6, len(trajectories), num_noise_types, num_noise_tests, len(triangl_methods)))
    
    noise_sigma_values = np.linspace(0, max_noise_sigma, num_noise_tests)
    is_inside_view = True
    
    for ptci, trajectory in enumerate(trajectories):
        cam2.camera_pose(    # take last pose of each trajectory
                params["cam_pose_offset"],
                trajectory["sideways_values"][-1], trajectory["towards_values"][-1], trajectory["angle_values"][-1] )
        
        for ntyi in range(num_noise_types):
            print "Performing noise type id", ntyi, "..."
            
            if ntyi == 0:      # Additive gaussian noise
                noise_discretized = False
                cam_k1 = 0.
            elif ntyi == 1:    # Additive gaussian noise + discretization
                noise_discretized = True
                cam_k1 = 0.
            elif ntyi == 2:    # Additive gaussian noise + discretization + radial distortion
                noise_discretized = True
                cam_k1 = params["cam_k1"]
            
            for cam in (cam1, cam2):
                cam.camera_intrinsics(params["cam_resolution"], cam_k1)
                cam.project_points(points_3D)
            
            for nti, noise_sigma in enumerate(noise_sigma_values):
                errs3D, errs2D, statuses = [], [], []
                for ti in range(len(triangl_methods)):
                    errs3D.append([])
                    errs2D.append([])
                    statuses.append([])
            
                reset_random()
                for trial in range(num_trials):
                    cam1.apply_noise(noise_sigma, noise_discretized)
                    cam2.apply_noise(noise_sigma, noise_discretized)
                    
                    is_inside_view *= (
                            all(0 <= cam2.points_2D[:, 0]) and all(cam2.points_2D[:, 0] < params["cam_resolution"][0]) and 
                            all(0 <= cam2.points_2D[:, 1]) and all(cam2.points_2D[:, 1] < params["cam_resolution"][1]) )
                    
                    u1 = cam1.normalized_points()
                    u2 = cam2.normalized_points()
                
                    for ti, triangl_method in enumerate(triangl_methods):
                        start_timer()
                        points_3D_calc, status = triangl_method(u1, cam1.P, u2, cam2.P)
                        stop_timer()
                        errs3D[ti].append(error_vectors_3D(points_3D, points_3D_calc))
                        errs2D[ti] += error_vectors_2D(points_3D_calc, cam1, cam2)
                        statuses[ti].append(status)
                
                for ti in range(len(triangl_methods)):
                    err3D_mean_summary[ptci, ntyi, nti, ti], err3D_median_summary[ptci, ntyi, nti, ti], errors = \
                            error_rms(errs3D[ti])
                    err2D_mean_summary[ptci, ntyi, nti, ti], err2D_median_summary[ptci, ntyi, nti, ti], _ = \
                            error_rms(errs2D[ti])
                    false_pos_summary[ptci, ntyi, nti, ti], false_neg_summary[ptci, ntyi, nti, ti] = \
                            robustness_stat(errors, statuses[ti])
    
    if not is_inside_view:
        print "Warning: some points fell out of view."
    
    variables = {
            "err3D_mean_summary"    : err3D_mean_summary,
            "err3D_median_summary"  : err3D_median_summary,
            "err2D_mean_summary"    : err2D_mean_summary,
            "err2D_median_summary"  : err2D_median_summary,
            "false_pos_summary"     : false_pos_summary,
            "false_neg_summary"     : false_neg_summary,
            "units"             : ["id of last pose's trajectory", "noise type id", "noise sigma id", "triangulation method"],
            "trajectories"      : trajectories,
            "noise_type_descr"  : ["Add. gauss. noise", "Add. gauss. noise + discret.", "Add. gauss. noise + discret. + rad. distort. (barrel)"],
            "noise_sigma_values": noise_sigma_values,
            "triangl_methods"   : [m.func_name for m in triangl_methods],
            "points_3D"             : points_3D,
            "robustness_thresh_max" : robustness_thresh_max,
            "robustness_thresh_min" : robustness_thresh_min,
            "num_trials"            : num_trials,
            "rseed"                 : rseed,
            "default_params"        : default_params,
            "num_noise_tests"       : num_noise_tests,
            "max_noise_sigma"       : max_noise_sigma }
    sio.savemat(filename, variables)



from time import time
t = 0
ts = 0
def start_timer():
    global ts
    ts = time()
def stop_timer():
    global t
    t += time() - ts
def print_timer():
    print t

def main():
    rstate = random.get_state()
    reset_random()
    
    print "Running tests 1 and 2 ..."
    test_1and2(trajectories)
    
    print "Running test 3 ..."
    test_3(trajectories)
    
    print_timer()
    random.set_state(rstate)

if __name__ == "__main__":
    main()
