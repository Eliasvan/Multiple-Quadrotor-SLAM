#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from math import tan, pi
import numpy as np
import cv2

import sys; sys.path.append("../PythonLibraries")
import transforms as trfm
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

""" Camera configurations """

class Camera:

    def camera_intrinsics(self, resolution, k1=0):
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

    def camera_pose(self, offset, sideways=0, towards=0, angle=0):
        """
        Generate 3x4 projection matrix of a camera initially
        with center at (0, 0, -"offset") with unit orientation (along +Z; Y-axis down)
        and transformed by a translation (+"sideways", 0, +"towards") and by a local rotation of "angle" around Y-axis.
        """
        rvec = (0, angle, 0)
        tvec = (sideways, 0, -offset + towards)
        
        R = cv2.Rodrigues(rvec)[0]
        tvec = R.dot(np.array(tvec).reshape(3, 1))
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
        
        noise = [random.normalvariate(0, sigma) for i in range(self.points_2D_exact.size)]
        points_2D = self.points_2D_exact + np.array(noise).reshape(self.points_2D_exact.shape)
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



""" Default scenario parameters, generators and random seeds """

default_parameters = {
    "finite_3D_points"      : True,
    "3D_points_r"           : 4,
    "3D_points_max_angle"   : pi / 4,
    "3D_points_x_on"        : True,
    "3D_points_y_on"        : True,
    "3D_points_z_on"        : True,
    
    "cam_resolution"        : (640, 480),
    "cam_k1"                : 0.3,
    "cam_pose_offset"       : 40,
    "cam_noise_sigma"       : 0.8,    # based on (99.7% of the) data of footage with rolling shutter
    "cam_noise_discretized" : True,
    
    "cam1_pose_sideways"    : 0,
    "cam1_pose_towards"     : 0,
    "cam1_pose_angle"       : 0,
    
    "cam2_pose_sideways"    : 5,
    "cam2_pose_towards"     : 0,
    "cam2_pose_angle"       : 0
}

def data_from_parameters(params):
    """
    Return the data of a scenario defined by the given parameters.
    The data consists of the 3D pointcloud, and the 2 cameras.
    """
    
    if params["finite_3D_points"]:
        points_3D = finite_3D_points(
                params["3D_points_r"],
                params["3D_points_x_on"], params["3D_points_y_on"], params["3D_points_z_on"] )
    else:
        points_3D = infinite_3D_points(
                params["3D_points_r"], params["3D_points_max_angle"],
                params["3D_points_x_on"], params["3D_points_y_on"] )
    
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

rseed = 123456789

def reset_random(delta_rseed=0):
    """
    Reset the random generator.
    This can be useful to reproduce the same results on different times,
    or to synchronize results on different times.
    
    Use "delta_rseed" to start from another random seed.
    """
    random.seed(rseed + delta_rseed)



""" Error evaluation functions """

def errors_3D(points_3D_exact, points_3D_calc):
    """
    Return the squared euclidean errors between each point of "points_3D_exact" and "points_3D_calc".
    The 3D points of "points_3D_calc" must not have homogeneous coordinates.
    
    This function only works for finite coordinates.
    """
    
    # Convert "points_3D_exact" from 4D to 3D (assumes weighting of 1.0)
    return np.sum((points_3D_calc - points_3D_exact[:, 0:3])**2, axis=1)

def errors_2D(points_3D_calc, cam1, cam2):
    """
    Return the squared euclidean errors between each point of "cam1" and "points_3D_calc".
    """
    if points_3D_calc.shape[1] == 3:    # convert to homogeneous coordinates, if needed
        points_3D_calc = np.concatenate((points_3D_calc, np.ones((len(points_3D_calc), 1))), axis=1)
    
    cam1_points_2D_calc = cam1.project_points(points_3D_calc, False)
    cam2_points_2D_calc = cam2.project_points(points_3D_calc, False)
    
    cam1_errors_2D = np.sum((cam1_points_2D_calc - cam1.points_2D_exact)**2, axis=1)
    cam2_errors_2D = np.sum((cam2_points_2D_calc - cam2.points_2D_exact)**2, axis=1)
    
    return cam1_errors_2D, cam2_errors_2D

def error_rms(*errors):
    """
    Return the root mean 
       and the root median
    of all (squared) errors.
    """
    if type(errors) == tuple:
        errors = np.concatenate(errors)
    
    return np.mean(errors), np.median(errors)



""" Test cases """

num_trials = 1    # amount of samples taken for a random variable    # TODO: set this to 100
triangl_methods = [    # triangulation methods to test
    triangulation.linear_LS_triangulation
]

def test1():
    """
    Effect of 3D point positions.
    """
    pass

def test2(sideways=19, towards=13, angle=pi/8, num_poses=100):
    """
    Effect of camera configurations.
    
    "sideways" determines the maximum 'sideway' movement of the second camera.
    "towards" determines the maximum 'towards' movement of the second camera.
    "angle" determines the maximum Y rotation of the second camera.
    The path between zero and "sideways" or "towards", consists of "num_poses" nodes.
    """
    params = dict(default_parameters)
    points_3D, cam1, cam2 = data_from_parameters(params)
    
    sideways_values = np.linspace(sideways, 0, num_poses)
    towards_values = np.linspace(0, towards, num_poses)
    angle_values = np.linspace(0, angle, num_poses)
    
    num_pose_configs = 4
    err3D_summary = np.zeros((num_pose_configs, num_poses, len(triangl_methods)))
    err2D_summary = np.zeros((num_pose_configs, num_poses, len(triangl_methods)))
    
    def execute_pose_config(ptci, pci, sideways, towards, angle):
        """
        "ptci": pose trajectory config iteration variable
        "pci": pose config iteration variable
        """
        
        cam2.camera_pose(params["cam_pose_offset"], sideways, towards, angle)
        cam2.project_points(points_3D)
        
        err3D = []
        err2D = []
        for ti in range(len(triangl_methods)):
            err3D.append([])
            err2D.append([])
        
        reset_random()
        for trial in range(num_trials):
            cam1.apply_noise(params["cam_noise_sigma"], params["cam_noise_discretized"])
            cam2.apply_noise(params["cam_noise_sigma"], params["cam_noise_discretized"])
            
            #np.set_printoptions(threshold=5)
            #print cam2.points_2D
            #print np.min(cam2.points_2D[:, 0]), np.max(cam2.points_2D[:, 0]), np.min(cam2.points_2D[:, 1]), np.max(cam2.points_2D[:, 1])
            #print all(0 <= cam2.points_2D[:, 0]) and all(cam2.points_2D[:, 0] < 640) and all(0 <= cam2.points_2D[:, 1]) and all(cam2.points_2D[:, 1] < 480)
            
            u1 = cam1.normalized_points()
            u2 = cam2.normalized_points()
            
            for ti, triangl_method in enumerate(triangl_methods):
                points_3D_calc, status = triangl_method(u1, cam1.P, u2, cam2.P)
                err3D[ti].append(errors_3D(points_3D, points_3D_calc))
                err2D[ti] += errors_2D(points_3D_calc, cam1, cam2)
        
        for ti in range(len(triangl_methods)):
            err3D_mean, err3D_median = error_rms(*err3D[ti])
            err2D_mean, err2D_median = error_rms(*err2D[ti])
            
            err3D_summary[ptci, pci, ti] = err3D_median
            err2D_summary[ptci, pci, ti] = err2D_median
    
    towards = angle = 0
    for pci, (sideways) in enumerate(sideways_values):
        execute_pose_config(0, pci, sideways, towards, angle)
    
    angle = 0
    for pci, (sideways, towards) in enumerate(zip(sideways_values, towards_values)):
        execute_pose_config(1, pci, sideways, towards, angle)
    
    towards = 0
    for pci, (sideways, angle) in enumerate(zip(sideways_values, angle_values)):
        execute_pose_config(2, pci, sideways, towards, angle)
    
    for pci, (sideways, towards, angle) in enumerate(zip(sideways_values, towards_values, angle_values)):
        execute_pose_config(3, pci, sideways, towards, angle)
    
    # TODO: save "err3D_summary" and "err2D_summary"
        

def test3():
    """
    Effect of noise (models).
    """
    pass



def main():
    rstate = random.getstate()
    reset_random()
    test1()
    test2()
    test3()
    random.setstate(rstate)

if __name__ == "__main__":
    main()
