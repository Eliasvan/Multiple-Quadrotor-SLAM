#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function    # Python 3 compatibility

import os
import numpy as np

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "python_libs"))
import dataset_tools



def parse_cmd_args():
    import argparse
    
    # Create parser object and help messages
    parser = argparse.ArgumentParser(
            description=
            "Transform camera trajectories and pointclouds by the transformation "
            "between the source (from) and groundtruth (to) camera trajectory, "
            "such that the first pose of the source trajectory matches the one of the groundtruth trajectory. "
            "The groundtruth trajectory will not be transformed. "
            'The transformed output files will have the suffix "-trfm".')
    
    parser.add_argument("groundtruth_trajectory",
                        help="filepath of the groundtruth camera trajectory, in TUM format")
    parser.add_argument("source_trajectory",
                        help="filepath of the source camera trajectory, in TUM format")
    
    parser.add_argument("-t", "--extra-input-trajectories", dest="extra_input_trajectories",
                        action="append", nargs='?',
                        help="filepath of another to-be-transformed camera trajectory, in TUM format")
    parser.add_argument("-m", "--input-maps", dest="input_maps",
                        action="append", nargs='?',
                        help="filepath of a to-be-transformed 3D map, in PCD format (pointcloud)")
    
    parser.add_argument("-f", "--at-frame", dest="at_frame",
                        type=int, default=1,
                        help="frame number with which first pose is associated "
                             "(default: 1)")
    parser.add_argument("-o", "--offset-time", dest="offset_time",
                        type=float, default=float("inf"),
                        help="if nonzero, this float will be used as timestamp offset from the first pose "
                             "to estimate the scale transformation as well "
                             "(default: inf)")
    
    # Parse arguments
    args = parser.parse_args()
    traj_to_file, traj_from_file, traj_input_files, map_input_files, at_frame, offset_time = \
            args.groundtruth_trajectory, args.source_trajectory, args.extra_input_trajectories, args.input_maps, args.at_frame, args.offset_time
    if traj_input_files == None: traj_input_files = []
    if map_input_files == None: map_input_files = []
    
    # Add the source trajectory to the to-be-transformed camera trajectories
    traj_input_files.insert(0, traj_from_file)
    
    return traj_to_file, traj_from_file, traj_input_files, map_input_files, at_frame, offset_time


def main():
    traj_to_file, traj_from_file, traj_input_files, map_input_files, at_frame, offset_time = parse_cmd_args()
    
    print ("Calculating transformation...")
    cam_trajectory_from = dataset_tools.load_cam_trajectory_TUM(traj_from_file)
    cam_trajectory_to = dataset_tools.load_cam_trajectory_TUM(traj_to_file)
    transformation = dataset_tools.transform_between_cam_trajectories(
            cam_trajectory_from, cam_trajectory_to, at_frame=at_frame, offset_time=offset_time )
    
    print ("Results:")
    delta_quaternion, delta_scale, delta_location = transformation
    print ("delta_quaternion:")
    print ("\t %s" % delta_quaternion)
    print ("delta_scale:")
    print ("\t %s" % delta_scale)
    print ("delta_location:")
    print ("\t %s" % delta_location)
    print ()
    
    for traj_input_file in traj_input_files:
        print ('Transforming traj "%s"...' % traj_input_file)
        dataset_tools.save_cam_trajectory_TUM(
                "%s-trfm%s" % tuple(os.path.splitext(traj_input_file)),
                dataset_tools.transformed_cam_trajectory(
                        dataset_tools.load_cam_trajectory_TUM(traj_input_file),
                        transformation ) )
    
    for map_input_file in map_input_files:
        print ('Transforming map "%s"...' % map_input_file)
        points, colors, _ = dataset_tools.load_3D_points_from_pcd_file(map_input_file, use_alpha=True)
        dataset_tools.save_3D_points_to_pcd_file(
                "%s-trfm%s" % tuple(os.path.splitext(map_input_file)),
                dataset_tools.transformed_points(points, transformation),
                colors )
    
    print ("Done.")

if __name__ == "__main__":
    main()
