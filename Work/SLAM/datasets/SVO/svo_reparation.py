#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function    # Python 3 compatibility

import os

import sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "python_libs"))
import dataset_tools



def join_path(*path_list):
    """Convenience function for creating OS-indep relative paths."""
    return os.path.relpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), *path_list))


def main():
    """
    The quaternions of the groundtruth will be normalized,
    and saved to another filename.
    """

    print ("Normalizing quaternions of groundtruth camera trajectory...")
    
    dataset_tools.save_cam_trajectory_TUM(
            join_path("sin2_tex2_h1_v8_d", "traj_groundtruth.txt"),
            dataset_tools.load_cam_trajectory_TUM(
                    join_path("sin2_tex2_h1_v8_d", "trajectory_nominal.txt") ) )

    print ("Done.")

if __name__ == "__main__":
    main()
