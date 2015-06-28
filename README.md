Multiple-Quadrotor-SLAM
=======================

Introduction
------------

This is the repository for the code of my thesis (2014-2015) at KULeuven.


Contents
--------

"/Work/python_libs/":
All globally used Python modules.

"/Work/calibration/":
Camera calibration tools and other related stuff.

"/Work/SLAM/":
The actual SLAM algorithm, datasets, and tools.

"/Work/triangulation_comparison/":
Generation of synthetic data,
used for analysing various triangulation methods and camera configurations.

"/Work/ARDrone2_tests/":
Tests of using the AR.Drone2.0.

"/Work/c2py_example/":
Example usage of the "/Work/python_libs/convert_c_to_ext_lib.py" module.


Dependencies
------------

- Python 2
- Numpy
- Scipy
- OpenCV 2

Other dependencies are described in the ReadMe files.
Blender is also required to visualize the "*.blend" files.


Notes
-----

Most of the code is written in Python,
and was tested using the Python 2 interpreter,
hence it's recommended to use Python 2 for most scripts.
This is due to the current dependency of OpenCV 2 on Python 2.
However, the code is written such that it should be compatible with Python 3
with no or minor modifications.
