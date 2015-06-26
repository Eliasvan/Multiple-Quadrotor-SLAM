Calibration Tools
=================

Usage
-----

"calibrate.py" is an interactive tool to calibrate a camera,
and much more,
such as image undistortion and real-time pose estimation.
To get more help, run the program and read the help message.

Run the program with:
$ ./calibrate.py


Data
----

Chessboard calibration images for different types of cameras have been provided:
- "chessboards" : C170 webcam
- "chessboards_blender" : Camera in Blender file "calibrate_pose_visualization.blend"
- "chessboards_front" and "chessboards_bottom" : front and bottom camera of the AR.Drone2.0 using the live profile (lowres)


Results
-------

Contains:
- Calibration results for all of the above mentioned data.
- "chessboards_extrinsic" contains the results of real-time pose estimation.


Visualization
-------------

Blender file "calibrate_pose_visualization.blend" is used for visualizing camera poses,
and used for generating chessboard images.

The Blender file is also used for analyzing results of the "triangl_pose_est_interactive" option,
outputting relative pose estimations and triangulated points.
Some demos of using this latter option in "calibration.py",
are shown in the "triangl_pose_est_interactive_demo_*.txt" files.
