Installation
============

First, make sure to have installed GTSAM,
this is explained in "/Draft/GTSAM-ReadMe.txt".

Then, run the following to compile the bundle-adjuster:
$ g++ bundle_adjust.cpp -lgtsam -lboost_system -lboost_filesystem -ltbb -ltbbmalloc -o bundle_adjust


Running
=======

Getting help
------------

Execute the following command to get a help/usage message:
$ ./bundle_adjust --help

Look in the "./example/" directory to get more info about the format of the input files.


Synthetic example
-----------------

Run the following command to generate the data (in memory) for this example,
and run BA on it: 
$ ./bundle_adjust  "./example" "synthetic" 2  1  1 1  0 1  2  1

Or run the following command to use the files containing synthetic data (also of this example),
and run BA on it: 
$ ./bundle_adjust  "./example" "synthetic" 2  1

The output will be "./example/traj_out.cam0-synthetic-BA.txt", "./example/traj_out.cam1-synthetic-BA.txt",
and "./example/map_out-synthetic-BA.pcd".

The output can be visualized using the "./example/visualize-BA.blend" file (in Blender).
In Blender, right-click on the script-window and select "Run script",
to load the original and bundle-adjusted trajectories and maps.
The first layer contains the original version, the second one contains the BA-ed version.
Tip: you can view multiple layers at the same time by holding SHIFT.
To load both versions, both layers should be activated.
The BA-ed camera is clearly less noisy than the original one.
Tip: use "View->Camera" or "View->Top" to switch between camera and top view, respectively,
and select a camera and use "View->Cameras->Set Active Object as Camera" to set the active camera.


Usage with SLAM algorithm
-------------------------

Go into the directory with the trajectories, map, and "BA_info.*.txt" files,
e.g. "/Draft/SLAM/datasets/ICL_NUIM/living_room_traj3n_frei_png", then run:
$ ../../../tools/bundle_adjustment/bundle_adjust  "." "slam2" 1  30  0

This will generate the "traj_out.cam0-slam2-BA.txt" and "map_out-slam2-BA.txt" files,
which can be opened with the Blender file "living_room.blend" in the parent directory.
See "/Draft/SLAM/datasets/ICL_NUIM/ReadMe.txt" for more info about the visualization.

Note: for the moment, the incremental approaches fail (underdetermined system exception)
on the input data of the slam2 algorithm.
For now, use the full optimization:
$ ../../../tools/bundle_adjustment/bundle_adjust  "." "slam2" 1  30  0 1  0 1  0
