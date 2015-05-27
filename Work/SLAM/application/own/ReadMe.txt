slam.py
=======

A first application to test feature-tracking.
Doesn't perform triangulation yet, and is very slow in tracking/matching (brute-force method).


slam2.py
========

Performs both pose estimation and map generation.

Run the following command in this directory to get more help and some example usages:
$ ./slam2.py --help

It is run in DEBUG mode by default.
To disable DEBUG mode, run with the "--use-debug=0" argument set.

During operation, use any key to proceed the program.
Some keys have special bindings, see the terminal output for details.
Avoid using the mouse in OpenCV's windows, it might disturb the key-events.


Running on the ICL_NUIM "livingroom" dataset, 4th trajectory
------------------------------------------------------------

Execute the example usage command corresponding with the ICL_NUIM "livingroom" dataset.

To run simultaneously with Blender,
use the "living_room.blend" file in "/Work/SLAM/datasets/ICL_NUIM/" and
execute the "live_results.py" script.
Each 30 frames the current camera trajectory and map will be saved and Blender will receive these updates.
Press ESCAPE in Blender to stop 'live-mode'.

Numerical evaluation of the results can be done,
first open a terminal in "/Work/SLAM/datasets/ICL_NUIM/living_room_traj3n_frei_png", then run:
$ ../../../tools/tum_benchmark_tools/evaluate_ate.py ./traj_groundtruth3.txt ./traj_out.cam0-slam2.txt --verbose --plot ./results_ate-slam2.pdf --plot_original > ./results_ate-slam2.txt
$ ../../../tools/tum_benchmark_tools/evaluate_rpe.py ./traj_groundtruth3.txt ./traj_out.cam0-slam2.txt --verbose --fixed_delta --plot ./results_rpe-slam2.pdf > ./results_rpe-slam2.txt


Running on the SVO dataset
--------------------------

Execute the example usage command corresponding with the SVO dataset.

Numerical evaluation of the results can be done,
first open a terminal in "/Work/SLAM/datasets/SVO/sin2_tex2_h1_v8_d/", then run:
$ ../../../tools/tum_benchmark_tools/evaluate_ate.py ./traj_groundtruth.txt ./traj_out.cam0-slam2.txt --verbose --plot ./results_ate-slam2.pdf --plot_original > ./results_ate-slam2.txt
$ ../../../tools/tum_benchmark_tools/evaluate_rpe.py ./traj_groundtruth.txt ./traj_out.cam0-slam2.txt --verbose --fixed_delta --plot ./results_rpe-slam2.pdf > ./results_rpe-slam2.txt

In case of bundle adjustment (see below):
$ ../../../tools/tum_benchmark_tools/evaluate_ate.py ./traj_groundtruth.txt ./traj_out.cam0-slam2-BA.txt --verbose --plot ./results_ate-slam2-BA.pdf --plot_original > ./results_ate-slam2-BA.txt
$ ../../../tools/tum_benchmark_tools/evaluate_rpe.py ./traj_groundtruth.txt ./traj_out.cam0-slam2-BA.txt --verbose --fixed_delta --plot ./results_rpe-slam2-BA.pdf > ./results_rpe-slam2-BA.txt


Bundle Adjustment (offline)
---------------------------

Make sure to have set the "--traj-out-file" and "--map-out-file" arguments,
see the example usages for more info.

Add the "-b slam2" argument to the command-line,
this will generate additional "BA_info.*.txt" files in the same directory
of where the trajectory file is saved.

You'll also need to create files describing the noise-models for the cameras and points,
see "/Work/SLAM/tools/bundle_adjustment/ReadMe.txt" for further instructions,
