slam.py
=======

A first application to test feature-tracking.
Doesn't perform triangulation yet, and is very slow in tracking/matching (brute-force method).


slam2.py
========

Performs both pose estimation and map generation.

Run "./slam2.py --help" in this directory to get more help and some example usages.

During operation, use any key to proceed the program.
Some keys have special bindings, see the terminal output for details.
Avoid using the mouse in OpenCV's windows, it might disturb the key-events.


Running on the ICL_NUIM "livingroom" dataset, 4th trajectory
------------------------------------------------------------

Execute the example usage command corresponding with the ICL_NUIM "livingroom" dataset.

To run simultaneously with Blender,
use the "living_room.blend" file in "/Draft/SLAM/datasets/ICL_NUIM/" and
execute the "live_results.py" script.
Each 30 frames the current camera trajectory and map will be saved and Blender will receive these updates.
Press ESCAPE in Blender to stop 'live-mode'.

Numerical evaluation of the results can be done,
first open a terminal in "/Draft/SLAM/datasets/ICL_NUIM/living_room_traj3n_frei_png", then run:
$ ../../../tools/tum_benchmark_tools/evaluate_ate.py ./livingRoom3n.gt.freiburg_exact ./traj_out-slam2.txt --verbose --plot ./results_ate-slam2.pdf > ./results_ate-slam2.txt
$ ../../../tools/tum_benchmark_tools/evaluate_rpe.py ./livingRoom3n.gt.freiburg_exact ./traj_out-slam2.txt --verbose --fixed_delta --plot ./results_rpe-slam2.pdf > ./results_rpe-slam2.txt
