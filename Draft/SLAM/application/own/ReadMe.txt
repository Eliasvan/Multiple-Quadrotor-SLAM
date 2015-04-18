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

To run together with Blender on the ICL_NUIM "livingroom" dataset,
use the "living_room.blend" file in "/Draft/SLAM/datasets/ICL_NUIM/" and
execute the "live_results.py" script.
Each 30 frames the current camera trajectory and map will be saved and Blender will receive these updates.
Press ESCAPE in Blender to stop 'live-mode'.
