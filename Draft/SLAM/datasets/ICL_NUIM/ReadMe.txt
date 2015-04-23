Intro
=====

This document describes the steps needed to examine
the ICL NUIM dataset (http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html),
and how to use them in conjunction with SLAM algorithms.

Only the "Living Room Dataset" is used, because it contains an explicit 3D model.


Installation
============

3D Model
--------

Download the archive http://www.doc.ic.ac.uk/~ahanda/VaFRIC/living_room_obj_mtl.tar.gz
and extract the directory "living_room_obj_mtl" to this directory.
Only the "living-room.obj" file will be used.

Later on we will have to 'repair' this file,
because the images were rendered based on the same model mirrored about the X-axis.

Trajectories
------------

There are 4 trajectories:
- "lr kt0"
- "lr kt1"
- "lr kt2"
- "lr kt3"    <-- most interesting one, contains a loop
We will use the last one, but the steps for the other trajectories are analogous.

Download the archive http://www.doc.ic.ac.uk/~ahanda/living_room_code.tgz
and extract the directory "living_room_code" to this directory.
Only the following files will be used:
- "livingroomlcmlog-2013-08-07.00.posesRenderingCommands.sh"
- "livingroomlcmlog-2013-08-07.01.posesRenderingCommands_copy.sh"
- "livingroomlcmlog-2013-08-07.02.posesRenderingCommands.sh"
- "livingroomlcmlog-2013-08-07.03.posesRenderingCommands.sh"
These correspond to the 4 trajectories respectively.

In the next section, these trajectories will be converted to the TUM camera trajectory format.

Note that we did not download the TUM camera trajectory files from ICL NUIM,
this is because of 3 reasons:
- they are undetermined up to an unknown translation
- they are noisy compared to the exact trajectory, hence naming this 'ground truth' is not valid
- they are transformed in an non-intuitive way
However, if you really want to use them,
add "--repair-noisy-trajectories" as a command-line argument to the "icl_nuim_reparation.py" script.
In this case you will have to use
- the original (i.e. non-mirrored) 3D model
- positive focal lengths for the camera calibration matrix

Images
------

We'll only use the noisy images.
From the ICL NUIM website, the "TUM RGB-D Compatible PNGs with noise" download link should be followed.

These will link to the following URLs:
- http://www.doc.ic.ac.uk/~ahanda/living_room_traj0n_frei_png.tar.gz
- http://www.doc.ic.ac.uk/~ahanda/living_room_traj1n_frei_png.tar.gz
- http://www.doc.ic.ac.uk/~ahanda/living_room_traj2n_frei_png.tar.gz
- http://www.doc.ic.ac.uk/~ahanda/living_room_traj3n_frei_png.tar.gz
These correspond to the 4 trajectories respectively.

For each archive, create a the directory "living_room_trajXn_frei_png" in this directory,
and extract the content to that directory. ("X" can be 0, 1, 2, or 3, referring to the trajectory id)


Repair and Conversion
=====================

Execute the "icl_nuim_reparation.py" script, this will generate:
- "living_room_obj_mtl/living-room_Xmirrored.obj"    = 3D model
- "living_room_traj0n_frei_png/traj_groundtruth0.txt"    = ground-truth camera trajectory 0 in TUM format  
- "living_room_traj1n_frei_png/traj_groundtruth1.txt"    = ground-truth camera trajectory 1 in TUM format
- "living_room_traj2n_frei_png/traj_groundtruth2.txt"    = ground-truth camera trajectory 2 in TUM format
- "living_room_traj3n_frei_png/traj_groundtruth3.txt"    = ground-truth camera trajectory 3 in TUM format


Visualization in Blender
========================

The .blend-file is already provided in "living_room.blend",
but the steps to obtain the crucial parts are described here.
(Tested with Blender v2.69)

3D Model
--------

Starting from an empty Blender file (remove the existing objects),
select "Import Wavefront (.obj)" to import the 3D model "living_room_obj_mtl/living-room_Xmirrored.obj".
Before confirming to import, make sure the following import-settings are set:
- "Forward": "Y Forward"
- "Up": "Z Up"
Then save the .blend-file to this directory.

Trajectories
------------

Now we will import camera trajectory 3,
first switch to the "Scripting" screen layout, instead of the "Default" screen layout.
Create a new text file inside Blender, and paste the following code:

import bpy
import sys; sys.path.append(bpy.path.abspath("//../../../python_libs"))
import blender_tools
# Load the camera trajectory and create such camera trajectory
blender_tools.load_and_create_cam_trajectory(bpy.path.abspath(
        "//living_room_traj3n_frei_png/traj_groundtruth3.txt" ))

Then right-click on the code and select "Run Script".
The camera trajectory is now created,
but since the dataset's camera's Y-focal length is negative, we will have to take this into account:
make the camera's local Y-scale negative.

Switch back to the "Default" screen layout, this will allow to use the timeline for quick navigation.
Switch to Camera view to view inside the camera.
The LEFT and RIGHT arrow keys can also be used for navigation.

Images
------

Click on the photo-camera icon (Render) in the SpaceProperties tab,
and adjust the "Resolution" to X: 640, Y: 480.
Additionally, you could set the "Frame Rate" to 30 fps,
and set the "End Frame" to the number of images (1241 for trajectory 3).

In the 3D View, switch to wireframe mode (press 'Z'),
then open up the Properties tab (press 'N'),
scroll down and enable the "Background Images" tab and set "Axis" to "Camera".

Now click on "Add Image", then click the folder icon to load all images
in "living_room_traj3n_frei_png/rgb/" (press 'A' to select all png images).

Set "Source" to "Image Sequence",
"Frames" to the number of .png-files (1241 in case of trajectory 3),
select "Front" (this will draw the images on-top in case of 3D View is in Solid render mode).
Use the "Start" slider to align the images with the trajectories in time (2 seems to be perfect for trajectory 3).

To adjust the 'zoom' of the camera, click on the camera icon in the SpaceProperties tab,
and adjust the "Focal Length" until the image matches the geometry (24.012 seems to be good for trajectory 3).

If desired, you can go back to Solid render mode (press 'Z').
Press Alt-A to play the entire sequence.
Don't forget to save.


Deployment of a SLAM algorithm
==============================

This is specific for each SLAM algorithm (and described in their ReadMe files),
but some algorithms may need to have additional data to initialize the algorithm.

In "living_room.blend", there is a script "extract_init.py" that takes care of this.
Other useful scripts include "load_gndtrth.py", "show_result.py" and "live_results.py".
