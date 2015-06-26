Usage
=====

Running the following will generate the results in "test_1and2.mat" and "test_3.mat":
$ ./triangulation_comparison.py

The trajectories are graphically represented in "trajectories.png".

The default 3D point distribution is the 'finite' point distribution,
also known as the "Sphere" distribution.

To generate the figures in "figures",
run the MATLAB/Octave script "visualize_tests.m".


Results
=======

"figures" and "figures_scene" contain the figures/results
on the 'finite' and 'scene' 3D point distributions, respectively.

The 'finite' point distribution is described in the "finite_3D_points()" function,
the 'scene' point distribution is described in the "scene_3D_points()" function.
Also see "points_finite.png" and "points_scene.png" respectively.

The scene used in the 'scene' point distribution originates
from the Point Cloud file "scene_3D_points.pcd",
and is generated with the Blender file "scene_3D_points.blend".
