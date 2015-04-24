Installation
============

Download and extract the "sin2_tex2_h1_v8_d/img" directory and "sin2_tex2_h1_v8_d/trajectory_nominal.txt" file from
"http://rpg.ifi.uzh.ch/datasets/sin2_tex2_h1_v8_d.tar.gz" to "/Draft/SLAM/datasets/SVO/sin2_tex2_h1_v8_d/".

Then run the following (to normalize the trajectory's quaternions):
$ ./svo_reparation.py

For some SLAM algorithms, it may be required to generate initialization data,
such as the first pose and some visible 3D points.
To generate this data, run:
$ ./svo_initialization.py
