SVO without ROS
===============


Installation    (local install)
------------

Make sure BOOST, Eigen 3, Suitesparse, OpenCV, Qt4, QGLViewer, and their development packages, are installed.

Open a terminal and create a directory SVO, e.g. in your home-directory:
$ mkdir SVO
$ cd SVO

We will now build and install SVO and 3th party libs to a local install path:
$ mkdir install_path
$ export SVO_INSTALL_PATH=`pwd`/install_path

First we build 3th party lib "Sophus":
$ git clone https://github.com/strasdat/Sophus.git
$ cd Sophus
$ git checkout a621ff
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=$SVO_INSTALL_PATH ..
$ make -j4
$ make install
$ cd ../..

To save and learn the install path to the compiler, and to export the library path:
$ echo "SVO_INSTALL_PATH=$SVO_INSTALL_PATH; export SVO_INSTALL_PATH" >> ~/.bashrc
$ echo "CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:\$SVO_INSTALL_PATH/include; export CPLUS_INCLUDE_PATH" >> ~/.bashrc
$ echo "LIBRARY_PATH=\$LIBRARY_PATH:\$SVO_INSTALL_PATH/lib; export LIBRARY_PATH" >> ~/.bashrc
$ echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$SVO_INSTALL_PATH/lib; export LD_LIBRARY_PATH" >> ~/.bashrc
$ bash

Then we build 3th party lib "fast" (not needed anymore if you have built it before):
$ git clone https://github.com/uzh-rpg/fast.git
$ cd fast
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ../..

Then we build 3th party lib "g2o":
$ git clone https://github.com/RainerKuemmerle/g2o.git
$ cd g2o
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=$SVO_INSTALL_PATH ..
$ make -j4
$ make install
$ cd ../..

Then we build 3th party tools "vikit_common":
$ git clone https://github.com/uzh-rpg/rpg_vikit.git
In "./rpg_vikit/vikit_common/CMakeLists.txt" set the flag USE_ROS to FALSE.
$ cd rpg_vikit/vikit_common
$ mkdir build
$ cd build
$ LDFLAGS="-lSophus" cmake ..
$ make -j4
$ cd ../../..

Finally we build "SVO":
$ git clone https://github.com/uzh-rpg/rpg_svo.git
In "./rpg_svo/svo/CMakeLists.txt" set the flag USE_ROS to FALSE, and HAVE_G2O to TRUE,
and add the following code at the bottom:
    ADD_EXECUTABLE(run_pipeline test/run_pipeline.cpp)
    TARGET_LINK_LIBRARIES(run_pipeline svo)
Then copy the file "run_pipeline.cpp" into directory "./rpg_svo/svo/test/".
$ cd rpg_svo/svo
$ mkdir build
$ cd build
$ LDFLAGS="-lSophus -lboost_filesystem" G2O_ROOT=$SVO_INSTALL_PATH cmake ..
$ make -j4
$ cd ../../..


Running on the "sin2_tex2_h1_v8_d" dataset
------------------------------------------

Open a terminal in directory "/Draft/SLAM/datasets/SVO/sin2_tex2_h1_v8_d", then run:
$ $SVO_INSTALL_PATH/../rpg_svo/svo/bin/run_pipeline  ./img/ ./traj_out-SVO.txt ./map_out-SVO.pcd  50  752 480  315.5 315.5 376.0 240.0

The results can be visualized using the "visualize.blend" file, and running the embedded Python script.


Running on the ICL_NUIM "livingroom" dataset, 4th trajectory
------------------------------------------------------------

Warning: I had to:
    - change "px_ref_.size() < 100" to "px_ref_.size() < 88" in "rpg_svo/svo/src/initialization.cpp"
    - change the "quality_min_fts" parameter from 50 to 25 in "rpg_svo/svo/src/config.cpp"
And even then the trajectory was not continuously tracked, and the map is almost non-existent.

Open a terminal in directory "/Draft/SLAM/datasets/ICL_NUIM/living_room_traj3n_frei_png/", then run:
$ $SVO_INSTALL_PATH/../rpg_svo/svo/bin/run_pipeline  ./rgb/ ./traj_out-SVO.txt ./map_out-SVO.pcd  30  640 480  481.20 -480.00 319.50 239.50
