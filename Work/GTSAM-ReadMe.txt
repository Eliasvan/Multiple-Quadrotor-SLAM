Installation    (local install)
============

Make sure BOOST version 1.43 or higher is installed.

Open a terminal and change to the top-level GTSAM source tree directory,
of the tarball (see https://collab.cc.gatech.edu/borg/gtsam/) that you unpacked to e.g. your home-directory.
We will now build and install GTSAM to a local install path:
$ mkdir install_path
$ export GTSAM_INSTALL_PATH=`pwd`/install_path
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=$GTSAM_INSTALL_PATH ..
$ make -j4 check
$ make install

To save and learn the install path to the compiler, and to export the library path:
$ echo "GTSAM_INSTALL_PATH=$GTSAM_INSTALL_PATH; export GTSAM_INSTALL_PATH" >> ~/.bashrc
$ echo "CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:\$GTSAM_INSTALL_PATH/include; export CPLUS_INCLUDE_PATH" >> ~/.bashrc
$ echo "LIBRARY_PATH=\$LIBRARY_PATH:\$GTSAM_INSTALL_PATH/lib; export LIBRARY_PATH" >> ~/.bashrc
$ echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$GTSAM_INSTALL_PATH/lib; export LD_LIBRARY_PATH" >> ~/.bashrc
$ bash


Usage
=====

Compiling a program that uses GTSAM    (e.g. "OdometryExample" from the tutorial)
-----------------------------------

$ g++ -lgtsam -lboost_system -ltbb -ltbbmalloc -o OdometryExample OdometryExample.cpp
$ ./OdometryExample


Example of usage in Python    (using an extension module)
--------------------------

In "/Work/python_libs/gtsam/" the python gtsam binding will be available.
(TODO)


Example of usage in Python    (using Weave from Scipy) (not recommended)
--------------------------

>>> from scipy import weave
>>> code = """
... NonlinearFactorGraph graph;
... 
... Pose2 priorMean(0.0, 0.0, 0.0);
... noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas((Vector(3) << 0.3, 0.3, theta_sigma));
... graph.add(PriorFactor<Pose2>(1, priorMean, priorNoise));
... 
... graph.print("Factor Graph:\\n");
... """
>>> theta_sigma = 0.15
>>> res = weave.inline(code, ["theta_sigma"], support_code="using namespace gtsam;", headers=["<gtsam/nonlinear/NonlinearFactorGraph.h>", "<gtsam/slam/PriorFactor.h>", "<gtsam/geometry/Pose2.h>"], libraries=["gtsam"])
Factor Graph:
size: 1
factor 0: PriorFactor on 1
  prior mean: (0, 0, 0)
  noise model: diagonal sigmas[0.3; 0.3; 0.15];
