Equipment
_________

AR-Drone 2



Links of additional ROS modules used (covered in the tutorial, see further)
____________________________________

https://github.com/AutonomyLab/ardrone_autonomy    (uses AR-Drone SDK version 2.0, when building)
https://github.com/mikehamer/ardrone_tutorials
https://github.com/ros-drivers/joystick_drivers



Tutorial
________

I started from these sites to fly the AR-Drone2 in ROS (indigo, latest version till now 2014-10):
    http://robohub.org/up-and-flying-with-the-ar-drone-and-ros-getting-started/
    http://robohub.org/up-and-flying-with-the-ar-drone-and-ros-joystick-control/
    Note: Ignore the instructions under the sections starting with:
        "If you don't have Linux, want to get flying quickly or don't know what I'm talking about:"

(I should note that I compiled ROS from source, and don't use the distribution packages.
 Therefore my directory structure looks like:
 ~/ros_catkin_ws/
    ~/ros_catkin_ws/build-isolated/
    ~/ros_catkin_ws/devel-isolated/
    ~/ros_catkin_ws/install-isolated/   (here is my local install; I have to run "source ~/ros_catkin_ws/install_isolated/setup.bash" to setup environment variables)
        ~/ros_catkin_ws/install-isolated/bin/
        ~/ros_catkin_ws/install-isolated/lib/
        ~/ros_catkin_ws/install-isolated/share/
    ~/ros_catkin_ws/src/
)

The installation-instruction of the first link needed some changes in order to work,
I will repeat the steps and provide info if something has to be changed:

---

$ sudo rosdep init
$ rosdep update

The following two commands didn't work for my install:
    $ sudo apt-get install ros-fuerte-joystick-drivers
    $ rosdep install joy
I did the following:
    1) Copy all packages from "https://github.com/ros-drivers/joystick_drivers" to "~/ros_catkin_ws/src/".
    2) $ cd ~/ros_catkin_ws
        $ rosdep install --from-paths src --ignore-src --rosdistro indigo -y
        $ ./src/catkin/bin/catkin_make_isolated --source src/joystick_drivers --install -DCMAKE_BUILD_TYPE=Release
        $ ./src/catkin/bin/catkin_make_isolated --source src/joy --install -DCMAKE_BUILD_TYPE=Release
        $ ./src/catkin/bin/catkin_make_isolated --source src/ps3joy --install -DCMAKE_BUILD_TYPE=Release
        $ ./src/catkin/bin/catkin_make_isolated --source src/spacenav_node --install -DCMAKE_BUILD_TYPE=Release
        $ ./src/catkin/bin/catkin_make_isolated --source src/wiimote --install -DCMAKE_BUILD_TYPE=Release

$ sudo apt-get install daemontools libudev-dev libiw-dev

Instead of:
    $ roscd
    $ pwd
    ~/ros_workspace
I did:
    $ cd ~/ros_catkin_ws/src

$ git clone https://github.com/AutonomyLab/ardrone_autonomy.git
$ git clone https://github.com/mikehamer/ardrone_tutorials.git
$ ls
...  ardrone_autonomy  ardrone_tutorials  ...
$ rospack profile

And again, the following commands didn't work for my install:
    $ roscd ardrone_auto<TAB should autocomplete>
    $ ./build_sdk.sh
    $ rosmake -a
I did the following:
    $ cd ~/ros_catkin_ws
    $ rosdep install --from-paths src --ignore-src --rosdistro indigo -y
    $ ./src/catkin/bin/catkin_make_isolated --source src/ardrone_autonomy --install -DCMAKE_BUILD_TYPE=Release
    $ ./src/catkin/bin/catkin_make_isolated --source src/ardrone_tutorials --install -DCMAKE_BUILD_TYPE=Release

---

That's it, now you should be able to run:
    $ roslaunch ardrone_tutorials keyboard_controller.launch
to control/test the drone (but first connect to the drone's wireless accesspoint).



Additions of my "ardrone_tutorials" version to the standard package (see "/Draft/ARDrone2Tests/ros_tools")
____________________________________________________________________

Contents:
    - joystick-controller mapping for the PS3-controller, see "joystick_controller.launch" for further comments
    - "joystick_controller_takephoto.launch": version that does the front-bottom switching and takes pictures of them,
            useful for relative pose calibration
    - "joystick_controller_takevideo.launch": version that records a raw image sequence,
            useful to use the output for further testing of algorithms, without having to fly again,
            no frames will be skipped due to encoding, and timestamps are recorded as well

If you only want to test the joystick-part (not the "takephoto" or "takevideo" part), do:
    $ roslaunch ardrone_tutorials joystick_controller.launch



Cam Calibration
_______________

https://github.com/AutonomyLab/ardrone_autonomy#how-can-i-calibrate-the-ardrone-frontbottom-camera



Notes
_____

The following ardrone video profile:
    MP4_360P_H264_720P_CODEC
    (= Live stream with MPEG4.2 soft encoder. Record stream with H264 hardware encoder in 720p mode.)
seems to be the default:
    - live video feed: 640x360 for both front and bottom cam (it seems like some vertical pixels are sacrificed for the bottom cam)
    - USB video record: 1280x720 for both front and bottom cam (now there are two unused black vertical bands of width 160px on both sides of the frame for the bottom cam)

The following ardrone video profile:
    H264_720P_CODEC
    (= Live stream with H264 hardware encoder configured in 720p mode. No record stream.)
doesn't seem to work (tested with "ardrone_navigation" (SDK), "ardrone_autonomy" (ROS) and "python-ardrone").
