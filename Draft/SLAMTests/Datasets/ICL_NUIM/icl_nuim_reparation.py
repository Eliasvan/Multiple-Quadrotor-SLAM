#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np

import sys; sys.path.append(os.path.join("..", "..", "..", "PythonLibraries"))
import transforms as trfm
import dataset_tools



""" File- import functions """

def load_cam_poses_POV(filename):
    """
    Load bash-script of PovRay commands, used to generate the ground-truth camera trajectories,
    and hence as accurate as you can get.
    There is no official format, but an example of a single line reads as follows:
    
    "/home/ahanda/povray.3.7.0.rc3.withdepthmap/unix/povray +Iliving_room.pov +Oscene_1202.png +W640 +H480 
    + Declare=val00=-0.965935 + Declare=val01=0.0127836 + Declare=val02=0.25847+ Declare=val10=0.0206222 
    + Declare=val11=0.999405 + Declare=val12=0.0276384+ Declare=val20=-0.257963 + Declare=val21=0.0320271 
    + Declare=val22=-0.965624+ Declare=val30=0.107827  + Declare=val31=1.21817 + Declare=val32=0.374226 
    +FN16 +wt1 -d +L/home/ahanda/povray.3.7.0.rc3.withdepthmap/include + Declare=use_baking=2 +A0.0"
    
    The camera pose projection matrices are returned.
    
    Note: a typical filename of the ICL NUIM dataset compatible with this function, is:
    "livingroomlcmlog-2013-08-07.03.posesRenderingCommands.sh", which can be found in
    "http://www.doc.ic.ac.uk/~ahanda/living_room_code.tgz".
    """
    Ps = []
    
    lines = open(filename, 'r').read().split('\n')
    for line in lines:
        if not line:
            continue
        
        M = np.eye(4)
        M[0:3, 0:4] = np.array([float(i[3:i.find('+')]) for i in line.split("Declare=val")[1:]]).reshape(4, 3).T
        
        # We take the inverse,
        # to obtain the trfm matrix that projects points from the world axis-system to the camera axis-system
        P = trfm.P_inv(M)
        Ps.append(P)
    
    return Ps


""" File-conversion functions, mostly to fix up files """

def mirror_wavefront_obj_file(filename_in, filename_out):
    """
    Mirror the X-coordinates of the given Wavefront .OBJ-file "filename_in",
    and save result to "filename_out".
    
    Some background info:
    PoseRay seems to invert the X-coordinates of an original Wavefront .OBJ-file,
    when converting to PovRay geometry.
    This is why the camera calibration matrix of the ICL NUIM dataset has a negative Y focal length,
    to compensate for the mirrored 3D scene.
    (2 successive mirrors result in zero mirror (+ possibly a rotation transformation))
    
    Warning: only the vertices are mirrored, the face-order is not modified,
    therefore it's discouraged to use the output for rendering.
    """
    lines = open(filename_in, 'r').read().split('\n')
    
    for i, line in enumerate(lines):
        words = line.split(' ')
        if words[0] in ("v", "vn"):
            words[1] = str(-float(words[1]))    # invert X-coordinate
            lines[i] = ' '.join(words)
    
    open(filename_out, 'w').write('\n'.join(lines))

def repair_ICL_NUIM_cam_trajectory(filename_in, filename_out,
                                   initial_location=None,
                                   rebuild_timestamps=True, delta_timestamp=0., fps=30):
    """
    After this repair, you will be able to use the output in conjunction with
    the original (i.e. non-mirrored) 3D scene of the dataset,
    provided that you don't apply the negative Y-factor of the camera calibration matrix.
    
    "filename_in" : filename of the input camera trajectory
    "filename_out" : filename of the repaired output camera trajectory
    "initial_location" : (optional) location of the first camera pose
        Note that all ICL NUIM camera trajectories are undetermined up to an unknown translation.
        To find the correct "initial_location", you can do the following:
            Ps = load_cam_poses_POV("traj1.posesRenderingCommands.sh")
            _, locs_exact, _ = dataset_tools.convert_cam_poses_to_cam_trajectory_TUM(Ps)
            initial_location = [-locs_exact[0][0], locs_exact[0][1], locs_exact[0][2]]
    "rebuild_timestamps" :
        (optional) if True, regenerate timestamps starting from start-time "delta_timestamp",
        at a rate of "fps"
    
    The new "timestps", "locations", "quaternions" will be returned.
    
    Note: this is not the recommended way to go,
    for accuracy, consider using "load_cam_poses_POV()" instead.
    """
    timestps, locations, quaternions = dataset_tools.load_cam_trajectory_TUM(filename_in)
    
    if initial_location:
        dlx, dly, dlz = np.array(initial_location) - np.array(locations[0])
    else:
        dlx = dly = dlz = 0
    
    if rebuild_timestamps:
        timestps = list(delta_timestamp + np.arange(len(timestps)) / float(fps))
    
    for i, (location, quaternion) in enumerate(zip(locations, quaternions)):
        lx, ly, lz = location
        locations[i] = [lx + dlx, ly + dly, -lz + dlz]
        
        qx, qy, qz, qw = quaternion
        quaternions[i] = [qw, qz, qy, -qx]
    
    dataset_tools.save_cam_trajectory_TUM(filename_out, timestps, locations, quaternions)
    return timestps, locations, quaternions



def main():
    """
    The exact trajectories will be extracted, along with the repaired 3D model.
    For now only the "Living Room Dataset" is supported.
    """
    
    print "Mirroring 3D model..."
    
    mirror_wavefront_obj_file(
            os.path.join("living_room_obj_mtl", "living-room.obj"),
            os.path.join("living_room_obj_mtl", "living-room_Xmirrored.obj") )
    
    print "Extracting exact ground-truth camera trajectories..."
    
    pov_bash_script_filenames = [
            "livingroomlcmlog-2013-08-07.00.posesRenderingCommands.sh",
            "livingroomlcmlog-2013-08-07.01.posesRenderingCommands_copy.sh",
            "livingroomlcmlog-2013-08-07.02.posesRenderingCommands.sh",
            "livingroomlcmlog-2013-08-07.03.posesRenderingCommands.sh" ]
    
    for i, pov_bash_script_filename in enumerate(pov_bash_script_filenames):
        dataset_tools.save_cam_trajectory_TUM(
                os.path.join("living_room_traj%sn_frei_png" % i, "livingRoom%sn.gt.freiburg_exact" % i),
                *dataset_tools.convert_cam_poses_to_cam_trajectory_TUM(load_cam_poses_POV(
                        os.path.join("living_room_code", pov_bash_script_filename) )))
    
    if "--repair-noisy-trajectories" in sys.argv:
        print "Repairing (noisy) camera trajectories..."
        
        for i, pov_bash_script_filename in enumerate(pov_bash_script_filenames):
            _, locs_exact, _ = dataset_tools.convert_cam_poses_to_cam_trajectory_TUM(
                    load_cam_poses_POV(os.path.join("living_room_code", pov_bash_script_filename)) )
            repair_ICL_NUIM_cam_trajectory(
                    os.path.join("living_room_traj%sn_frei_png" % i, "livingRoom%sn.gt.freiburg" % i),
                    os.path.join("living_room_traj%sn_frei_png" % i, "livingRoom%sn.gt.freiburg_repaired" % i),
                    [-locs_exact[0][0], locs_exact[0][1], locs_exact[0][2]] )
    
    print "Done."

if __name__ == "__main__":
    main()
