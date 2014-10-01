TODO
====

- [x] apply mask to goodFeauturesToTrack() -> no converge ratio-test

- [ ] solvePnp() between non-keyframes

- [x] in case of bad frames, if #extrapolatedOFpoints < threshold

- [ ] triangulate (first write calibration-tests program),
  & pose estimation

- [ ] draw distances, axisSystem, random color of triangl point (for all points of one pass)
 + relate to cloudpoints (instead of reindexing)

- [ ] bundle adjustment between (50) keyframes
 + again solvePnp() between non-keyframes

- [ ] visualize cloudpoints

- [ ] Write LATEX Chapter 1


Other ideas
-----------

(- Use 'sub-keyframes': frames where the average OF vector between each 'sub-keyframe' is large enough (instead of when a non-degenerate fundamental matrix is detected)
  => is_keyframe is checked between a 'sub-keyframe' (call it 'sub-base') and the current frame, when it is True, the new 'sub-base' is selected as the next 'sub-keyframe'  that has matches of not-yet triangulated points with the current frame)

- 3D-3D correspondence, if same points are visible again.
  => maybe implement a tree-based representation of 3D space, to allow fast logarithmic point-lookup


Possible Problems
=================

- Number of points to estimate pose becomes too small

- No fundamental matrix can be found (because of not enough non-planar geometry)

- Solving pose and doing triangulation between different cameras
  => different K matrices


Two quadrotors with multiple on-board fixed cameras
===================================================

- Calibrate each camera seperately

- Calibrate the relative fixed poses of the on-board cameras of a quadrotor, and choose a reference-camera (which will view the single chessboard at deploy-time)
  => at calibration-time, multiple chessboards will be used, placed at known 3D positions, solvePnp() is used for each camera, and the average relative poses will be saved

- At deploy-time, the quadrotors start in such a way that the reference-camera of each quadrotor has the single chessboard in sight (which will serve as the world axis-system, and to eliminate the up-to-scale degree of freedom)

- During flight, each camera of a quadrotor is used to calculate an initial pose estimate and triangulation in the same way as "slam2.py" does (= monocular stereo)

- Then the pose of the reference-camera of each quadrotor is computed as a weighted version of the pose estimations (see previous step) of all cameras of that quadrotor (the earlier calibrated relative fixed poses are used to transform them to the reference-camera)
  => weighting is based on the reprojection error of each camera; if the reprojection error is too high (or a frame is rejected), the weighting is set to '0'
  => then the new pose of each camera on the quadrotor is refined (by using the above calculated weighted reference-camera pose and the calibrated relative fixed camera poses)
  => the total reprojection error (monoculars) for each quadrotor (weighted in the same way) can then also be calculated (see two steps further)

- Then the relative pose and triangulation are calculated (by calculating the rich feature descriptors of the keypoints, then matching them (along an estimate of the epipolar line) and then finding the essential matrix) between each set of corresponding cameras of the two quadrotors (= real stereo), only if both cameras in this set don't have a rejected frame
  => if reprojection error is not reasonable, the results of this camera set are simply rejected, otherwise ...
  => the triangulation result (of not-yet triangulated points of previous frame) overwrites the initial mocular one, and the corresponding colors and patch orientations are also calculated
  => the relative pose is saved for this set of cameras, along with the reprojection error (see next step)

- Then the final relative pose is calculated as a weighted version of the non-rejected relative poses between each set of corresponding cameras of the two quadrotors (see previous step)
  => weighting is based on the reprojection error of each relative pose calculation
  => then the new pose of the reference-camera on each quadrotor is refined (by using the above calculated final relative pose, and the original reference-camera pose of both quadrotors weighted by the total reprojection error (monoculars) for each quadrotor (see two steps back))
  => as a result all the other camera poses of the quadrotor can be refined (because of the calibrated fixed camera poses)

- Each time new points are triangulated, 3D-3D correspondences with earlier already-triangulated points should be checked, to link the matches together, and to increase pose estimation accuracy

- Bundle adjustment for all last 50 keyframes,
also across different cameras? 
