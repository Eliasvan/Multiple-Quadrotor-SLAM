/* ----------------------------------------------------------------------------

 * Note: this file uses parts of the scenario used in "SFMdata.h"
 * of the GTSAM example files.
 * Credits should go to their respective authors.

 * -------------------------------------------------------------------------- */

/**
 * 8 points on a 10 meter cube,
 * captured by a robot (with a positive height) rotating around and facing towards the cube.
 * 
 * In case the parameter "nrCameras" is set to '2', another robot,
 * positioned symmetric to the first robot (negative height)
 * and with an offset angle of 45 degree around the Z axis,
 * adds additional measurements.
 */

#pragma once

#include <gtsam/linear/NoiseModel.h>

// And the required noise generators
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3DS2.h>

#include "DataStructures.hpp"

using namespace gtsam;

typedef PinholeCamera<Cal3DS2> DistortedCamera;


/* ************************************************************************* */


std::vector<double> generateDiagonalNoise(noiseModel::Diagonal::shared_ptr noise,
                                          boost::random::mt19937& rng,
                                          boost::random::normal_distribution<>& nd)
{
    std::vector<double> output;
    for (size_t i = 0; i < noise->dim(); ++i)
        output.push_back(noise->sigma(i) * nd(rng));
    return output;
}


std::vector<double> generateIsotropicNoise(noiseModel::Isotropic::shared_ptr noise,
                                           boost::random::mt19937& rng,
                                           boost::random::uniform_real_distribution<>& ud,
                                           boost::random::normal_distribution<>& nd)
{
    std::vector<double> output;
    double radius = noise->sigma() * nd(rng);
    double theta = ud(rng);
    if (noise->dim() == 2) {
        output.push_back(radius * cos(theta));
        output.push_back(radius * sin(theta));
    } else if (noise->dim() == 3) {
        double phi = ud(rng) / 2.;
        output.push_back(radius * cos(theta) * sin(phi));
        output.push_back(radius * sin(theta) * sin(phi));
        output.push_back(radius * cos(phi));
    } else
        throw std::invalid_argument("Only 2D or 3D Isotropic noise is supported.");
    return output;
}


/* ************************************************************************* */


BAdata createNoiseModelsAndPointsAndCamerasAndDataAndOdometry(const size_t nrCameras = 1, const size_t nrFrames = 20)
{
    BAdata data;
    
    // We only consider one camera (with multiple frames),
    // or two cameras in this example
    if ((nrCameras != 1) && (nrCameras != 2))
        throw std::invalid_argument("In this scenario, support for only 1 or 2 cameras is implemented.");
    
    size_t p, s, c, f, i;    // iter vars for respectively points3D, steps, cams, frames, other
    size_t nrSteps = nrFrames;
    
    // Initialize some variables with empty arrays, according to nrCameras
    for (c = 0; c < nrCameras; ++c) {
        data.odometryNoise.push_back(std::vector<noiseModel::Base::shared_ptr>());
        data.poses.push_back(std::vector< boost::shared_ptr<TrajectoryNode> >());
        data.points2D.push_back(std::vector< std::vector<Point2> >());
        data.point2D3DAssocs.push_back(std::vector< std::vector<Association2D3D> >());
    }
    
    
    /* Noise Models */
    
    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<> ud(0.0, 2*M_PI);    // uniform angle
    boost::random::normal_distribution<> nd(0.0, 1.0);    // standard normal distribution
    
    // Define the pose noise model, for all cams the same
    for (c = 0; c < nrCameras; ++c) {
        data.poseNoise.push_back(noiseModel::Diagonal::Sigmas(
                (Vector(6) << Vector3::Constant(0.02),Vector3::Constant(0.1)) )); // 0.02 rad std on roll,pitch,yaw; 10cm on x,y,z
    }
    
    // Define the (deltaPose) odometry noise model matrix, from a cam to another cam, all elements the same
    for (size_t c_from = 0; c_from < nrCameras; ++c_from) {
        for (size_t c_to = 0; c_to < nrCameras; ++c_to) {
            data.odometryNoise[c_from].push_back(noiseModel::Diagonal::Sigmas(
                    (Vector(6) << Vector3::Constant(0.05),Vector3::Constant(0.2)) ));
            //data.odometryNoise[c_from].push_back(noiseModel::Diagonal::Sigmas(
            //        (Vector(6) << Vector3::Constant(0.001),Vector3::Constant(0.1)) ));
        }
    }
    
    // Define the 3D point noise model
    data.point3DNoise = noiseModel::Isotropic::Sigma(3, 0.2);
    
    // Define the camera observation noise model, for all cams the same
    for (c = 0; c < nrCameras; ++c)
        data.point2DNoise.push_back(noiseModel::Isotropic::Sigma(2, 1.0)); // one pixel in u and v
    
    
    /* Calibration */
    
    // Create the camera calibration, for all cams the same
    for (c = 0; c < nrCameras; ++c)
        data.calibrations.push_back(boost::make_shared<Cal3DS2>(500.0, 500.0, 0.0, 640./2, 480./2, 0.0, 0.0));
    
    
    /* Measurements */
    
    // Create the set of ground-truth landmarks
    size_t nrPoints = 8;
    data.points3D.push_back(MapPoint(10.0,10.0,10.0));
    data.points3D.push_back(MapPoint(-10.0,10.0,10.0));
    data.points3D.push_back(MapPoint(-10.0,-10.0,10.0));
    data.points3D.push_back(MapPoint(10.0,-10.0,10.0));
    data.points3D.push_back(MapPoint(10.0,10.0,-10.0));
    data.points3D.push_back(MapPoint(-10.0,10.0,-10.0));
    data.points3D.push_back(MapPoint(-10.0,-10.0,-10.0));
    data.points3D.push_back(MapPoint(10.0,-10.0,-10.0));
    //data.points3D.push_back(MapPoint(0.0,0.0,0.0));
    
    size_t nrPointsInit = 4;   // this many points will have known 3D positions in the first step
    double height = 10.0;
    double radius = 40.0;
    double theta = 0.0;
    Point3 up(0, 0, 1);
    Point3 target(0, 0, 0);
    DistortedCamera cameras[2], cameras_prev[2];
    for (s = 0; s < nrSteps; ++s) {
        // Specify when and which new 3D points were added;
        // in this example, after the second step every point has been triangulated
        data.point3DAddedIdxs.push_back(std::vector<size_t>());
        if (s == 0)    // first step
            for (p = 0; p < nrPointsInit; ++p)
                data.point3DAddedIdxs[s].push_back(p);
        else if (s == 1)    // second step
            for (p = nrPointsInit; p < nrPoints; ++p)
                data.point3DAddedIdxs[s].push_back(p);
        
        // Create 2D points, 2D -> 3D associations, and poses
        for (c = 0; c < nrCameras; ++c) {
            // Create ground-truth camera; if two cams used, give them another position
            Point3 position;
            if (c == 0)
                position = Point3(radius * cos(theta), radius * sin(theta), height);
            else if (c == 1)
                position = Point3(radius * cos(theta + M_PI/4), radius * sin(theta + M_PI/4), -height);
            cameras[c] = DistortedCamera::Lookat(position, target, up, *data.calibrations[c]);
            
            // Create 2D points and the associations with their corresponding triangulated 3D points
            data.point2D3DAssocs[c].push_back(std::vector<Association2D3D>());
            data.points2D[c].push_back(std::vector<Point2>());
            if (s == 0) {
                // In the first step, only the initPoints (containing prior factors)
                // in the first frame are already triangulated
                for (p = 0; p < nrPointsInit; ++p) {
                    Association2D3D assoc;
                    assoc.frame = f = 0;
                    assoc.point2D = data.points2D[c][f].size();
                    assoc.point3D = p;
                    data.point2D3DAssocs[c][s].push_back(assoc);
                    data.points2D[c][f].push_back(cameras[c].project(data.points3D[p].p));
                }
            } else {
                if (s == 1) {
                    // In the second step, we first add points (other than the initPoints) to the first frame
                    for (p = nrPointsInit; p < nrPoints; ++p) {
                        Association2D3D assoc;
                        assoc.frame = f = 0;
                        assoc.point2D = data.points2D[c][f].size();
                        assoc.point3D = p;
                        data.point2D3DAssocs[c][s].push_back(assoc);
                        data.points2D[c][f].push_back(cameras_prev[c].project(data.points3D[p].p));
                    }
                }
                // Then all points of the current frame are added to the current frame
                for (p = 0; p < nrPoints; ++p) {
                    Association2D3D assoc;
                    assoc.frame = f = s;
                    assoc.point2D = data.points2D[c][f].size();
                    assoc.point3D = p;
                    data.point2D3DAssocs[c][s].push_back(assoc);
                    data.points2D[c][f].push_back(cameras[c].project(data.points3D[p].p));
                }
            }
            
            // Intentionally add perturbation to image points, off ground-truth
            for (i = 0; i < data.point2D3DAssocs[c][s].size(); ++i) {
                Association2D3D assoc = data.point2D3DAssocs[c][s][i];
                f = assoc.frame;
                std::vector<double> point2DNoise = generateIsotropicNoise(
                        boost::dynamic_pointer_cast<noiseModel::Isotropic>(data.point2DNoise[c]), rng, ud, nd );
                data.points2D[c][f][assoc.point2D] = data.points2D[c][f][assoc.point2D].compose(
                        Point2(point2DNoise[0], point2DNoise[1]) );
            }
            
            // Create ground-truth poses, at frame 'f'
            // (added an offset of 1, since trajectory in TUM format starts with frame '1')
            f = s;    // here, a frame corresponds with a step
            data.poses[c].push_back(boost::make_shared<TrajectoryNode>(cameras[c].pose(), 1 + f));
            
            // Save camera
            cameras_prev[c] = cameras[c];
        }
        theta += 2*M_PI / nrFrames;
        
        // Calculate odometry
        {
        data.odometry.push_back(std::vector<Pose3>());
        data.odometryAssocs.push_back(std::vector<AssociationOdo>());
        
        // Between different frames of same camera
        for (c = 0; c < nrCameras; ++c) {
            if (s == 0)
                continue;    // the first step doesn't have delta-poses
            AssociationOdo assoc;
            assoc.from_cam = c;
            assoc.from_frame = s - 1;
            assoc.to_cam = c;
            assoc.to_frame = s;
            data.odometryAssocs[s].push_back(assoc);
            data.odometry[s].push_back(Pose3(
                    data.poses[assoc.from_cam][assoc.from_frame]->p.between(
                    data.poses[assoc.to_cam][assoc.to_frame]->p ) ));
        }
        
        // Between same frame of different cameras
        if (nrCameras == 2) {
            // e.g. stereo vision
            f = s;    // here, a frame corresponds with a step
            AssociationOdo assoc;
            assoc.from_cam = 0;
            assoc.from_frame = f;
            assoc.to_cam = 1;
            assoc.to_frame = f;
            data.odometryAssocs[s].push_back(assoc);
            data.odometry[s].push_back(Pose3(
                    data.poses[assoc.from_cam][assoc.from_frame]->p.between(
                    data.poses[assoc.to_cam][assoc.to_frame]->p ) ));
        }
            
        // Intentionally add perturbation to odometry, off ground-truth
        for (i = 0; i < data.odometryAssocs[s].size(); ++i) {
            AssociationOdo assoc = data.odometryAssocs[s][i];
            std::vector<double> odometryNoise = generateDiagonalNoise(
                    boost::dynamic_pointer_cast<noiseModel::Diagonal>(data.odometryNoise[assoc.from_cam][assoc.to_cam]),
                    rng, nd );
            data.odometry[s][i] = data.odometry[s][i].compose(Pose3(
                    Rot3::rodriguez(odometryNoise[0], odometryNoise[1], odometryNoise[2]),
                    Point3(odometryNoise[3], odometryNoise[4], odometryNoise[5]) ));
        }
        }
        
        // Intentionally add perturbation to pose, off ground truth
        f = s;    // here, a frame corresponds with a step
        for (c = 0; c < nrCameras; ++c) {
            //data.poses[c][f]->p = 
            //        data.poses[c][f]->compose(Pose3(Rot3::rodriguez(-0.0004, 0.0007, 0.001), Point3(0.05, -0.10, 0.20)));
            std::vector<double> poseNoise = generateDiagonalNoise(
                    boost::dynamic_pointer_cast<noiseModel::Diagonal>(data.poseNoise[c]),
                    rng, nd );
            data.poses[c][f]->p = data.poses[c][f]->p.compose(Pose3(
                    Rot3::rodriguez(poseNoise[0], poseNoise[1], poseNoise[2]),
                    Point3(poseNoise[3], poseNoise[4], poseNoise[5]) ));
        }
    }
    
    // Intentionally add perturbation to landmarks, off ground truth, except <nrPointsInit> first points
    for (p = nrPointsInit; p < nrPoints; ++p) {
        std::vector<double> point3DNoise = generateIsotropicNoise(
                boost::dynamic_pointer_cast<noiseModel::Isotropic>(data.point3DNoise), rng, ud, nd );
        data.points3D[p].p = data.points3D[p].p.compose(
                Point3(point3DNoise[0], point3DNoise[1], point3DNoise[2]) );
    }
    
    return data;
}
