// Compile with: g++ bundle_adjust.cpp -lgtsam -lboost_system -lboost_filesystem -ltbb -ltbbmalloc -o bundle_adjust

/* ----------------------------------------------------------------------------

 * Note: this file uses parts of the code and comments used in "VisualISAM2Example.cpp"
 * of the GTSAM example files.
 * Credits should go to their respective authors.

 * -------------------------------------------------------------------------- */

#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam/nonlinear/ISAM2.h>

// Initially, the first two poses will be optimized
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <vector>
#include <fstream>

#include "DataStructures.hpp"
#include "IO.hpp"
#include "GenerateData.hpp"

using namespace gtsam;

// Set this to 1 to print extensive BA info (incremental graph and values) per step
#define DEBUG 0


/* ************************************************************************* */


bool validateDataSufficientlyConstrainted(BAdata& data, bool useOdometry)
{
    bool valid = true;
    
    size_t p, s, c, f, i;
    size_t nrPoints = data.points3D.size();
    size_t nrSteps = data.point3DAddedIdxs.size();
    size_t nrCameras = data.calibrations.size();
    size_t nrFrames = nrSteps;
    
    size_t num_unknowns = 0;
    size_t num_constraints = 0;
    // TODO: A global "num_constraints" does not take linkage of constraints with unknowns into account;
    //       this means that if the function doesn't fail on validation,
    //       some unknowns may still be underdetermined (and others will be overdetermined).
    //       If the function does fail on validation, the system is guaranteed to be underdetermined.
    
    // Lists indicating existance and count, for each 3D point in the graph
    std::vector<bool> point3DIsEstimated(nrPoints, false);
    std::vector<size_t> point3DCounter(nrPoints, 0);
    
    // Lists indicating existance and count, for each pose in the graph
    std::vector< std::vector<bool> > poseIsEstimated;
    std::vector< std::vector<size_t> > poseCounter;
    for (c = 0; c < nrCameras; ++c) {
        poseIsEstimated.push_back(std::vector<bool>(nrFrames, false));
        poseCounter.push_back(std::vector<size_t>(nrFrames, 0));
    }
    
    for (s = 0; s < nrSteps; ++s) {
        
        /* Values */
        
        // Simulate adding new 3D points estimates to the graph
        for (i = 0; i < data.point3DAddedIdxs[s].size(); ++i) {
            p = data.point3DAddedIdxs[s][i];
            assert (!point3DIsEstimated[p]);    // 3D point can only be added once
            point3DIsEstimated[p] = true;
            num_unknowns += 3;    // 3D point has 3 DOFs/unknowns
        }
        
        // Simulate adding new (only valid) pose estimates to the graph
        for (c = 0; c < nrCameras; ++c) {
            f = s;    // here, a frame corresponds with a step
            if (data.poses[c][f]) {
                poseIsEstimated[c][f] = true;
                num_unknowns += 6;    // pose has 6 DOFs/unknowns
            }
        }
        
        /* Factors */
        
        // Simulate adding prior factors to the graph
        if (s == 0) {
            for (c = 0; c < nrCameras; ++c) {
                // Add a prior on the first pose
                f = s;    // here, a frame corresponds with a step
                assert (poseIsEstimated[c][f]);
                ++poseCounter[c][f];
                num_constraints += 6;    // prior on pose adds 6 constraints

                // Add priors on first landmarks
                for (i = 0; i < data.point2D3DAssocs[c][s].size(); ++i) {
                    assert (data.point2D3DAssocs[c][s][i].frame == f);    // all should lay in the same frame (= first)
                    p = data.point2D3DAssocs[c][s][i].point3D;
                    assert (point3DIsEstimated[p]);
                    ++point3DCounter[p];
                    num_constraints += 3;    // prior on 3D point adds 3 constraints
                }
            }
        }
        
        // Simulate adding projective factors to the graph
        for (c = 0; c < nrCameras; ++c) {
            for (i = 0; i < data.point2D3DAssocs[c][s].size(); ++i) {
                Association2D3D assoc = data.point2D3DAssocs[c][s][i];
                
                // This pose is part of a projective factor
                f = assoc.frame;
                assert (poseIsEstimated[c][f]);
                ++poseCounter[c][f];
                
                // This 3D point is also part of the current projective factor
                p = assoc.point3D;
                assert (point3DIsEstimated[p]);
                ++point3DCounter[p];
                
                num_constraints += 2;    // 2D projection adds 2 constraints
            }
        }
        
        // Simulate adding odometry factors to the graph, if desired
        if (useOdometry) {
            for (i = 0; i < data.odometryAssocs[s].size(); ++i) {
                AssociationOdo assoc = data.odometryAssocs[s][i];
                
                // This pose is part of an odometry factor
                c = assoc.from_cam;
                f = assoc.from_frame;
                assert (poseIsEstimated[c][f]);
                ++poseCounter[c][f];
                
                // This pose is also part of the current odometry factor
                c = assoc.to_cam;
                f = assoc.to_frame;
                assert (poseIsEstimated[c][f]);
                ++poseCounter[c][f];
                
                num_constraints += 6;    // odometry adds 6 constraints (including scale)
            }
        }
        
        /* Constraint Checks */
        
        // Factor count for 3D points (for efficiency: only need to check the ones that were added in this step)
        for (i = 0; i < data.point3DAddedIdxs[s].size(); ++i)
           assert (point3DCounter[data.point3DAddedIdxs[s][i]] >= 2);    // not very useful, but testing anyway
        
        // Factor count for poses (for efficiency: only need to check the ones that were added in this step)
        for (c = 0; c < nrCameras; ++c) {
           f = s;    // here, a frame corresponds with a step
           if (data.poses[c][f])
               assert (poseCounter[c][f] >= 1);    // not very useful, but testing anyway
        }
        
        // Check num_unknowns vs num_constraints
        if (num_unknowns > num_constraints) {
            valid = false;
            std::cout << "Warning: num_unknowns (" << num_unknowns
                      << ") > num_constraints (" << num_constraints
                      << ") at step " << s << std::endl;
        }
    }
    
    return valid;
}


/* ************************************************************************* */


char poseChar(size_t c)
{
    // First cam has character 'c', second cam 'd', and so on
    return 'c' + c;
}


Values performBundleAdjustment(BAdata& data,
                               bool useOdometry, bool fullOptimizeAtSecondPoints3DBatch, ushort iSAM_version)
{
    if (!( (0 <= iSAM_version) && (iSAM_version <= 2) ))
        throw std::invalid_argument("iSAM_version should be '0', '1', or '2'.");
    
    bool needsFullOptimize = (fullOptimizeAtSecondPoints3DBatch || !iSAM_version);
    
    // Create an iSAM1 object.
    int relinearizeInterval = 1;
    NonlinearISAM isam1(relinearizeInterval);
    
    // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps to maintain proper linearization
    // and efficient variable ordering, iSAM2 performs partial relinearization/reordering at each step. A parameter
    // structure is available that allows the user to set various properties, such as the relinearization threshold
    // and type of linear solver. For this example, we we set the relinearization threshold small so the iSAM2 result
    // will approach the batch result.
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    ISAM2 isam2(parameters);

    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    Values initialEstimate, currentEstimate;
    
    size_t p, s, c, f, i;    // iter vars for respectively points3D, steps, cams, frames, other
    size_t nrPoints = data.points3D.size();
    size_t nrSteps = data.point3DAddedIdxs.size();
    size_t nrCameras = data.calibrations.size();
    size_t nrFrames = nrSteps;

    // Check whether the first pose of some of the cameras is known
    if (nrCameras < 1)
        throw std::invalid_argument("Number of cameras should be minimal 1.");
    else {
        for (c = 0; c < nrCameras; ++c)
            if (data.poses[c][0])
                break;
        if (c == nrCameras)
            throw std::invalid_argument("The first frame of at least one camera should contain a valid pose.");
    }
    
    std::cout << "Running ";
    if (iSAM_version == 0)
        std::cout << "full optimization (Levenberg-Marquardt)";
    else if (iSAM_version == 1)
        std::cout << "iSAM1";
    else if (iSAM_version == 2)
        std::cout << "iSAM2";
    std::cout << " on " << nrPoints << " 3D points and "
                        << nrCameras << " camera(s) with each " 
                        << nrFrames << " frames." << std::endl << std::endl;
    
    // Loop over the different steps, adding the observations to iSAM incrementally
    for (s = 0; s < nrSteps; ++s) {
        
        std::cout << "Step " << s << std::endl;
        if (!data.point3DAddedIdxs[s].empty())
            std::cout << "    adding a batch of new 3D points (" << data.point3DAddedIdxs[s].size() << ")" << std::endl;
        
        /* Values */
        
        // Add an initial guess for the current poses, if valid
        for (c = 0; c < nrCameras; ++c)
            if (data.poses[c][s])
                initialEstimate.insert(Symbol(poseChar(c), s), data.poses[c][s]->p);
        
        // Add initial guess for each landmark observation as necessary
        for (i = 0; i < data.point3DAddedIdxs[s].size(); ++i) {
            p = data.point3DAddedIdxs[s][i];
            initialEstimate.insert(Symbol('p', p), data.points3D[p].p);
        }
        
        /* Factors (priors) */

        // If this is the first step, add a prior on the first pose to set the coordinate frame
        // and a prior on the first landmarks to set the scale
        if (s == 0) {
            for (c = 0; c < nrCameras; ++c) {
                // Add a prior on valid poses at the first frame
                f = s;    // here, a frame corresponds with a step
                if (data.poses[c][f]) {
                    graph.add(PriorFactor<Pose3>(Symbol(poseChar(c), f), data.poses[c][f]->p, data.poseNoise[c]));
                }

                // Add priors on first landmarks
                for (i = 0; i < data.point2D3DAssocs[c][s].size(); ++i) {
                    p = data.point2D3DAssocs[c][s][i].point3D;
                    graph.add(PriorFactor<Point3>(Symbol('p', p), data.points3D[p].p, data.point3DNoise));
                }
            }
        }
        
        /* Factors (non-priors) */

        // For each camera,
        // for each observation in the current step (can be in different frames),
        // add visual measurement factors in frame "f" at 2D point "point2D", pointing to 3D point (index) "p"
        for (c = 0; c < nrCameras; ++c) {
            for (i = 0; i < data.point2D3DAssocs[c][s].size(); ++i) {
                Association2D3D assoc = data.point2D3DAssocs[c][s][i];
                f = assoc.frame;
                Point2 point2D = data.points2D[c][f][assoc.point2D];
                p = assoc.point3D;
                graph.add(GenericProjectionFactor<Pose3, Point3, Cal3DS2>(
                        point2D, data.point2DNoise[c], Symbol(poseChar(c), f), Symbol('p', p), data.calibrations[c] ));
            }
        }
        
        // Add a odometry factors, if desired
        if (useOdometry) {
            for (i = 0; i < data.odometryAssocs[s].size(); ++i) {
                AssociationOdo assoc = data.odometryAssocs[s][i];
                Key poseSymbolFrom = Symbol(poseChar(assoc.from_cam), assoc.from_frame);
                Key poseSymbolTo = Symbol(poseChar(assoc.to_cam), assoc.to_frame);
                noiseModel::Base::shared_ptr odometryNoise = data.odometryNoise[assoc.from_cam][assoc.to_cam];
                graph.add(BetweenFactor<Pose3>(poseSymbolFrom, poseSymbolTo, data.odometry[s][i], odometryNoise));
            }
        }
        
        /* Update factor graph, perform incremental bundle adjustment */
        
#if DEBUG
        // Print the incremental graph
        graph.print("\n\n");
#endif
            
        // Do a full optimize for first two batches of 3D points (or at the last step if it was not yet done),
        // if desired and required
        if (needsFullOptimize && (
                ((s > 0) && !data.point3DAddedIdxs[s].empty() && iSAM_version) || (s == nrSteps - 1) )) {
            std::cout << "    performing full optimization" << std::endl;
            LevenbergMarquardtOptimizer batchOptimizer(graph, initialEstimate);
            initialEstimate = batchOptimizer.optimize();
            needsFullOptimize = false;
            
            if (s == nrSteps - 1)    // in this case, we don't need to calculate the best estimate anymore
                return initialEstimate;
        }
        
        // After the initial full optimization (if desired), iSAM is updated and "initialEstimate" is cleared
        if (!needsFullOptimize) {
            // Update iSAM with the new factors
            try {
                if (iSAM_version == 1)
                    isam1.update(graph, initialEstimate);
                else if (iSAM_version == 2) {
                    isam2.update(graph, initialEstimate);
                    // Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
                    // If accuracy is desired at the expense of time, update(*) can be called additional times
                    // to perform multiple optimizer iterations every step.
                    //isam2.update();
                }
            } catch (IndeterminantLinearSystemException e) {
                Key k = e.nearbyVariable();
                Symbol(k).print("IndeterminantLinearSystemException nearby Symbol ");
                throw;
            }
            
#if DEBUG
            // Print the current estimate of the whole graph
            if (iSAM_version == 1)
                currentEstimate = isam1.estimate();
            else if (iSAM_version == 2)
                currentEstimate = isam2.calculateEstimate();
            std::cout << "****************************************************" << endl;
            std::cout << "Step " << s << ": " << endl;
            currentEstimate.print("Current estimate: ");
            std::cout << "****************************************************" << endl;
#endif

            // Clear the factor graph and values for the next iteration
            graph.resize(0);
            initialEstimate.clear();
        }
    }
    
    if (iSAM_version == 1)
        return isam1.estimate();
    else if (iSAM_version == 2)
        return isam2.calculateBestEstimate();
}


/* ************************************************************************* */


void updateMapWithEstimate(BAdata& data, Values& estimate)
{
    //Values::ConstFiltered<Point3> points3D = estimate.filter<Point3>();
    for (size_t p = 0; p < data.points3D.size(); ++p)
        data.points3D[p].p = (Point3&)estimate.at(Symbol('p', p));
}


void updateTrajectoriesWithEstimate(BAdata& data, Values& estimate)
{
    //Values::ConstFiltered<Pose3> poses = estimate.filter<Pose3>();
    size_t nrCameras = data.poses.size();
    size_t nrFrames = data.poses[0].size();
    for (size_t c = 0; c < nrCameras; ++c)
        for (size_t f = 0; f < nrFrames; ++f)
            if (data.poses[c][f])    // pose should be valid
                data.poses[c][f]->p = (Pose3&)estimate.at(Symbol(poseChar(c), f));
}


/* ************************************************************************* */


void runFromGeneratedData(const std::string& baseDir, const std::string& baseName, const size_t nrCameras,
                          const bool useOdometry, const bool fullOptimizeAtSecondPoints3DBatch,
                          const ushort iSAM_version)
{
    Filenames filenames = createFilenames(baseDir, baseName, nrCameras);
    
    // Create the set of noise models, landmarks, poses and odometry
    BAdata data = createNoiseModelsAndPointsAndCamerasAndDataAndOdometry(nrCameras);
    validateDataIntegrity(data, nrCameras);
    validateDataSufficientlyConstrainted(data, useOdometry);
    
    // Perform BA, and print result
    Values finalEstimate = performBundleAdjustment(data, useOdometry, fullOptimizeAtSecondPoints3DBatch, iSAM_version);
    finalEstimate.print("Final estimate: ");
    
    // Update data with result
    updateMapWithEstimate(data, finalEstimate);
    updateTrajectoriesWithEstimate(data, finalEstimate);
    
    // Save result
    saveResult(filenames, data);
}


void runFromFiles(const std::string& baseDir, const std::string& baseName, const size_t nrCameras,
                  const int fps,
                  const bool useOdometry, const bool fullOptimizeAtSecondPoints3DBatch,
                  const double startTime, const bool firstFrameStartsAfterStartTime,
                  const ushort iSAM_version)
{
    Filenames filenames = createFilenames(baseDir, baseName, nrCameras);
    
    // Load the set of noise models, landmarks, poses and odometry
    BAdata data = loadData(filenames, fps, startTime, firstFrameStartsAfterStartTime);
    validateDataIntegrity(data, nrCameras);
    validateDataSufficientlyConstrainted(data, useOdometry);
    
    // Perform BA, and print result
    Values finalEstimate = performBundleAdjustment(data, useOdometry, fullOptimizeAtSecondPoints3DBatch, iSAM_version);
    finalEstimate.print("Final estimate: ");
    
    // Update data with result
    updateMapWithEstimate(data, finalEstimate);
    updateTrajectoriesWithEstimate(data, finalEstimate);
    
    // Save result
    saveResult(filenames, data);
}


/* ************************************************************************* */


int main(int argc, char* argv[])
{
    // Defaults
    bool useOdometry = true;
    bool fullOptimizeAtSecondPoints3DBatch = true;
    double startTime = 0.;
    bool firstFrameStartsAfterStartTime = true;
    ushort iSAM_version = 2;
    bool runFromGenerated = false;
    
    // Print help message, if needed
    if ((argc-1 < 4) || (argc-1 > 10)) {
        std::cout << "Usage: " << argv[0] << " "
                  << " <baseDir> <baseName (e.g. algorithm)> <nrCameras> "
                  << " <fps> "
                  << " [<useOdometry (default: " << useOdometry << ")> [<fullOptimizeAtSecondPoints3DBatch (default: " <<  fullOptimizeAtSecondPoints3DBatch << ")> "
                  << " [<startTime (default: " << startTime << ")> [<firstFrameStartsAfterStartTime (default: " << firstFrameStartsAfterStartTime << ")> "
                  << " [<iSAM_version (default: " << iSAM_version << ")> "
                  << " [<runFromGenerated (default: " << runFromGenerated << ")>]]]]]]" << std::endl
                  << std::endl
                  << "Bundle-Adjustment (using full optimization (L-M), iSAM, or iSAM2)." << std::endl
                  << std::endl
                  << "The input files are given by the path '<baseDir>/<title>[-<baseName>].<ext>' " << std::endl
                  << "where <title> and <ext> are determined by the pupose of that file; " << std::endl
                  << "the output files are '<baseDir>/traj_out-<baseName>-BA.txt' "
                  << "and '<baseDir>/map_out-<baseName>-BA.txt'. " << std::endl
                  << "See the files in the 'example' directory for all required files." << std::endl
                  << std::endl
                  << "If <fps> is set to '0', each separate pose in the trajectory files " << std::endl
                  << "will correspond directly with frameIdx in the 'BA_info.*' files, " << std::endl
                  << "thus this requires all cams to be synced: " << std::endl
                  << "each frame should have the same timestamp across cams." << std::endl
                  << std::endl
                  << "Setting <fullOptimizeAtSecondPoints3DBatch> to '1' "
                  << "will perform a full optimization " << std::endl
                  << "after the second (or later) step, when a non-empty batch of 3D points is added. " << std::endl
                  << "If <iSAM_version> is to '0', no incremental BA will be performed, " << std::endl
                  << "but the full optimization will be performed (using Levenberg-Marquardt)." << std::endl
                  << std::endl
                  << "If <runFromGenerated> is set to '1', no input files will be used, " << std::endl
                  << "instead, the input will be generated: 8 points on a 10 meter cube, " << std::endl
                  << "captured by a robot (with a positive height) "
                  << "rotating around and facing towards the cube. " << std::endl
                  << "The <fps>, <startTime> and <firstFrameStartsAfterStartTime> parameters "
                  << "will have no effect. " << std::endl
                  << "In case <nrCameras> is set to '2', another robot, " << std::endl
                  << "positioned symmetric to the first robot (negative height) "
                  << "and with an offset angle of 45 degree around the Z axis, " << std::endl
                  << "adds additional measurements. " << std::endl;
        return -1;
    }
    
    // Read required command-line arguments
    argv++;
    std::string baseDir = *(argv++);
    std::string baseName = *(argv++);
    size_t nrCameras = atol(*(argv++));
    int fps = atoi(*(argv++));
    
    // Read optional command-line arguments
    if (argc >= 6)  { useOdometry = atoi(*(argv++));
    if (argc >= 7)  { fullOptimizeAtSecondPoints3DBatch = atoi(*(argv++));
    if (argc >= 8)  { startTime = atof(*(argv++));
    if (argc >= 9)  { firstFrameStartsAfterStartTime = atoi(*(argv++));
    if (argc >= 10) { iSAM_version = atoi(*(argv++));
    if (argc >= 11) { runFromGenerated = atoi(*(argv++)); }}}}}}
    
    if (runFromGenerated)
        runFromGeneratedData(baseDir, baseName, nrCameras,
                             useOdometry, fullOptimizeAtSecondPoints3DBatch,
                             iSAM_version);
    else
        runFromFiles(baseDir, baseName, nrCameras,
                     fps,
                     useOdometry, fullOptimizeAtSecondPoints3DBatch,
                     startTime, firstFrameStartsAfterStartTime,
                     iSAM_version);
    
    return 0;
}
