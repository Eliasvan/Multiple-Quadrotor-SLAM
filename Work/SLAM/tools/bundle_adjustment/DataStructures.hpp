#pragma once

#include <gtsam/linear/NoiseModel.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3DS2.h>

using namespace gtsam;


/* ************************************************************************* */


typedef struct TrajectoryNode {
    Pose3 p;     // pose
    double t;    // timestamp
    TrajectoryNode(Pose3 p_, double t_) : p(p_), t(t_) {}
} TrajectoryNode;


// "white" color (PCD encoding)
static float MapPoint_colorWhite = *reinterpret_cast<float *>(new uint32_t(
        (0xFF << 0) + (0xFF << 8) + (0xFF << 16) + (0xFD << 24) ));
          /* B */       /* G */       /* R */         /* A */

typedef struct MapPoint {
    Point3 p;    // 3D point
    float c;    // color (PCD format)
    MapPoint() : c(MapPoint_colorWhite) {}
    MapPoint(double x, double y, double z) : p(x, y, z), c(MapPoint_colorWhite) {}
} MapPoint;


typedef struct AssociationOdo {
    /* All variables denote indices in arrays */
    size_t from_cam;
    size_t from_frame;
    size_t to_cam;
    size_t to_frame;
} AssociationOdo;


typedef struct Association2D3D {
    /* All variables denote indices in arrays */
    size_t frame;
    size_t point2D;
    size_t point3D;
} Association2D3D;


/* ************************************************************************* */


typedef struct BAdata {
    std::vector<noiseModel::Base::shared_ptr> poseNoise;                       // for each cam
    std::vector< std::vector<noiseModel::Base::shared_ptr> > odometryNoise;    // matrix: rows represent {from cam}, columns represent {to cam}
    noiseModel::Base::shared_ptr point3DNoise;
    std::vector<noiseModel::Base::shared_ptr> point2DNoise;                    // for each cam
    
    /* Note 1: Hierarchy in most following arrays is as follows:
     *          For each Camera
     *              For each Step / Frame
     *                  Some properties
     * where "steps" and "frames" (or frame-numbers) don't always correspond with eachother,
     * e.g. 2D -> 3D associations can include older frames than the current one,
     * but for the "poses" array, "steps" and "frames" *do* correspond with eachother.
     * 
     * Note 2: Some poses at certain steps might be invalid,
     * this is represented by an empty (shared) pointer.
     * 
     * Note 3: All 3D point indices should be defined exactly once in the "point3DAddedIdxs" array,
     * and no "point2D3DAssocs" can be made containing the 3D point index
     * if it's not yet included in the "point3DAddedIdxs" array at that step.
     */
    
    std::vector< boost::shared_ptr<Cal3DS2> > calibrations;                        // calibration for each cam
    
    std::vector< std::vector< boost::shared_ptr<TrajectoryNode> > > poses;         // pose for each step, for each cam
    
    std::vector< std::vector<Pose3> > odometry;                                    // array of delta-poses, for each step
    std::vector< std::vector<AssociationOdo> > odometryAssocs;                     // associations from {cam and frame idx} to {cam and frame idx}, for each delta-pose, for each step
    
    std::vector<MapPoint> points3D;                                                // 3D point array
    std::vector< std::vector<size_t> > point3DAddedIdxs;                           // array of 3D point idxs that are newly added, for each step
    std::vector< std::vector< std::vector<Point2> > > points2D;                    // 2D coordinates of each feature point, for each frame, for each cam
    std::vector< std::vector< std::vector<Association2D3D> > > point2D3DAssocs;    // associations between {frame, 2D point and 3D point idx}, for each step, for each cam
} BAdata;


/* ************************************************************************* */


void validateDataIntegrity(BAdata& data, size_t nrCameras)
{
    size_t c, s, i;
    
    // Check list length constraints : nrCameras
    assert (data.poses.size() == nrCameras);
    assert (data.calibrations.size() == nrCameras);
    assert (data.points2D.size() == nrCameras);
    assert (data.point2D3DAssocs.size() == nrCameras);
    assert (data.poseNoise.size() == nrCameras);
    assert (data.point2DNoise.size() == nrCameras);
    assert (data.odometryNoise.size() == nrCameras);
    for (c = 0; c < nrCameras; ++c)
        assert (data.odometryNoise[c].size() == nrCameras);
    
    // Check list length constraints : nrFrames
    assert (nrCameras > 0);
    size_t nrFrames = data.poses[0].size();
    for (c = 0; c < nrCameras; ++c)
        assert (data.poses[c].size() == nrFrames);
    
    // Check list length constraints : nrSteps
    size_t nrSteps = data.point3DAddedIdxs.size();
    assert (nrSteps == nrFrames);
    assert (data.odometry.size() == nrSteps);
    assert (data.odometryAssocs.size() == nrSteps);
    
    // Check list length constraints : internal associations in each step
    for (s = 0; s < nrSteps; ++s)
        assert (data.odometry[s].size() == data.odometryAssocs[s].size());
    
    // Check 3D point indices
    size_t nrPoints = data.points3D.size();
    for (s = 0; s < nrSteps; ++s)
        for (i = 0; i < data.point3DAddedIdxs[s].size(); ++i) {
            assert (data.point3DAddedIdxs[s][i] >= 0);
            assert (data.point3DAddedIdxs[s][i] < nrPoints);
        }
    
    // Check association indices : 2D -> 3D
    for (c = 0; c < nrCameras; ++c)
        for (s = 0; s < nrSteps; ++s)
            for (i = 0; i < data.point2D3DAssocs[c][s].size(); ++i) {
                Association2D3D assoc = data.point2D3DAssocs[c][s][i];
                assert (assoc.frame >= 0);
                assert (assoc.frame <= s);    // looking into the future is not possible
                assert (assoc.point2D >= 0);
                assert (assoc.point2D < data.points2D[c][assoc.frame].size());
                assert (assoc.point3D >= 0);
                assert (assoc.point3D < nrPoints);
                assert (data.poses[c][assoc.frame]);    // check whether an estimate of this pose exists
            }
    
    // Check association indices : odometry
    for (s = 0; s < nrSteps; ++s)
        for (i = 0; i < data.odometryAssocs[s].size(); ++i) {
            AssociationOdo assoc = data.odometryAssocs[s][i];
            assert (assoc.from_cam >= 0);
            assert (assoc.from_cam < nrCameras);
            assert (assoc.to_cam >= 0);
            assert (assoc.to_cam < nrCameras);
            assert (assoc.from_frame >= 0);
            assert (assoc.from_frame <= s);    // looking into the future is not possible
            assert (assoc.to_frame >= 0);
            assert (assoc.to_frame <= s);    // looking into the future is not possible
            assert (!( (assoc.from_cam == assoc.to_cam) &&         // odometry between the exact same image ...
                       (assoc.from_frame == assoc.to_frame) ));    // ... makes little sense
            assert (data.poses[assoc.from_cam][assoc.from_frame]);    // check whether an estimate ...
            assert (data.poses[assoc.to_cam][assoc.to_frame]);        // ... of these poses exist
        }
}
