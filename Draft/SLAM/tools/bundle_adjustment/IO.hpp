#pragma once

#include <gtsam/linear/NoiseModel.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3DS2.h>

// We need to create filepaths in an OS independent way
#include <boost/filesystem.hpp>

#include "DataStructures.hpp"

using namespace gtsam;


/* ************************************************************************* */


typedef struct Filenames {
    /* Input files */
   
    std::string map_in;
    std::vector<std::string> trajectories_in;     // for each cam
    
    std::vector<std::string> poseNoise;           // for each cam
    std::string odometryNoise;
    std::string point3DNoise;
    std::vector<std::string> point2DNoise;        // for each cam
    
    std::vector<std::string> calibrations;        // for each cam
    
    std::string odometry;
    std::string odometryAssocs;
    std::string point3DAddedIdxs;
    std::vector<std::string> points2D;            // for each cam
    std::vector<std::string> point2D3DAssocs;     // for each cam
    
    /* Output files */
    
    std::string map_out;
    std::vector<std::string> trajectories_out;    // for each cam
} Filenames;


Filenames createFilenames(const std::string& baseDir, const std::string& baseName, const size_t nrCameras)
{
    Filenames filenames;
    
    boost::filesystem::path basePath = boost::filesystem::path(baseDir.c_str());
    std::stringstream name;
    size_t c;
    
    /* Input files : map and trajectories */
    
    // "map_out-algorithm.pcd"
    name.str(""); name << "map_out-" << baseName << ".pcd";
    filenames.map_in = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "traj_out.camC-algorithm.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "traj_out.cam" << c << "-" << baseName << ".txt";
        filenames.trajectories_in.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    /* Input files : noise models */
    
    // "BA_info.noise.pose.camC-algorithm.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "BA_info.noise.pose.cam" << c << "-" << baseName << ".txt";
        filenames.poseNoise.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    // "BA_info.noise.odometry-algorithm.txt"
    name.str(""); name << "BA_info.noise.odometry-" << baseName << ".txt";
    filenames.odometryNoise = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "BA_info.noise.point3D-algorithm.txt"
    name.str(""); name << "BA_info.noise.point3D-" << baseName << ".txt";
    filenames.point3DNoise = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "BA_info.noise.point2D.camC-algorithm.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "BA_info.noise.point2D.cam" << c << "-" << baseName << ".txt";
        filenames.point2DNoise.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    /* Input files : camera calibrations */
    
    // "BA_info.calibrations.camC.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "BA_info.calibrations.cam" << c << ".txt";
        filenames.calibrations.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    /* Input files : additional state info required for bundle adjustment */
    
    // "BA_info.measurements.odometry-algorithm.txt"
    name.str(""); name << "BA_info.measurements.odometry-" << baseName << ".txt";
    filenames.odometry = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "BA_info.measurements.odometryAssocs-algorithm.txt"
    name.str(""); name << "BA_info.measurements.odometryAssocs-" << baseName << ".txt";
    filenames.odometryAssocs = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "BA_info.measurements.point3DAddedIdxs-algorithm.txt"
    name.str(""); name << "BA_info.measurements.point3DAddedIdxs-" << baseName << ".txt";
    filenames.point3DAddedIdxs = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "BA_info.measurements.points2D.camC-algorithm.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "BA_info.measurements.points2D.cam" << c << "-" << baseName << ".txt";
        filenames.points2D.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    // "BA_info.measurements.point2D3DAssocs.camC-algorithm.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "BA_info.measurements.point2D3DAssocs.cam" << c << "-" << baseName << ".txt";
        filenames.point2D3DAssocs.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    /* Output files : map and trajectories */
    
    // "map_out-algorithm-BA.pcd"
    name.str(""); name << "map_out-" << baseName << "-BA.pcd";
    filenames.map_out = (basePath / boost::filesystem::path(name.str().c_str())).string();
    
    // "traj_out.camC-algorithm-BA.txt"
    for (c = 0; c < nrCameras; ++c) {
        name.str(""); name << "traj_out.cam" << c << "-" << baseName << "-BA.txt";
        filenames.trajectories_out.push_back((basePath / boost::filesystem::path(name.str().c_str())).string());
    }
    
    return filenames;
}


/* ************************************************************************* */


template <typename T>
std::vector< std::vector<T> > loadAscii(const std::string& filename,
                                        const boost::function<T(std::vector<std::string>&, size_t&)> decoder,
                                        size_t decoder_arg = 0,
                                        bool emptyLinesTriggerNewList = true) 
{
    std::vector< std::vector<T> > output;
    std::vector<T> firstListEntry;
    output.push_back(firstListEntry);
    
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        throw std::runtime_error("Can't open file '" + filename + "'.");
    }
    
    size_t decoder_arg_prev = decoder_arg;    // this variable can be used as memory between calls to the decoder
    std::string line;
    while (std::getline(file, line)) {
        // A comment in the file has been detected
        if (!line.empty() && (line[0] == '#'))
            continue;
        
        // An empty line denotes a new list entry in the output, if desired
        if (emptyLinesTriggerNewList && line.empty()) {
            std::vector<T> newListEntry;
            output.push_back(newListEntry);
            continue;
        }
        
        // Split the line by spaces, decode it, and at the result to the current list entry
        std::vector<std::string> splitted;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ' '))
            splitted.push_back(item);
        T t = decoder(splitted, decoder_arg);
        if (decoder_arg == decoder_arg_prev)
            output.back().push_back(t);    // if the passed argument remains unchanged, save the result
        else
            decoder_arg_prev = decoder_arg;
    }
    file.close();
    
    return output;
}


size_t decode_size_t(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 1);
    return (size_t)atol(splitted[0].c_str());
}


AssociationOdo decode_AssociationOdo(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 4);
    AssociationOdo assoc;
    assoc.from_cam   = (size_t)atol(splitted[0].c_str());
    assoc.from_frame = (size_t)atol(splitted[1].c_str());
    assoc.to_cam     = (size_t)atol(splitted[2].c_str());
    assoc.to_frame   = (size_t)atol(splitted[3].c_str());
    return assoc;
}


Association2D3D decode_Association2D3D(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 3);
    Association2D3D assoc;
    assoc.frame   = (size_t)atol(splitted[0].c_str());
    assoc.point2D = (size_t)atol(splitted[1].c_str());
    assoc.point3D = (size_t)atol(splitted[2].c_str());
    return assoc;
}


Point2 decode_Point2(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 2);
    return Point2(atof(splitted[0].c_str()), atof(splitted[1].c_str()));
}


Pose3 decode_Pose3(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 7);
    Point3 t(atof(splitted[0].c_str()), atof(splitted[1].c_str()), atof(splitted[2].c_str()));
    Rot3 R = Rot3::quaternion(atof(splitted[6].c_str()),
                              atof(splitted[3].c_str()), atof(splitted[4].c_str()), atof(splitted[5].c_str()));
    return Pose3(R, t);
}


boost::shared_ptr<Cal3DS2> decode_Cal3DS2(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 9);
    return boost::make_shared<Cal3DS2>(
            atof(splitted[0].c_str()), atof(splitted[1].c_str()), atof(splitted[2].c_str()), atof(splitted[3].c_str()),
            atof(splitted[4].c_str()), atof(splitted[5].c_str()), atof(splitted[6].c_str()), atof(splitted[7].c_str()),
            atof(splitted[8].c_str()) );
}


boost::shared_ptr<TrajectoryNode> decode_TrajectoryNode(std::vector<std::string>& splitted, size_t arg) {
    assert (splitted.size() == 8);
    double t = atof(splitted[0].c_str());
    splitted = std::vector<std::string>(splitted.begin() + 1, splitted.end());
    Pose3 p = decode_Pose3(splitted, arg);
    return boost::make_shared<TrajectoryNode>(p, t);
}


MapPoint decode_MapPoint(std::vector<std::string>& splitted, size_t& status) {
    assert (splitted.size() >= 1);
    MapPoint point3D;
    if ( (splitted[0] == "VERSION") || 
            (splitted[0] == "FIELDS") || (splitted[0] == "SIZE") || (splitted[0] == "TYPE") ||
            (splitted[0] == "COUNT") || (splitted[0] == "WIDTH") || (splitted[0] == "HEIGHT") ||
            (splitted[0] == "VIEWPOINT") || (splitted[0] == "POINTS") || (splitted[0] == "DATA") ) {
        if (splitted[0] == "FIELDS") {    // check whether the fields are correct
            assert (splitted.size() >= 1 + 3);
            assert ((splitted[1] == "x") && (splitted[2] == "y") && (splitted[3] == "z"));
            if (splitted.size() > 1 + 3)
                assert (splitted[4] == "rgb");
        }
        ++status;    // signal that the current line in the PCD file is not a point, but part of the header
    }
    else {    // assume here we're decoding a point
        assert (splitted.size() >= 3);
        point3D = MapPoint(atof(splitted[0].c_str()), atof(splitted[1].c_str()), atof(splitted[2].c_str()));
        if (splitted.size() > 3)
            point3D.c = atof(splitted[3].c_str());    // color
    }
    return point3D;
}


noiseModel::Base::shared_ptr decode_NoiseModel(std::vector<std::string>& splitted, size_t noiseDim) {
    assert (splitted.size() >= 1);
    std::string noiseType(splitted[0]);
    assert (!noiseType.empty());
    splitted = std::vector<std::string>(splitted.begin() + 1, splitted.end());
    
    if (noiseType == "Unit") {
        assert (splitted.size() == 0);
        return noiseModel::Unit::Create(noiseDim);
    } else if (noiseType == "Isotropic") {
        assert (splitted.size() == 1);
        return noiseModel::Isotropic::Sigma(noiseDim, atof(splitted[0].c_str()));
    } else if ((noiseType == "Diagonal") || (noiseType == "Constrained")) {
        assert (splitted.size() == noiseDim);
        Vector v = Vector(noiseDim);
        for (size_t i = 0; i < noiseDim; ++i)
            v[i] = atof(splitted[i].c_str());
        if (noiseType == "Diagonal")
            return noiseModel::Diagonal::Sigmas(v);
        else
            return noiseModel::Constrained::MixedSigmas(v);
    } else
        throw std::runtime_error("Noise-type '" + noiseType + "' unknown.");
}


/* ************************************************************************* */


void fillHolesInTrajectories(BAdata& data, const int fps, const double startTime, const bool firstFrameStartsAfterStartTime)
{
    size_t nrCameras = data.poses.size();
    size_t nrSteps = data.point3DAddedIdxs.size();
    size_t nrFrames;
    size_t c, f;
    
    // Determine end-time, and total number of frames
    assert (nrCameras > 0);
    double endTime = startTime;
    for (c = 0; c < nrCameras; ++c)
        if (!data.poses[c].empty() && (data.poses[c].back()->t > endTime))
            endTime = data.poses[c].back()->t;
    if (fps > 0)
        nrFrames = round((endTime - startTime) * fps);
    else
        nrFrames = data.poses[0].size();    // special case: assume all trajectories have same length
    
    if (fps > 0) {
        // Create iterators for the original poses, to be used below
        std::vector< std::vector< boost::shared_ptr<TrajectoryNode> >::iterator > pose_its;
        for (c = 0; c < nrCameras; ++c)
            pose_its.push_back(data.poses[c].begin());
        
        // Detect and fill holes in the trajectory, save them in "poses_new", then replace "data.poses"
        std::vector< std::vector< boost::shared_ptr<TrajectoryNode> > > poses_new;
        if (!firstFrameStartsAfterStartTime)
            ++nrFrames;
        for (c = 0; c < nrCameras; ++c) {
            poses_new.push_back(std::vector< boost::shared_ptr<TrajectoryNode> >());
            for (f = 0; f < nrFrames; ++f) {
                double t = startTime + (f + firstFrameStartsAfterStartTime) / (double)fps;
                
                // Go to next pose in proximity of "t"
                while ((pose_its[c] != data.poses[c].end()) && ((**pose_its[c]).t < t - 0.5 / fps))
                    ++pose_its[c];
                
                // There exists a pose at (approx) timestamp "t"
                if ( (pose_its[c] != data.poses[c].end()) && 
                        ((t - 0.5 / fps <= (**pose_its[c]).t) && 
                        ((**pose_its[c]).t < t + 0.5 / fps)) ) {
                    poses_new[c].push_back(*pose_its[c]);
                }
                
                // No such pose exists => fill hole with empty pose
                else {
                    boost::shared_ptr<TrajectoryNode> node;
                    poses_new[c].push_back(node);
                }
            }
        }
        data.poses = poses_new;
    }
    
    // Fill holes at end, to match nrSteps
    assert (nrSteps >= nrFrames);
    for (c = 0; c < nrCameras; ++c)
        for (f = nrFrames; f < nrSteps; ++f) {
            boost::shared_ptr<TrajectoryNode> node;
            data.poses[c].push_back(node);
        }
}


BAdata loadData(Filenames& filenames, const int fps = 1, const double startTime = 0., const bool firstFrameStartsAfterStartTime = true)
{
    BAdata data;
    size_t c;
    
    // Load noise models
    for (c = 0; c < filenames.poseNoise.size(); ++c)
        data.poseNoise.push_back(
                loadAscii<noiseModel::Base::shared_ptr>(filenames.poseNoise[c], decode_NoiseModel, 6)[0][0] );
    data.odometryNoise = loadAscii<noiseModel::Base::shared_ptr>(filenames.odometryNoise, decode_NoiseModel, 6);
    data.point3DNoise = loadAscii<noiseModel::Base::shared_ptr>(filenames.point3DNoise, decode_NoiseModel, 3)[0][0];
    for (c = 0; c < filenames.point2DNoise.size(); ++c)
        data.point2DNoise.push_back(
                loadAscii<noiseModel::Base::shared_ptr>(filenames.point2DNoise[c], decode_NoiseModel, 2)[0][0] );
    
    // Load cam calibrations
    for (c = 0; c < filenames.calibrations.size(); ++c)
        data.calibrations.push_back(
                loadAscii< boost::shared_ptr<Cal3DS2> >(filenames.calibrations[c], decode_Cal3DS2)[0][0] );
    
    // Load odometry + associations
    data.odometry = loadAscii<Pose3>(filenames.odometry, decode_Pose3);
    data.odometryAssocs = loadAscii<AssociationOdo>(filenames.odometryAssocs, decode_AssociationOdo);
    
    // Load 3D and 2D points + associations
    data.points3D = loadAscii<MapPoint>(filenames.map_in, decode_MapPoint)[0];
    data.point3DAddedIdxs = loadAscii<size_t>(filenames.point3DAddedIdxs, decode_size_t);
    for (c = 0; c < filenames.points2D.size(); ++c)
        data.points2D.push_back(loadAscii<Point2>(filenames.points2D[c], decode_Point2));
    for (c = 0; c < filenames.point2D3DAssocs.size(); ++c)
        data.point2D3DAssocs.push_back(
                loadAscii<Association2D3D>(filenames.point2D3DAssocs[c], decode_Association2D3D) );
    
    // Load cam trajectories (poses)
    for (c = 0; c < filenames.trajectories_in.size(); ++c)
        data.poses.push_back(
                loadAscii< boost::shared_ptr<TrajectoryNode> >(filenames.trajectories_in[c], decode_TrajectoryNode)[0] );
    fillHolesInTrajectories(data, fps, startTime, firstFrameStartsAfterStartTime);
    
    return data;
}


/* ************************************************************************* */


void saveTrajectory(const std::string& traj_out_file, std::vector< boost::shared_ptr<TrajectoryNode> >& poses)
{
    // Write trajectory to file
    std::ofstream ofs_traj(traj_out_file.c_str());
    
    ofs_traj << 
    "# Format: timestamp tx ty tz qx qy qz qw" << std::endl <<
    "# Where translations and quaternions are defined in world coordinates (=> inverse of pose)" << std::endl;

    size_t nrFrames = poses.size();
    for (size_t f = 0; f < nrFrames; ++f)
    {
        // Skip invalid poses
        boost::shared_ptr<TrajectoryNode> p = poses[f];
        if (!p)
            continue;
        
        // Access the pose of the camera
        Pose3 world_transf = p->p;//p->p.inverse();
        Quaternion quat = world_transf.rotation().toQuaternion();
        Point3 transl = world_transf.translation();
        ofs_traj  << std::setprecision(16)
                  << p->t << " "
                  << transl.x() << " " << transl.y() << " " << transl.z() << " "
                  << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
                  << std::endl;
    }
}


void saveMap(const std::string& map_out_file, std::vector<MapPoint>& points3D)
{
    // Write pcd file for pointcloud visualization (e.g. in Blender)
    std::ofstream ofs_map(map_out_file.c_str());
    
    size_t nrPoints = points3D.size();
    ofs_map << "# .PCD v.7 - Point Cloud Data file format" << std::endl
            << "VERSION .7" << std::endl
            << "FIELDS x y z rgb" << std::endl
            << "SIZE 4 4 4 4" << std::endl
            << "TYPE F F F F" << std::endl
            << "COUNT 1 1 1 1" << std::endl
            << "WIDTH " << nrPoints << std::endl
            << "HEIGHT 1" << std::endl
            << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
            << "POINTS " << nrPoints << std::endl
            << "DATA ascii" << std::endl;
    
    for (size_t p = 0; p < nrPoints; ++p) {
        Point3 point3D = points3D[p].p;
        float c = points3D[p].c;
        
        ofs_map << std::setprecision(16) << point3D.x() << " " << point3D.y() << " " << point3D.z() << " "
                << std::setprecision(9)  << c << std::endl;
    }
}


void saveResult(Filenames& filenames, BAdata& data)
{
    saveMap(filenames.map_out, data.points3D);
    for (size_t c = 0; c < data.poses.size(); ++c)
        saveTrajectory(filenames.trajectories_out[c], data.poses[c]);
}
