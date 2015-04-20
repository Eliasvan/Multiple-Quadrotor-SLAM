// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <boost/filesystem.hpp>
#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/feature_detection.h>
#include <svo/depth_filter.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>
#include "test_utils.h"

namespace svo {

struct ConvergedSeed {
  int x_, y_;
  Vector3d pos_;
  cv::Vec3b col_;
  ConvergedSeed(int x, int y, Vector3d pos, cv::Vec3b col) :
    x_(x), y_(y), pos_(pos), col_(col)
  {}
};

class BenchmarkNode
{
  std::string img_dir_;
  std::string traj_out_file_;
  std::string map_out_file_;
  int fps_;
  vk::AbstractCamera* cam_;
  FrameHandlerMono* vo_;
  DepthFilter* depth_filter_;
  std::list<ConvergedSeed> results_;

public:
  BenchmarkNode(std::string img_dir, std::string traj_out_file, std::string map_out_file,
                int fps,
                double width, double height,
                double fx, double fy, double cx, double cy,
                double d0, double d1, double d2, double d3, double d4);
  ~BenchmarkNode();
  void depthFilterCb(svo::Point* point, double depth_sigma2);
  int runFromFolder();
};

BenchmarkNode::BenchmarkNode(std::string img_dir, std::string traj_out_file, std::string map_out_file,
                             int fps,
                             double width, double height,
                             double fx, double fy,
                             double cx, double cy,
                             double d0, double d1, double d2, double d3, double d4) :
                             img_dir_(img_dir), traj_out_file_(traj_out_file), map_out_file_(map_out_file), fps_(fps)
{
  //cam_ = new vk::PinholeCamera(752, 480, 315.5, 315.5, 376.0, 240.0);
  cam_ = new vk::PinholeCamera(width, height, fx, fy, cx, cy, d0, d1, d2, d3, d4);
  vo_ = new FrameHandlerMono(cam_);
  vo_->start();
}

BenchmarkNode::~BenchmarkNode()
{
  delete vo_;
  delete cam_;
}

void BenchmarkNode::depthFilterCb(svo::Point* point, double depth_sigma2)
{
  cv::Vec3b color = point->obs_.front()->frame->img_pyr_[0].at<cv::Vec3b>(point->obs_.front()->px[0], point->obs_.front()->px[1]);
  results_.push_back(ConvergedSeed(
      point->obs_.front()->px[0], point->obs_.front()->px[1], point->pos_, color));
  delete point->obs_.front();
}

int BenchmarkNode::runFromFolder()
{
    // get a sorted list of files in the img directory
    boost::filesystem::path img_dir_path(img_dir_.c_str());
    if (!boost::filesystem::exists(img_dir_path))
        return -1;
    
    // get all files in the img directory
    size_t max_len = 0;
    std::list<std::string> imgs;
    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    for (boost::filesystem::directory_iterator file(img_dir_path); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  || 
                 filename_path.extension() == ".jpg"  || 
                 filename_path.extension() == ".jpeg" || 
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs.push_back(filename);
            max_len = max(max_len, filename.length());
        }
    }
    
    // sort them by filename; add leading zeros to make filename-lengths equal if needed
    std::map<std::string, std::string> sorted_imgs;
    for (std::list<std::string>::iterator img = imgs.begin(); img != imgs.end(); ++img)
        sorted_imgs[std::string(max_len - img->length(), '0') + (*img)] = *img;
    
    // run SVO (pose estimation)
    std::list<FramePtr> frames;
    int frame_counter = 1;
    for (std::map<std::string, std::string>::iterator it = sorted_imgs.begin(); it != sorted_imgs.end(); ++it)
    {
        // load image
        boost::filesystem::path img_path = img_dir_path / boost::filesystem::path(it->second.c_str());
        if (frame_counter == 1)
            std::cout << "reading image " << img_path.string() << std::endl;
        cv::Mat img(cv::imread(img_path.string(), 0));
        assert(!img.empty());

        // process frame
        vo_->addImage(img, frame_counter / (double)fps_);

        // display tracking quality
        if (vo_->lastFrame() != NULL) {
            std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                      << "#Features: " << vo_->lastNumObservations() << " \t"
                      << "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms" << std::endl;
            frames.push_back(vo_->lastFrame());
        }
        
        frame_counter++;
    }
    
    // create new depth-filter, to construct the pointcloud based on the final poses
    svo::feature_detection::DetectorPtr feature_detector(
        new svo::feature_detection::FastDetector(
            cam_->width(), cam_->height(), svo::Config::gridSize(), svo::Config::nPyrLevels()));
    DepthFilter::callback_t depth_filter_cb = boost::bind(&BenchmarkNode::depthFilterCb, this, _1, _2);
    depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
    depth_filter_->options_.verbose = true;
    
    // write trajectories to file, and run depth-filter
    std::ofstream ofs_traj(traj_out_file_.c_str());
    frame_counter = 1 + 10;
    for (std::list<FramePtr>::iterator frame=frames.begin(); frame!=frames.end(); ++frame)
    {
        Eigen::Matrix<double, 6, 6> cov = (*frame)->Cov_;
        bool skip_frame = false;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                if (! ((1.e-16 < fabs(cov(i,j))) && (fabs(cov(i,j)) < 1.e+16)) )    // likely an invalid pose
                    skip_frame = true;
        if (skip_frame) {
            frame_counter++;
            continue;
        }
        
        // access the pose of the camera via vo_->lastFrame()->T_f_w_.
        Sophus::SE3 world_transf = (*frame)->T_f_w_.inverse();
        Eigen::Quaterniond quat = world_transf.unit_quaternion();
        Eigen::Vector3d transl = world_transf.translation();
        //std::cout << "Quaternion: " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << " \n"
        //          << "Translation: " << transl(0) << " " << transl(1) << " " << transl(2) << " \n";
        ofs_traj  << (*frame)->timestamp_ << " "
                  << transl(0) << " " << transl(1) << " " << transl(2) << " "
                  << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
                  << std::endl;
        
        // once in 10 frames, add a keyframe to the depth-filter
        if (frame_counter > 10) {
            frame_counter = 1;
            depth_filter_->addKeyframe(*frame, 2, 0.5);
        } else
            depth_filter_->addFrame(*frame);
        
        frame_counter++;
    }

    /*
    // write ply file for pointcloud visualization in Meshlab
    std::string trace_map_name("results/depth_filter.ply");
    std::ofstream ofs_map(trace_map_name.c_str());
    ofs_map << "ply" << std::endl
        << "format ascii 1.0" << std::endl
        << "element vertex " << results_.size() << std::endl
        << "property float x" << std::endl
        << "property float y" << std::endl
        << "property float z" << std::endl
        << "property uchar blue" << std::endl
        << "property uchar green" << std::endl
        << "property uchar red" << std::endl
        << "end_header" << std::endl;
    
    for (std::list<ConvergedSeed>::iterator seed = results_.begin(); seed != results_.end(); ++seed)
    {
        Eigen::Vector3d p = seed->pos_;
        cv::Vec3b c = seed->col_;
        ofs_map << p[0] << " " << p[1] << " " << p[2] << " "
                << (int) c[0] << " " << (int) c[1] << " " << (int) c[2] << std::endl;
    }
    */
    
    // write pcd file for pointcloud visualization (e.g. in Blender)
    std::ofstream ofs_map(map_out_file_.c_str());
    ofs_map << "# .PCD v.7 - Point Cloud Data file format" << std::endl
            << "VERSION .7" << std::endl
            << "FIELDS x y z rgb" << std::endl
            << "SIZE 4 4 4 4" << std::endl
            << "TYPE F F F F" << std::endl
            << "COUNT 1 1 1 1" << std::endl
            << "WIDTH " << results_.size() << std::endl
            << "HEIGHT 1" << std::endl
            << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl
            << "POINTS " << results_.size() << std::endl
            << "DATA ascii" << std::endl;
    
    for (std::list<ConvergedSeed>::iterator seed = results_.begin(); seed != results_.end(); ++seed)
    {
        Eigen::Vector3d p = seed->pos_;
        cv::Vec3b c = seed->col_;
        
        // pcd encoding of color
        #define clamp(v, lower, upper)    min(max((v), (lower)), (upper))
        uint32_t *f = new uint32_t((clamp((int)round(c[0]), 0, 0xFF) <<  0) + 
                                   (clamp((int)round(c[1]), 0, 0xFF) <<  8) + 
                                   (clamp((int)round(c[2]), 0, 0xFF) << 16) + 
                                                               (0xFD << 24));
        ofs_map << p[0] << " " << p[1] << " " << p[2] << " "
                << std::setprecision(8) << *reinterpret_cast<float *>(f) << std::endl;
    }
    
    std::cout << "Done." << std::endl;
    return 0;
}

} // namespace svo

int main(int argc, char** argv)
{
    // print help message, if needed
    if ((argc-1 < 10) || (argc-1 > 15)) {
        std::cout << "Usage: " << argv[0] << " "
                  << " <img_dir> <traj_out_file (TUM format)> <map_out_file (PCD format)> "
                  << " <fps> "
                  << " <width> <height> "
                  << " <fx> <fy> <cx> <cy> "
                  << " [<d0> [<d1> [<d2> [<d3> [<d4>]]]]]" << std::endl;
        return -1;
    }
    
    // read cmdl arguments
    argv++;
    std::string traj_out_file, map_out_file, img_dir;
    img_dir = *(argv++);
    traj_out_file = *(argv++);
    map_out_file = *(argv++);
    int fps;
    fps = atoi(*(argv++));
    double width, height;
    width = atof(*(argv++));
    height = atof(*(argv++));
    double fx, fy, cx, cy;
    fx = atof(*(argv++));
    fy = atof(*(argv++));
    cx = atof(*(argv++));
    cy = atof(*(argv++));
    double d0, d1, d2, d3, d4;
    d0 = d1 = d2 = d3 = d4 = 0.0;
    if (argc >= 12) { d0 = atof(*(argv++));
    if (argc >= 13) { d1 = atof(*(argv++));
    if (argc >= 14) { d2 = atof(*(argv++));
    if (argc >= 15) { d3 = atof(*(argv++));
    if (argc >= 16) { d4 = atof(*(argv++)); }}}}}
    
    // run pipeline
    svo::BenchmarkNode benchmark(img_dir, traj_out_file, map_out_file,
                                    fps,
                                    width, height,
                                    fx, fy, cx, cy,
                                    d0, d1, d2, d3, d4);
    return benchmark.runFromFolder();
}
