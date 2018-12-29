//
// Created by zzz on 18-12-26.
//

#ifndef INC_3DFEAT_NET_ROS_FEATNETFRAME_H
#define INC_3DFEAT_NET_ROS_FEATNETFRAME_H
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <Eigen/Core>
#include "ransac_match.h"
#include "visualization/visualize.h"

namespace FeatNet{

#define FEATURE_DIM 32

struct Frame{
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> features;
  std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> > descriptors;
  Eigen::Matrix3d rotation;
  Eigen::Vector3d position;
};

class FeatNetFrame {
public:
  static FeatNetFrame& Instance(){
    static FeatNetFrame instance;
    return instance;
  };

  FeatNetFrame();
  explicit FeatNetFrame(FeatNetFrame const&) = delete;
  void operator=(FeatNetFrame const&) = delete;
  bool LoadDataFromFile(std:: string file_path);
  void feature_tracking();
  void match(const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_left_vec,
             const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_right_vec,
             std::vector<int>& match12,
             std::vector<float>& error12);
  float makeDist(const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_l,
                 const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_r);

  void publish();
private:

  Eigen::Matrix3d cur_rotation_;
  Eigen::Vector3d cur_position_;

  Frame* cur_frame_ptr_;
  Frame* pre_frame_ptr_;
  std::vector<Frame> features_frames_;

  std::shared_ptr<RansacMatch> ransac_matcher_;
  std::shared_ptr<Visualize> visual_;
};
}

#endif //INC_3DFEAT_NET_ROS_FEATNETFRAME_H
