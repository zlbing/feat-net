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
#include <eigen3/Eigen/Dense>

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
  bool LoadDataFromFile();
  void feature_tracking();
  void match(const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_left_vec,
             const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_right_vec,
             std::vector<int>& match12,
             std::vector<float>& error12);
  float makeDist(const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_l,
                 const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_r);

  void estimateRt(const std::vector<Eigen::Vector3f>& points_l,
                  const std::vector<Eigen::Vector3f>& points_r,
                  Eigen::Matrix3d& rotation,
                  Eigen::Vector3d& position);

  void getRandomSeq(int len, int max_value, std::vector<int>& seq);


  void ransac(const std::vector<Eigen::Vector3f>& left_points,
              const std::vector<Eigen::Vector3f>& right_points,
              float error);

  void euc3Ddist(const std::vector<Eigen::Vector3f>& left_points,
                 const std::vector<Eigen::Vector3f>& right_points,
                 Eigen::Matrix3d& rotation,
                 Eigen::Vector3d& position,
                 std::vector<bool>& inliers,
                 float error_threshold);
private:
  std::string file_path_;
  std::string points_file_path_;
  std::string transform_file_path_;
  std::string desc_file_path_;

  Eigen::Matrix3d cur_rotation_;
  Eigen::Vector3d cur_position_;

  Frame* cur_frame_ptr_;
  Frame* pre_frame_ptr_;
  std::vector<Frame> features_frames_;

  std::uniform_int_distribution<int> toolUniform_;
  std::mt19937 toolGenerator_;
};
}

#endif //INC_3DFEAT_NET_ROS_FEATNETFRAME_H
