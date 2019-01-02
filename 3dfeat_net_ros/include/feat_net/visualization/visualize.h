//
// Created by zzz on 18-12-29.
//

#ifndef INC_3DFEAT_NET_ROS_VISUALIZE_H
#define INC_3DFEAT_NET_ROS_VISUALIZE_H
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseArray.h>

#include <Eigen/Dense>

#include "feat_net/data_type.h"

namespace FeatNet{

class Visualize {
public:
  Visualize();
  explicit Visualize(Visualize const&) = delete;
  void operator=(Visualize const&) = delete;

  void addFramePose(Eigen::Quaterniond& qua,
                    Eigen::Vector3d& position,
                    std::string frame_id);

  void addFrameCloud(std::vector<Eigen::Vector3f>& point_cloud,
                     std::string frame_id);

  void addLoopPath(const std::vector<Pose>& trajectory_,
                   std::string frame_id);

  void publish();

private:
  nav_msgs::Path pose_path_;

  nav_msgs::Path loop_pose_path_;

  sensor_msgs::PointCloud frame_pointcloud_;

  geometry_msgs::PoseArray looped_frame_pose_,frame_pose_;

  ros::Publisher pub_pose_path_,pub_loop_pose_path_;

  ros::Publisher pub_cloud_;

  ros::Publisher pub_pose_,pub_loop_pose_;
};

};
#endif //INC_3DFEAT_NET_ROS_VISUALIZE_H
