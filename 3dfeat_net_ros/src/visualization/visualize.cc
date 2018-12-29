//
// Created by zzz on 18-12-29.
//

#include "visualization/visualize.h"
namespace FeatNet {
  Visualize::Visualize() {
    ros::NodeHandle nh("~");
    pub_cloud_ = nh.advertise<sensor_msgs::PointCloud>("frame_cloud",10);
    pub_pose_path_ = nh.advertise<nav_msgs::Path>("frame_pose",10);
  };

  void Visualize::addFramePose(Eigen::Matrix3d& rotation,
                               Eigen::Vector3d& position,
                               std::string frame_id){
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id=frame_id;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.pose.position.x = position.x();
    pose_stamped.pose.position.y = position.y();
    pose_stamped.pose.position.z = position.z();
    Eigen::Quaterniond qua(rotation);
    pose_stamped.pose.orientation.x = qua.x();
    pose_stamped.pose.orientation.y = qua.y();
    pose_stamped.pose.orientation.z = qua.z();
    pose_stamped.pose.orientation.w = qua.w();

    pose_path_.header = pose_stamped.header;
    pose_path_.poses.push_back(pose_stamped);
  }

  void Visualize::addFrameCloud(std::vector<Eigen::Vector3f>& point_cloud,
                                std::string frame_id){
    frame_pointcloud_.points.clear();
    frame_pointcloud_.header.frame_id = frame_id;
    frame_pointcloud_.header.stamp = ros::Time::now();
    for(size_t i=0; i < point_cloud.size(); i++){
      geometry_msgs::Point32 p;
      p.x = point_cloud[i].x();
      p.y = point_cloud[i].y();
      p.z = point_cloud[i].z();
      frame_pointcloud_.points.push_back(p);
    }
  }
  void Visualize::publish(){
    pub_cloud_.publish(frame_pointcloud_);
    pub_pose_path_.publish(pose_path_);
  }
}