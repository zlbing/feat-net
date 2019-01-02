//
// Created by zzz on 18-12-29.
//

#include "feat_net/visualization/visualize.h"
namespace FeatNet {
  Visualize::Visualize() {
    ros::NodeHandle nh("~");
    pub_cloud_ = nh.advertise<sensor_msgs::PointCloud>("frame_cloud",10);
    pub_loop_cloud_ = nh.advertise<sensor_msgs::PointCloud>("loop_frame_cloud",10);

    pub_pose_path_ = nh.advertise<nav_msgs::Path>("frame_pose_path",10);
    pub_loop_pose_path_ = nh.advertise<nav_msgs::Path>("loop_frame_pose_path",10);

    pub_loop_pose_ = nh.advertise<geometry_msgs::PoseArray>("loop_frame_pose",10);
    pub_pose_ = nh.advertise<geometry_msgs::PoseArray>("frame_pose",10);
  };

  void Visualize::addFramePose(Eigen::Quaterniond& qua,
                               Eigen::Vector3d& position,
                               std::string frame_id){
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id=frame_id;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.pose.position.x = position.x();
    pose_stamped.pose.position.y = position.y();
    pose_stamped.pose.position.z = position.z();

    pose_stamped.pose.orientation.x = qua.x();
    pose_stamped.pose.orientation.y = qua.y();
    pose_stamped.pose.orientation.z = qua.z();
    pose_stamped.pose.orientation.w = qua.w();

    pose_path_.header = pose_stamped.header;
    pose_path_.poses.push_back(pose_stamped);

    frame_pose_.poses.push_back(pose_stamped.pose);
    frame_pose_.header = pose_stamped.header;
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

  void Visualize::addLoopPath(const std::vector<Pose>& trajectory_, std::string frame_id){
    loop_pose_path_.poses.clear();
    looped_frame_pose_.poses.clear();
    for(size_t i=0;i<trajectory_.size(); i++){
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.frame_id=frame_id;
      pose_stamped.header.stamp = ros::Time::now();
      pose_stamped.pose.position.x = trajectory_[i].position.x();
      pose_stamped.pose.position.y = trajectory_[i].position.y();
      pose_stamped.pose.position.z = trajectory_[i].position.z();
      pose_stamped.pose.orientation.x = trajectory_[i].rotation.x();
      pose_stamped.pose.orientation.y = trajectory_[i].rotation.y();
      pose_stamped.pose.orientation.z = trajectory_[i].rotation.z();
      pose_stamped.pose.orientation.w = trajectory_[i].rotation.w();
      loop_pose_path_.poses.push_back(pose_stamped);
      loop_pose_path_.header = pose_stamped.header;
      looped_frame_pose_.poses.push_back(pose_stamped.pose);
      looped_frame_pose_.header = pose_stamped.header;
    }
  }
  void Visualize::publish(){
    pub_cloud_.publish(frame_pointcloud_);
    pub_pose_path_.publish(pose_path_);
    pub_pose_.publish(frame_pose_);

    pub_loop_pose_path_.publish(loop_pose_path_);
    pub_loop_pose_.publish(looped_frame_pose_);
  }
}