//
// Created by zzz on 18-12-26.
//
#include <ros/ros.h>
#include <gs/gs.h>

#include "feat_net/feat_net_frame.h"

int main(int argc, char **argv) {
  gs::init(argc, argv, "Feat_tracking");

  ros::NodeHandle nh("~");
  std::string file_path;
  nh.param(std::string("file_path"),file_path,std::string("/home/data/featNetData_1"));

  FeatNet::FeatNetFrame feat_net_frames;
  bool load_success = feat_net_frames.LoadDataFromFile(file_path);
  if(load_success){
    feat_net_frames.feature_tracking();
  }else{
    std::cout<<"data file load failed"<<std::endl;
  }

  ros::spin();
  return 0;
}