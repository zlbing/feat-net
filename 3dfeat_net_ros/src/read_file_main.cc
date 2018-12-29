//
// Created by zzz on 18-12-26.
//
#include <ros/ros.h>
#include "feat_net_frame.h"
#include <gs/gs.h>
int main(int argc, char **argv) {
  gs::init(argc, argv, "Feat_tracking");

  ros::NodeHandle nh("~");

  FeatNet::FeatNetFrame feat_net_frames;
  bool load_success = feat_net_frames.LoadDataFromFile("/home/data/featNetData");
  if(load_success){
    feat_net_frames.feature_tracking();
  }else{
    std::cout<<"data file load failed"<<std::endl;
  }

  ros::spin();
  return 0;
}