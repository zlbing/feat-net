//
// Created by zzz on 18-12-29.
//

#ifndef INC_3DFEAT_NET_ROS_VISUALIZE_H
#define INC_3DFEAT_NET_ROS_VISUALIZE_H
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
namespace FeatNet{

class Visualize {
public:
  Visualize();
  explicit Visualize(Visualize const&) = delete;
  void operator=(Visualize const&) = delete;

private:
  nav_msgs::Path pose_path_;
};

};
#endif //INC_3DFEAT_NET_ROS_VISUALIZE_H
