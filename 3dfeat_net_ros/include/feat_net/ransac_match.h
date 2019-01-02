//
// Created by zzz on 18-12-29.
//

#ifndef INC_3DFEAT_NET_RANSAC_MATCH_H
#define INC_3DFEAT_NET_RANSAC_MATCH_H
#include <Eigen/Dense>
#include <iostream>
namespace FeatNet{
#define EPS 2.22e-16
  class RansacMatch {
  public:
    RansacMatch(int ransac_number=3);
    explicit RansacMatch(RansacMatch const&) = delete;
    void operator=(RansacMatch const&) = delete;

    int ransac(const std::vector<Eigen::Vector3f>& left_points,
                const std::vector<Eigen::Vector3f>& right_points,
                float error,
                Eigen::Matrix4d& delta);

    void estimateRt(const std::vector<Eigen::Vector3f>& points_l,
                    const std::vector<Eigen::Vector3f>& points_r,
                    Eigen::Matrix3d& rotation,
                    Eigen::Vector3d& position);

    void getRandomSeq(int len, int max_value, std::vector<int>& seq);

    void euc3Ddist(const std::vector<Eigen::Vector3f>& left_points,
                   const std::vector<Eigen::Vector3f>& right_points,
                   Eigen::Matrix3d& rotation,
                   Eigen::Vector3d& position,
                   std::vector<bool>& inliers,
                   float error_threshold);
  private:
    std::uniform_int_distribution<int> toolUniform_;
    std::mt19937 toolGenerator_;
    int ransac_number_;
  };
}



#endif //INC_3DFEAT_NET_RANSAC_MATCH_H
