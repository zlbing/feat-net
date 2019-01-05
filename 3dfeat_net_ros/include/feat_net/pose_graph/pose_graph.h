//
// Created by zzz on 19-1-1.
//

#ifndef INC_3DFEAT_NET_POSE_GRAPH_H
#define INC_3DFEAT_NET_POSE_GRAPH_H


#include <Eigen/Dense>
#include <gs/gs.h>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>

#include "feat_net/data_type.h"
#include "feat_net/ransac_match.h"
#include "feat_net/pose_graph/pose_graph_3d_error_term.h"
#include "feat_net/pose_graph/pose_graph_4dof_error.h"

namespace FeatNet{
  class PoseGraph {
  public:
    PoseGraph();

    explicit PoseGraph(PoseGraph const&) = delete;
    void operator=(PoseGraph const&) = delete;

    void addPose(Pose& pose);

    void addConstraint(int i, int j, Eigen::Matrix4d& delta);

    bool detectLoop(int index_i, int index_j, Eigen::Matrix4d& delta);

    bool detectFrameMatch(int index_i, int index_j, Eigen::Matrix4d& delta);

    void match(const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_left_vec,
               const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_right_vec,
               std::vector<int>& match12,
               std::vector<float>& error12);

    float makeDist(const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_l,
                   const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_r);

    Pose& getPoseFromIndex(int index);

    void optimize3DPoseGraph(int index_i, int index_j, Eigen::Matrix4d& delta);

    void optimize4DoFPoseGraph(int index_i, int index_j, Eigen::Matrix4d& delta);


    int getSizeOfTrajectory(){
      return trajectory_.size();
    }
    const std::vector<Pose>& getTracjectory(){
      return trajectory_;
    }
  private:
    std::vector<Pose> trajectory_;
    std::shared_ptr<RansacMatch> ransac_matcher_ptr_;

    VectorOf3DConstraints constraints;
  };
}



#endif //INC_3DFEAT_NET_POSE_GRAPH_H
