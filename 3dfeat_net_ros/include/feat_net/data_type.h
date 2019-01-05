//
// Created by zzz on 19-1-1.
//

#ifndef INC_3DFEAT_NET_DATA_TYPE_H
#define INC_3DFEAT_NET_DATA_TYPE_H
#include <Eigen/Dense>

namespace FeatNet{

#define FEATURE_DIM 32

  struct Frame{
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> features;
    std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> > descriptors;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d position;
  };

  class Pose{
  public:
    Eigen::Quaterniond rotation;
    Eigen::Vector3d position;
    std::vector<Eigen::Vector3f> features;
    std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> > descriptors;
    std::vector<Eigen::Vector3f> points;
    int id;
  };

  struct Pose3d {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;

    // The name of the data type in the g2o file format.
    static std::string name() {
      return "VERTEX_SE3:QUAT";
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  // The constraint between two vertices in the pose graph. The constraint is the
  // transformation from vertex id_begin to vertex id_end.
  struct Constraint3d {
    int id_begin;
    int id_end;

    // The transformation that represents the pose of the end frame E w.r.t. the
    // begin frame B. In other words, it transforms a vector in the E frame to
    // the B frame.
    Pose3d t_be;

    // The inverse of the covariance matrix for the measurement. The order of the
    // entries are x, y, z, delta orientation.
    Eigen::Matrix<double, 6, 6> information;

    // The name of the data type in the g2o file format.
    static std::string name() {
      return "EDGE_SE3:QUAT";
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >
          VectorOf3DConstraints;
};
#endif //INC_3DFEAT_NET_DATA_TYPE_H
