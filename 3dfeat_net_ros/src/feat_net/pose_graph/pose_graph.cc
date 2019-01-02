//
// Created by zzz on 19-1-1.
//

#include "feat_net/pose_graph/pose_graph.h"
namespace FeatNet{
  PoseGraph::PoseGraph(){
    ransac_matcher_ptr_ = std::make_shared<RansacMatch>();
  }

  void PoseGraph::addPose(Pose& pose){
    trajectory_.push_back(pose);
//    for(int i=trajectory_.size() -2; i>=0; i--){
//      Eigen::Matrix4d delta;
//      if(detectFrameMatch(i,trajectory_.size()-1,delta)){
//        Constraint3d constraint;
//        constraint.id_begin = i;
//        constraint.id_end = trajectory_.size()-1;
//        Pose3d pose3d;
//        pose3d.p = delta.block<3,1>(0,3);
//        pose3d.q = delta.block<3,3>(0,0);
//        constraint.t_be = pose3d;
//        constraint.information = Eigen::Matrix<double, 6, 6>::Identity();
//        constraints.push_back(constraint);
//      }else{
//        break;
//      }
//    }
  }

  bool PoseGraph::detectFrameMatch(int index_i, int index_j, Eigen::Matrix4d& delta){
    Pose& pose_i = getPoseFromIndex(index_i);
    Pose& pose_j = getPoseFromIndex(index_j);

    std::vector<int> match12;
    std::vector<float> error12;
    match(pose_i.descriptors,
          pose_j.descriptors,
          match12,
          error12);

    std::vector<Eigen::Vector3f> selected_right_features;
    std::vector<Eigen::Vector3f> selected_left_features;

    for(size_t i=0; i < match12.size(); i++){
      selected_left_features.push_back(pose_i.features[i]);
      selected_right_features.push_back(pose_j.features[match12[i]]);
    }

    int inliers = ransac_matcher_ptr_->ransac(selected_left_features, selected_right_features,1,delta);

    if(inliers < 20){
      return false;
    }else{
      return true;
    }
  }
  bool PoseGraph::detectLoop(int index_i, int index_j, Eigen::Matrix4d& delta){

    if(fabs(index_i - index_j)<10){
      return false;
    }

    Pose& pose_i = getPoseFromIndex(index_i);
    Pose& pose_j = getPoseFromIndex(index_j);

    std::vector<int> match12;
    std::vector<float> error12;
    match(pose_i.descriptors,
          pose_j.descriptors,
          match12,
          error12);

    std::vector<Eigen::Vector3f> selected_right_features;
    std::vector<Eigen::Vector3f> selected_left_features;

    for(size_t i=0; i < match12.size(); i++){
      selected_left_features.push_back(pose_i.features[i]);
      selected_right_features.push_back(pose_j.features[match12[i]]);
    }

    int inliers = ransac_matcher_ptr_->ransac(selected_left_features, selected_right_features,1,delta);

    if(inliers < 25){
      return false;
    }else{
      return true;
    }
  }

  void PoseGraph::optimize3DPoseGraph(int index_i, int index_j, Eigen::Matrix4d& delta){
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);

    ceres::LocalParameterization* quaternion_local_parameterization =
            new ceres::EigenQuaternionParameterization;

    for(int i = index_i; i <= index_j; i++){
      problem.AddParameterBlock(trajectory_[i].rotation.coeffs().data(),4,quaternion_local_parameterization);
      problem.AddParameterBlock(trajectory_[i].position.data(),3);
      for(int j=1; j< 5; j++){
        if(i - j >0){
          Pose3d pose3d;
          Eigen::Vector3d delta_position = trajectory_[i-j].rotation.inverse() * (trajectory_[i].position - trajectory_[i-j].position);
          Eigen::Quaterniond delta_rotation = trajectory_[i-j].rotation.inverse() * trajectory_[i].rotation;
          pose3d.p = delta_position;
          pose3d.q = delta_rotation;
          ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(pose3d,
                                                                            Eigen::Matrix<double, 6, 6>::Identity());

          problem.AddResidualBlock(cost_function, loss_function,
                                   trajectory_[i-j].position.data(),
                                   trajectory_[i-j].rotation.coeffs().data(),
                                   trajectory_[i].position.data(),
                                   trajectory_[i].rotation.coeffs().data());


        }
      }
    }

//    for(int i=0; i<constraints.size(); i++){
//      const Constraint3d& constraint = constraints[i];
//      if(!(constraint.id_begin>=index_i && constraint.id_begin <= index_j)){
//        continue;
//      }
//      if(!(constraint.id_end>=index_i && constraint.id_end <= index_j)){
//        continue;
//      }
//      ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(constraint.t_be,
//                                                                        constraint.information);
//
//      problem.AddResidualBlock(cost_function, loss_function,
//                                   trajectory_[constraint.id_begin].position.data(),
//                                   trajectory_[constraint.id_begin].rotation.coeffs().data(),
//                                   trajectory_[constraint.id_end].position.data(),
//                                   trajectory_[constraint.id_end].rotation.coeffs().data());
//    }

    //add loop residual
    Pose3d pose3d;
    pose3d.p=delta.block<3,1>(0,3);
    pose3d.q=delta.block<3,3>(0,0);
    ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(pose3d, Eigen::Matrix<double, 6, 6>::Identity() * 5);
    problem.AddResidualBlock(cost_function, loss_function,
                             trajectory_[index_i].position.data(),
                             trajectory_[index_i].rotation.coeffs().data(),
                             trajectory_[index_j].position.data(),
                             trajectory_[index_j].rotation.coeffs().data());

    for(int i=index_i; i<= index_i+5; i++){
      problem.SetParameterBlockConstant(trajectory_[i].position.data());
      problem.SetParameterBlockConstant(trajectory_[i].rotation.coeffs().data());
    }

    for(size_t i=0; i< trajectory_.size(); i++){
      GS_INFO("position[%d]=%lf %lf %lf",i,
              trajectory_[i].position.x(),
              trajectory_[i].position.y(),
              trajectory_[i].position.z());
    }
    ceres::Solve(options, &problem, &summary);

    GS_INFO("%s",summary.FullReport().c_str());

    for(size_t i=0; i< trajectory_.size(); i++){
      GS_INFO("position[%d]=%lf %lf %lf",i,
              trajectory_[i].position.x(),
              trajectory_[i].position.y(),
              trajectory_[i].position.z());
    }
  }

  void PoseGraph::optimize4DoFPoseGraph(int index_i, int index_j, Eigen::Matrix4d& delta){
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization* angle_local_parameterization =
            ceres::AngleLocalParameterization::Create();
    int max_length = index_j+1;
    double t_array[max_length][3]; ////(x,y,z)
    Eigen::Quaterniond q_array[max_length]; ////q
    double euler_array[max_length][3];
    for(int i=index_i; i<=index_j; i++){

      t_array[i][0] = trajectory_[i].position(0);
      t_array[i][1] = trajectory_[i].position(1);
      t_array[i][2] = trajectory_[i].position(2);
      q_array[i] = trajectory_[i].rotation;

      Eigen::Vector3d euler_angle = ceres::R2ypr(q_array[i].toRotationMatrix());
      euler_array[i][0] = euler_angle.x();
      euler_array[i][1] = euler_angle.y();
      euler_array[i][2] = euler_angle.z();

      problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
      problem.AddParameterBlock(t_array[i], 3);

      if (i <= index_i+5)
      {
        problem.SetParameterBlockConstant(euler_array[i]);
        problem.SetParameterBlockConstant(t_array[i]);
      }

      for(int j=1; j< 5; j++) {
        if (i - j > 0) {
          Eigen::Vector3d euler_conncected =  ceres::R2ypr(q_array[i-j].toRotationMatrix());
          Eigen::Vector3d relative_t(t_array[i][0] - t_array[i-j][0], t_array[i][1] - t_array[i-j][1], t_array[i][2] - t_array[i-j][2]);
          relative_t = q_array[i-j].inverse() * relative_t;
          double relative_yaw = euler_array[i][0] - euler_array[i-j][0];
          ceres::CostFunction* cost_function = ceres::FourDOFError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                                     relative_yaw, euler_conncected.y(), euler_conncected.z());
          problem.AddResidualBlock(cost_function, NULL, euler_array[i-j],
                                   t_array[i-j],
                                   euler_array[i],
                                   t_array[i]);
        }
      }
    }


    Eigen::Vector3d euler_conncected = ceres::R2ypr(q_array[index_i].toRotationMatrix());
    Eigen::Vector3d relative_t;
    relative_t = delta.block<3,1>(0,3);
    double relative_yaw = ceres::R2ypr(delta.block<3,3>(0,0))[0];
    ceres::CostFunction* cost_function = ceres::FourDOFWeightError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                                     relative_yaw, euler_conncected.y(), euler_conncected.z());
    problem.AddResidualBlock(cost_function, loss_function,
                             euler_array[index_i],
                             t_array[index_i],
                             euler_array[index_j],
                             t_array[index_j]);


    ceres::Solve(options, &problem, &summary);
    GS_INFO("%s",summary.FullReport().c_str());

    for(int i=index_i; i<=index_j; i++){
      Eigen::Quaterniond tmp_q;
      tmp_q = ceres::ypr2R(Eigen::Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
      Eigen::Vector3d tmp_t = Eigen::Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
      trajectory_[i].position = tmp_t;
      trajectory_[i].rotation = tmp_q;
    }
  }
  void PoseGraph::match(const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_left_vec,
                        const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_right_vec,
                        std::vector<int>& match12,
                        std::vector<float>& error12){

//    std::cout<<"match in"<<std::endl;
//    std::cout<<"desc_left_vec size="<<desc_left_vec.size()<<std::endl;
//    std::cout<<"desc_right_vec size="<<desc_right_vec.size()<<std::endl;

    match12.resize(desc_left_vec.size());
    error12.resize(desc_left_vec.size());

    for(size_t i=0; i < desc_left_vec.size(); i++){
      int index =-1;
      float error = 1e5;
      for(size_t j=0; j < desc_right_vec.size(); j++){
        float error_i_j = makeDist(desc_left_vec[i], desc_right_vec[j]);
        if(error > error_i_j){
          error = error_i_j;
          index = j;
        }
      }
//      std::cout<<"index="<<index<<" error="<<error<<std::endl;
      match12[i]=index;
      error12[i]=error;
    }
//    std::cout<<"match out"<<std::endl;
  }
  float PoseGraph::makeDist(const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_l,
                            const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_r){
    double error=0;
    for(int i=0; i<desc_l.rows(); i++){
      error += (desc_l(i,0) - desc_r(i,0)) * (desc_l(i,0) - desc_r(i,0));
    }
    return std::sqrt(error);
  }

  Pose& PoseGraph::getPoseFromIndex(int index){
    for(size_t i=0; i < trajectory_.size(); i++){
      if(trajectory_[i].id == index){
        return trajectory_[i];
      }
    }
  }
}