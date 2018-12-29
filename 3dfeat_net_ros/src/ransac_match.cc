//
// Created by zzz on 18-12-29.
//

#include "ransac_match.h"

namespace FeatNet{

  RansacMatch::RansacMatch(int ransac_number){
    toolUniform_ = std::uniform_int_distribution<int>(0, 10000);
    ransac_number_ =ransac_number;
  };

  void RansacMatch::estimateRt(const std::vector<Eigen::Vector3f>& points_l,
                                const std::vector<Eigen::Vector3f>& points_r,
                                Eigen::Matrix3d& rotation,
                                Eigen::Vector3d& position){
//    std::cout<<"estimateRt in"<<std::endl;

    assert(points_l.size() == points_r.size());

    Eigen::Vector3f l_centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f r_centroid = Eigen::Vector3f::Zero();
    for(size_t i=0; i< points_l.size(); i++){
      l_centroid += points_l[i];
      r_centroid += points_r[i];
    }
    l_centroid = l_centroid/points_l.size();
    r_centroid = r_centroid/points_l.size();
//    std::cout<<"l_centroid="<<l_centroid.transpose()<<std::endl;
//    std::cout<<"r_centroid="<<r_centroid.transpose()<<std::endl;

    std::vector<Eigen::Vector3d> l_centrized, r_centrized;
    std::vector<Eigen::Vector3d> r12, r21;

    std::vector<Eigen::Matrix3d> r22;
    for(size_t i=0; i< points_l.size(); i++){
      l_centrized.push_back(points_l[i].cast<double>() - l_centroid.cast<double>());
      r_centrized.push_back(points_r[i].cast<double>() - r_centroid.cast<double>());
      r12.push_back(r_centrized.back() - l_centrized.back());
      r21.push_back(l_centrized.back() - r_centrized.back());
    }

//    for(size_t i=0; i< r12.size(); i++){
//      std::cout<<"r12["<<i<<"]="<<r12[i].transpose()<<std::endl;
//    }
//    for(size_t i=0; i< r21.size(); i++){
//      std::cout<<"r21["<<i<<"]="<<r21[i].transpose()<<std::endl;
//    }

    std::vector<Eigen::Vector3d> r22_1;
    for(size_t i=0; i < points_l.size(); i++){
      r22_1.push_back(l_centrized[i].cast<double>() + r_centrized[i].cast<double>());
//      std::cout<<"r22_1["<<i<<"]="<<r22_1.back().transpose()<<std::endl;
    }

    for(size_t i=0; i < points_l.size(); i++){
      Eigen::Matrix3d rr = Eigen::Matrix3d::Zero();
      rr(0,1) = -r22_1[i](2);
      rr(0,2) = r22_1[i](1);
      rr(1,0) = r22_1[i](2);
      rr(1,2) = -r22_1[i](0);
      rr(2,0) = -r22_1[i](1);
      rr(2,1) = r22_1[i](0);
      r22.push_back(rr);
//      std::cout<<"r22["<<i<<"]=\n"<<rr<<std::endl;
    }


    Eigen::Matrix4d B = Eigen::Matrix4d::Zero();
    for(size_t i=0; i < points_l.size(); i++){
      Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
      A.block<1,3>(0,1) = r12[i];
      A.block<3,1>(1,0) = r21[i];
      A.block<3,3>(1,1) = r22[i];
//      std::cout<<"A\n"<<A<<std::endl;
      B = B + A.transpose() * A;
//      std::cout<<"B\n"<<B<<std::endl;
    }
//    std::cout<<"B\n"<<B<<std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix4d V = svd.matrixV();
    Eigen::Vector4d qua_vec=V.block<4,1>(0,3);
//    std::cout<<"qua_vec="<<qua_vec.transpose()<<std::endl;
    Eigen::Quaterniond qua = Eigen::Quaterniond(qua_vec(0),qua_vec(1),qua_vec(2),qua_vec(3));
    Eigen::Matrix4d T1,T2,T3;
    T1 = Eigen::Matrix4d::Identity();
    T1.block<3,1>(0,3) = -r_centroid.cast<double>();
    T2.setZero();
    T2.block<3,3>(0,0) = Eigen::Matrix3d(qua);
    T2(3,3) = 1;
    T3 = Eigen::Matrix4d::Identity();
    T3.block<3,1>(0,3) = l_centroid.cast<double>();

    Eigen::Matrix4d T = T3 * T2 * T1;

    rotation = T.block<3,3>(0,0);
    position = T.block<3,1>(0,3);
//    std::cout<<"T1\n"<<T1<<"\nT2\n"<<T2<<"\nT3\n"<<T3<<std::endl;
//    std::cout<<"T\n"<<T<<"\nEps\n"<<svd.matrixU()<<std::endl;
  }

  void RansacMatch::getRandomSeq(int len, int max_value, std::vector<int>& seq){
    std::vector<bool> idx(max_value, false);
    seq.resize(len);
    for(size_t i=0; i<len; i++){
      int value = -1;
      while(true){
        value = toolUniform_(toolGenerator_) % max_value;
        if(idx[value])continue;
        idx[value] = true;
        break;
      }
      if(value < 0){
        std::cerr<<"RansacMatch::getRandomSeq random error"<<std::endl;
      }
      seq[i]=value;
    }
  }

  bool RansacMatch::ransac(const std::vector<Eigen::Vector3f>& left_points,
                            const std::vector<Eigen::Vector3f>& right_points,
                            float error,
                            Eigen::Matrix4d& delta){
    std::cout<<"ransac left_points size="<<left_points.size()<<" right_points="<<right_points.size()<<std::endl;

    std::vector<int> ransac_vec;
    std::vector<Eigen::Vector3f> ransac_l_points,ransac_r_points;

    int iter =0;
    Eigen::Matrix3d best_rotation;
    Eigen::Vector3d best_position;

    int max_inliers = 0;
    std::vector<bool> best_inliers;
    int N =1;
    double p = 0.99;
    while(iter < N){
      ransac_l_points.clear();
      ransac_r_points.clear();

      getRandomSeq(ransac_number_, left_points.size(), ransac_vec);

      Eigen::Matrix3d rotation;
      Eigen::Vector3d position;

      for(size_t i=0; i< ransac_vec.size(); i++){
        ransac_l_points.push_back(left_points[ransac_vec[i]]);
        ransac_r_points.push_back(right_points[ransac_vec[i]]);
      }

      estimateRt(ransac_l_points, ransac_r_points, rotation, position);
      std::vector<bool> inliers;
      euc3Ddist(left_points,right_points,rotation,position,inliers,error);
      int inliers_number =0;
      for(size_t i=0; i<inliers.size(); i++){
        if(inliers[i])inliers_number++;
      }
      if(inliers_number >= max_inliers){
        max_inliers =inliers_number;
        best_inliers = inliers;
        best_rotation = rotation;
        best_position = position;

        double fracinliers = inliers_number * 1.0 / left_points.size();
        double NoOutliers = 1 - std::pow(fracinliers,ransac_number_);
        NoOutliers = std::max(EPS, NoOutliers);
        NoOutliers = std::min(1-EPS, NoOutliers);
        N = std::log(1-p)/std::log(NoOutliers);
        N = std::max(N,10);

      }
      iter++;
    }

    std::cout<<"\nmax_inliers="<<max_inliers<<" N="<<N<<std::endl;
    std::cout<<"best_rotation="<<best_rotation<<std::endl;
    std::cout<<"best_position="<<best_position.transpose()<<std::endl;

    if(max_inliers < ransac_number_){
      return false;
    }else{
      ransac_l_points.clear();
      ransac_r_points.clear();
      for(size_t i=0; i < best_inliers.size();i++){
        if(best_inliers[i]){
          std::cout<<i+1<<" ";
          ransac_l_points.push_back(left_points[i]);
          ransac_r_points.push_back(right_points[i]);
        }
      }
      std::cout<<std::endl;
      estimateRt(ransac_l_points, ransac_r_points, best_rotation, best_position);

      euc3Ddist(left_points,right_points,best_rotation,best_position,best_inliers,error);

      std::cout<<"max_inliers="<<max_inliers
               <<"iter size="<<iter
               <<"\nbest_rotation\n"<<best_rotation
               <<"\nbest_position\n"<<best_position.transpose()<<std::endl;
    }

    delta = Eigen::Matrix4d::Identity();
    delta.block<3,3>(0,0) = best_rotation;
    delta.block<3,1>(0,3) = best_position;
    return true;
  }

  void RansacMatch::euc3Ddist(const std::vector<Eigen::Vector3f>& left_points,
                               const std::vector<Eigen::Vector3f>& right_points,
                               Eigen::Matrix3d& rotation,
                               Eigen::Vector3d& position,
                               std::vector<bool>& inliers,
                               float error_threshold){
    inliers.resize(left_points.size(),false);
    for(size_t i=0; i< left_points.size(); i++){
      float error = (left_points[i] - rotation.cast<float>() * right_points[i] - position.cast<float>()).norm();
      if(error < error_threshold){
        inliers[i] = true;
      }
    }
  }
}