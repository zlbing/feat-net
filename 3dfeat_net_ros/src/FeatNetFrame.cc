//
// Created by zzz on 18-12-26.
//

#include "FeatNetFrame.h"
namespace FeatNet{

  FeatNetFrame::FeatNetFrame(){
    file_path_="/home/data/featNetData";
    points_file_path_ = file_path_ + "/use_data";
    transform_file_path_ = file_path_ + "/pose.txt";
    desc_file_path_ = file_path_ + "/use_data/results";

    toolUniform_ = std::uniform_int_distribution<int>(0, 10000);
  }

  bool FeatNetFrame::LoadDataFromFile(){
    namespace bf = boost::filesystem;
    {
      //get all points from file
      std::vector<bf::path> file_vec;
      if(bf::is_directory(points_file_path_)){
        std::copy(bf::directory_iterator(points_file_path_), bf::directory_iterator(), std::back_inserter(file_vec));
        std::sort(file_vec.begin(), file_vec.end());
      }

      for(std::vector<bf::path>::const_iterator it(file_vec.begin()), it_end(file_vec.end()); it != it_end; ++it){
        if(it->string().compare(it->string().size()-4, it->string().size(), ".bin")){
          //if it equals .bin, and will return zero
          continue;
        }
        std::ifstream data_file;
        data_file.open((*it).c_str(), std::ios::in | std::ios::binary);
        if(data_file.is_open()){
          Frame frame;
          Eigen::Vector3f point_data;
          while(data_file.read((char*)point_data.data(), 3 * sizeof(typename Eigen::Vector3f::Scalar))){
            frame.points.push_back(point_data);
          }
          data_file.close();
          features_frames_.push_back(frame);
        }else{
          std::cerr<<"data_file="<<(*it)<<" is not opened"<<std::endl;
          exit(-1);
        }
      }
      std::cout<<"Load points file size="<<features_frames_.size()<<std::endl;
    }

    {
      //get frame pose from file
      std::ifstream pose_data_file;
      Eigen::Matrix3d rotation;
      Eigen::Vector3d position;
      pose_data_file.open(transform_file_path_.c_str(), std::ios::in);
      if(pose_data_file.is_open()){
        std::string str;
        int frame_index=0;
        while(std::getline(pose_data_file, str, '\n')){
          std::vector<std::string> str_vec;
          boost::split(str_vec,str,boost::is_any_of(","));

          rotation = Eigen::Quaterniond(boost::lexical_cast<double>(str_vec[0]),
                                        boost::lexical_cast<double>(str_vec[1]),
                                        boost::lexical_cast<double>(str_vec[2]),
                                        boost::lexical_cast<double>(str_vec[3])).toRotationMatrix();
          position = Eigen::Vector3d(boost::lexical_cast<double>(str_vec[4]),
                                     boost::lexical_cast<double>(str_vec[5]),
                                     boost::lexical_cast<double>(str_vec[6]));
          features_frames_[frame_index].rotation = rotation;
          features_frames_[frame_index].position = position;
          frame_index++;
        }
        pose_data_file.close();
      }else{
        std::cerr<<"pose_data_file="<<transform_file_path_<<" is not opened"<<std::endl;
        exit(-1);
      }
    }

    {
      //get descriptors and features
      std::vector<bf::path> file_vec;
      file_vec.clear();
      if(bf::is_directory(desc_file_path_)){
        std::copy(bf::directory_iterator(desc_file_path_), bf::directory_iterator(), std::back_inserter(file_vec));
        std::sort(file_vec.begin(), file_vec.end());
      }

      for(size_t index=0; index < file_vec.size(); index++) {
        if (file_vec[index].string().compare(file_vec[index].string().size() - 4, file_vec[index].string().size(),
                                             ".bin")) {
          //if it equals .bin, and will return zero
          continue;
        }
        std::ifstream data_file;
        data_file.open(file_vec[index].string().c_str(), std::ios::in | std::ios::binary);
        Eigen::Matrix<float, FEATURE_DIM+3, 1> point_data;

        while(data_file.read((char*)point_data.data(), (FEATURE_DIM+3) * sizeof(float))){
          features_frames_[index].features.push_back(point_data.block<3,1>(0,0));
          features_frames_[index].descriptors.push_back(point_data.block<FEATURE_DIM,1>(3,0));
        }

//        std::cout<<"frame index="<<index<<" feature size="<<features_frames_[index].features.size()<<std::endl;
      }
    }
  }


  void FeatNetFrame::feature_tracking(){

    for(size_t i=0; i< features_frames_.size(); i++){
      if(i==0){
        pre_frame_ptr_ = &features_frames_[i];
        cur_frame_ptr_ = &features_frames_[i];
        cur_rotation_ = cur_frame_ptr_->rotation;
        cur_position_ = cur_frame_ptr_->position;
        continue;
      }
      cur_frame_ptr_ = &features_frames_[i];
      std::vector<int> match12;
      std::vector<float> error12;

      match(pre_frame_ptr_->descriptors,
            cur_frame_ptr_->descriptors,
            match12,
            error12);

      std::vector<Eigen::Vector3f> selected_features;
      for(size_t i=0; i < match12.size(); i++){
        selected_features.push_back(cur_frame_ptr_->features[match12[i]]);
//        std::cout<<"i="<<i<<" j="<<match12[i]<<std::endl;
      }

      ransac(pre_frame_ptr_->features, selected_features,1);
      pre_frame_ptr_ =cur_frame_ptr_;
    }


  }

  void FeatNetFrame::match(const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_left_vec,
                           const std::vector< Eigen::Matrix<float, FEATURE_DIM, 1> >& desc_right_vec,
                           std::vector<int>& match12,
                           std::vector<float>& error12){
    std::cout<<"match in"<<std::endl;
    std::cout<<"desc_left_vec size="<<desc_left_vec.size()<<std::endl;
    std::cout<<"desc_right_vec size="<<desc_right_vec.size()<<std::endl;

    match12.resize(desc_left_vec.size());
    error12.resize(desc_left_vec.size());

    for(size_t i=0; i < desc_left_vec.size(); i++){
      int index =-1;
      float error = 1e5;
//      std::cout<<"desc_left_vec["<<i<<"]="<<desc_left_vec[i].transpose()<<std::endl;
      for(size_t j=0; j < desc_right_vec.size(); j++){
//        std::cout<<"desc_left_vec["<<j<<"]="<<desc_right_vec[j].transpose()<<std::endl;

        float error_i_j=makeDist(desc_left_vec[i], desc_right_vec[j]);
        if(error > error_i_j){
          error = error_i_j;
          index = j;
        }
      }
//      std::cout<<"index="<<index<<" error="<<error<<std::endl;
      match12[i]=index;
      error12[i]=error;
    }
    std::cout<<"match out"<<std::endl;
  }

  float FeatNetFrame::makeDist(const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_l,
                              const Eigen::Matrix<float, FEATURE_DIM, 1>& desc_r){
    double error=0;
    for(int i=0; i<desc_l.rows(); i++){
      error += (desc_l(i,0) - desc_r(i,0)) * (desc_l(i,0) - desc_r(i,0));
    }
    return std::sqrt(error);
  }

  void FeatNetFrame::estimateRt(const std::vector<Eigen::Vector3f>& points_l,
                                const std::vector<Eigen::Vector3f>& points_r,
                                Eigen::Matrix3d& rotation,
                                Eigen::Vector3d& position){
//    std::cout<<"estimateRt in"<<std::endl;

    assert(points_l.size() == points_r.size());

    Eigen::Vector3f l_centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f r_centroid = Eigen::Vector3f::Zero();
    for(size_t i=0; i< points_l.size(); i++){
      l_centroid(i) = points_l[i].mean();
      r_centroid(i) = points_r[i].mean();
    }

//    std::cout<<"l_centroid="<<l_centroid.transpose()<<std::endl;
//    std::cout<<"r_centroid="<<r_centroid.transpose()<<std::endl;

    std::vector<Eigen::Vector3d> l_centrized, r_centrized;
    std::vector<Eigen::Vector3d> r12, r21;

    std::vector<Eigen::Matrix3d> r22;
    for(size_t i=0; i< points_l.size(); i++){
      l_centrized.push_back(points_l[i].cast<double>() - Eigen::Vector3f(l_centroid(i),l_centroid(i),l_centroid(i)).cast<double>());
      r_centrized.push_back(points_r[i].cast<double>() - Eigen::Vector3f(r_centroid(i),r_centroid(i),r_centroid(i)).cast<double>());
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
      rr(0,1) = -r22_1[2](i);
      rr(0,2) = r22_1[1](i);
      rr(1,0) = r22_1[2](i);
      rr(1,2) = -r22_1[0](i);
      rr(2,0) = -r22_1[1](i);
      rr(2,1) = r22_1[0](i);
      r22.push_back(rr);
//      std::cout<<"r22["<<i<<"]=\n"<<rr<<std::endl;
    }


    Eigen::Matrix4d B = Eigen::Matrix4d::Zero();
    for(size_t i=0; i < points_l.size(); i++){
      Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
      Eigen::Vector3d r12_rowi(r12[0][i],r12[1][i],r12[2][i]);
      Eigen::Vector3d r21_coli(r21[0][i],r21[1][i],r21[2][i]);

      A.block<1,3>(0,1) = r12_rowi;
      A.block<3,1>(1,0) = r21_coli;
      A.block<3,3>(1,1) = r22[i];
//      std::cout<<"A\n"<<A<<std::endl;
      B = B + A.transpose() * A;
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

  void FeatNetFrame::getRandomSeq(int len, int max_value, std::vector<int>& seq){
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
      seq[i]=value;
    }
  }

  void FeatNetFrame::ransac(const std::vector<Eigen::Vector3f>& left_points,
                            const std::vector<Eigen::Vector3f>& right_points,
                            float error){
    std::cout<<"ransac left_points size="<<left_points.size()<<" right_points="<<right_points.size()<<std::endl;
    int ransac_number = 3;
    std::vector<int> ransac_vec;
    std::vector<Eigen::Vector3f> ransac_l_points,ransac_r_points;

/*
 *
 * data
    5.5900    6.3590   12.2966
    2.1838   -4.0179   -6.9858
    0.8632    1.0971    3.4053

    3.8301    3.9086    7.6601
    2.1458   -7.5521  -13.0683
    0.6632    2.6165    3.6896

x_centroid    8.0819
   -2.9400
    1.7885

y_centroid    5.1329
   -6.1582
    2.3231


R12    1.1890    3.1802   -0.7346
    0.4985   -0.3159    0.9848
   -1.6876   -2.8643   -0.2502


R21   -1.1890   -0.4985    1.6876
   -3.1802    0.3159    2.8643
    0.7346   -0.9848    0.2502

R22_1   -3.7947   -2.9471    6.7419
   13.4277   -2.4718  -10.9559
   -2.5853   -0.3980    2.9833

R22
(:,:,1) =

         0    2.5853   13.4277
   -2.5853         0    3.7947
  -13.4277   -3.7947         0


(:,:,2) =

         0    0.3980   -2.4718
   -0.3980         0    2.9471
    2.4718   -2.9471         0


(:,:,3) =

         0   -2.9833  -10.9559
    2.9833         0   -6.7419
   10.9559    6.7419         0

estimate Rigid transform
 B
 24.5002    7.0841   -6.5051  -63.6703
    7.0841  326.6966  125.9907  -31.0569
   -6.5051  125.9907  102.6985   64.4846
  -63.6703  -31.0569   64.4846  376.5548
T
    0.8591   -0.4151   -0.2994    1.8113
    0.3699    0.9079   -0.1973    1.2106
    0.3537    0.0588    0.9335   -1.8335
         0         0         0    1.0000
 */
//    ransac_l_points.push_back(Eigen::Vector3f(5.5900,    6.3590,   12.2966));
//    ransac_l_points.push_back(Eigen::Vector3f(2.1838,   -4.0179,   -6.9858));
//    ransac_l_points.push_back(Eigen::Vector3f(0.8632,    1.0971,    3.4053));
//
//    ransac_r_points.push_back(Eigen::Vector3f(3.8301,    3.9086,    7.6601));
//    ransac_r_points.push_back(Eigen::Vector3f(2.1458,   -7.5521,  -13.0683));
//    ransac_r_points.push_back(Eigen::Vector3f(0.6632,    2.6165,    3.6896));

    int iter =0;
    Eigen::Matrix3d best_rotation;
    Eigen::Vector3d best_position;

    int max_inliers = 0;
    std::vector<bool> best_inliers;
    while(iter < 20){
      ransac_l_points.clear();
      ransac_r_points.clear();

      getRandomSeq(ransac_number, left_points.size(), ransac_vec);
//      std::cout<<"ransac_vec size="<<ransac_vec.size()<<std::endl;
      Eigen::Matrix3d rotation;
      Eigen::Vector3d position;

      for(size_t i=0; i< ransac_vec.size(); i++){
        ransac_l_points.push_back(left_points[ransac_vec[i]]);
        ransac_r_points.push_back(right_points[ransac_vec[i]]);
      }

      estimateRt(ransac_l_points, ransac_r_points, rotation, position);
      std::vector<bool> inliers;
      euc3Ddist(left_points,right_points,rotation,position,inliers,1);
      int inliers_number =0;
      for(size_t i=0; i<inliers.size(); i++){
        if(inliers[i])inliers_number++;
      }
      if(inliers_number >= max_inliers){
        max_inliers =inliers_number;
        best_inliers = inliers;
        best_rotation = rotation;
        best_position = position;
      }
      iter++;
    }

    std::cout<<"max_inliers="<<max_inliers
             <<"\nbest_rotation\n"<<best_rotation
             <<"\nbest_position\n"<<best_position.transpose()<<std::endl;
  }

  void FeatNetFrame::euc3Ddist(const std::vector<Eigen::Vector3f>& left_points,
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