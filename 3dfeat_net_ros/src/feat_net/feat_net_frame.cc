//
// Created by zzz on 18-12-26.
//

#include "feat_net/feat_net_frame.h"
namespace FeatNet{

  FeatNetFrame::FeatNetFrame(){
    ransac_matcher_ptr_ = std::make_shared<RansacMatch>();
    visual_ptr_ = std::make_shared<Visualize>();
    pose_graph_ptr_ = std::make_shared<PoseGraph>();
  }

  bool FeatNetFrame::LoadDataFromFile(std::string file_path){

    std::string  points_file_path = file_path + "/use_data";
    std::string  transform_file_path = file_path + "/pose.txt";
    std::string  desc_file_path = file_path + "/use_data/results";

    namespace bf = boost::filesystem;
    {
      //get all points from file
      std::vector<bf::path> file_vec;
      if(bf::is_directory(points_file_path)){
        std::copy(bf::directory_iterator(points_file_path), bf::directory_iterator(), std::back_inserter(file_vec));
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
      Eigen::Quaterniond rotation;
      Eigen::Vector3d position;
      pose_data_file.open(transform_file_path.c_str(), std::ios::in);
      if(pose_data_file.is_open()){
        std::string str;
        int frame_index=0;
        while(std::getline(pose_data_file, str, '\n')){
          std::vector<std::string> str_vec;
          boost::split(str_vec,str,boost::is_any_of(","));

          rotation = Eigen::Quaterniond(boost::lexical_cast<double>(str_vec[3]),
                                        boost::lexical_cast<double>(str_vec[0]),
                                        boost::lexical_cast<double>(str_vec[1]),
                                        boost::lexical_cast<double>(str_vec[2]));
          position = Eigen::Vector3d(boost::lexical_cast<double>(str_vec[4]),
                                     boost::lexical_cast<double>(str_vec[5]),
                                     boost::lexical_cast<double>(str_vec[6]));
          features_frames_[frame_index].rotation = rotation;
          features_frames_[frame_index].position = position;
          frame_index++;
        }
        pose_data_file.close();
      }else{
        std::cerr<<"pose_data_file="<<transform_file_path<<" is not opened"<<std::endl;
        exit(-1);
      }
    }

    {
      //get descriptors and features
      std::vector<bf::path> file_vec;
      file_vec.clear();
      if(bf::is_directory(desc_file_path)){
        std::copy(bf::directory_iterator(desc_file_path), bf::directory_iterator(), std::back_inserter(file_vec));
        std::sort(file_vec.begin(), file_vec.end());
      }

      for(size_t frame_index=0; frame_index < file_vec.size(); frame_index++) {
        if (file_vec[frame_index].string().compare(file_vec[frame_index].string().size() - 4, file_vec[frame_index].string().size(),
                                             ".bin")) {
          //if it equals .bin, and will return zero
          continue;
        }
        std::ifstream data_file;
        data_file.open(file_vec[frame_index].string().c_str(), std::ios::in | std::ios::binary);
        Eigen::Matrix<float, FEATURE_DIM+3, 1> point_data;

        while(data_file.read((char*)point_data.data(), (FEATURE_DIM+3) * sizeof(float))){
          features_frames_[frame_index].features.push_back(point_data.block<3,1>(0,0));
          features_frames_[frame_index].descriptors.push_back(point_data.block<FEATURE_DIM,1>(3,0));
        }

//        std::cout<<"frame index="<<index<<" feature size="<<features_frames_[index].features.size()<<std::endl;
      }
    }
  };

  void FeatNetFrame::feature_tracking(){
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    int looped=0;
    int max_looped = 3;
    for(size_t i=0; i< features_frames_.size(); i++){
      if(i==0){
        pre_frame_ptr_ = &features_frames_[i];
        cur_frame_ptr_ = &features_frames_[i];
        cur_rotation_ = cur_frame_ptr_->rotation;
        cur_position_ = cur_frame_ptr_->position;

        Pose pose_node;
        pose_node.position = cur_position_;
        pose_node.rotation = cur_rotation_;
        pose_node.features = cur_frame_ptr_->features;
        pose_node.descriptors = cur_frame_ptr_->descriptors;
        pose_node.id = pose_graph_ptr_->getSizeOfTrajectory();
        pose_node.points = cur_frame_ptr_->points;
        pose_graph_ptr_->addPose(pose_node);

        continue;
      }
      printf("\n\n\n%04d.bin %04d.bin",(int)i-1,(int)i);
      cur_frame_ptr_ = &features_frames_[i];
      std::vector<int> match12;
      std::vector<float> error12;

      match(pre_frame_ptr_->descriptors,
            cur_frame_ptr_->descriptors,
            match12,
            error12);

      std::vector<Eigen::Vector3f> selected_features;
      for(size_t j=0; j < match12.size(); j++){
        selected_features.push_back(cur_frame_ptr_->features[match12[j]]);
//        std::cout<<"i="<<i<<" j="<<match12[i]<<std::endl;
      }
      Eigen::Matrix4d delta;
      ransac_matcher_ptr_->ransac(pre_frame_ptr_->features, selected_features,1,delta);
      pose = pose * delta;
      cur_rotation_ = pose.block<3,3>(0,0);
      cur_position_ = pose.block<3,1>(0,3);
      std::cout<<"current pose\n"<<pose<<std::endl;
      pre_frame_ptr_ = cur_frame_ptr_;

//      cur_rotation_ = cur_frame_ptr_->rotation;
//      cur_position_ = cur_frame_ptr_->position;
      Pose pose_node;
      pose_node.position = cur_position_;
      pose_node.rotation = cur_rotation_;
      pose_node.features = cur_frame_ptr_->features;
      pose_node.descriptors = cur_frame_ptr_->descriptors;
      pose_node.id = pose_graph_ptr_->getSizeOfTrajectory();
      pose_node.points = cur_frame_ptr_->points;

      //detect loop closure
      pose_graph_ptr_->addPose(pose_node);

      pose_graph_ptr_->addConstraint(i-1,i,delta);

      Eigen::Matrix4d loop_delta;
      if(looped < max_looped && pose_graph_ptr_->detectLoop(0,pose_graph_ptr_->getSizeOfTrajectory() -1, loop_delta)){
        GS_WARN("loop detection suceessful index_i=%d, index_j=%d",0,pose_graph_ptr_->getSizeOfTrajectory()-1);
        GS_INFO("loop detal position=%lf %lf %lf",loop_delta(0,3),loop_delta(1,3),loop_delta(2,3));
        pose_graph_ptr_->addConstraint(0, pose_graph_ptr_->getSizeOfTrajectory()-1, loop_delta);
        pose_graph_ptr_->optimize3DPoseGraph(0, pose_graph_ptr_->getSizeOfTrajectory()-1, loop_delta);
//        pose_graph_ptr_->optimize4DoFPoseGraph(0, pose_graph_ptr_->getSizeOfTrajectory() -1, loop_delta);
        visual_ptr_->addLoopPath(pose_graph_ptr_->getTracjectory(),"world");
        looped++;
      }
      publish();

      if(looped >= max_looped){
        exit(0);
      }
//      usleep(1000);
    }
    exit(0);
  };

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
      for(size_t j=0; j < desc_right_vec.size(); j++){
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

  void FeatNetFrame::publish(){
    std::vector<Eigen::Vector3f> points_in_world;
    for(size_t i=0; i< cur_frame_ptr_->points.size(); i++){
      points_in_world.push_back(cur_rotation_.cast<float>()* cur_frame_ptr_->points[i] + cur_position_.cast<float>());
    }
    visual_ptr_->addFramePose(cur_rotation_,cur_position_,"world");
    visual_ptr_->addFrameCloud(points_in_world, "world");
    visual_ptr_->publish();
  }
}