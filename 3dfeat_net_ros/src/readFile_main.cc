//
// Created by zzz on 18-12-26.
//

#include "FeatNetFrame.h"

int main(int argc, char **argv) {

  FeatNet::FeatNetFrame feat_net_frames;
  feat_net_frames.LoadDataFromFile();
  feat_net_frames.feature_tracking();
}