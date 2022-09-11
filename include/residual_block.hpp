// Copyright 2020-present pytorch-cpp Authors
// Copyright ynjassionchen@gmail.com
#pragma once

#include <torch/torch.h>

#include "module_utils.hpp"

namespace torch {
namespace nn {

class ResidualBlockImpl : public torch::nn::ModelImpl {
   public:
    ResidualBlockImpl(int64_t in_channels, int64_t out_channels,
                      int64_t stride = 1,
                      torch::nn::Sequential downsample = nullptr);

    torch::Tensor forward(torch::Tensor x);

   private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Sequential downsampler;
};

TORCH_MODULE(ResidualBlock);

}  // namespace nn
}  // namespace torch