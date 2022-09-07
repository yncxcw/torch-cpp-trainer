// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>

#include "module_utils.hpp"
#include "residual_block.hpp"

namespace torch {
namespace nn {

template <typename Block>
class ResNetImpl : public torch::nn::Module {
   public:
    explicit ResNetImpl(const std::array<int64_t, 3>& layers,
                        int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

   private:
    int64_t in_channels = 16;
    torch::nn::Conv2d conv{convNxN<3>(3, 16)};
    torch::nn::BatchNorm2d bn{16};
    torch::nn::ReLU relu;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::AvgPool2d avg_pool{8};
    torch::nn::Linear fc;

    torch::nn::Sequential make_layer(int64_t out_channels, int64_t blocks,
                                     int64_t stride = 1);
};

// TORCH_MODULE(RestNet) won't work becaulse ResNetImpl needs a template
// argument. Wrap class into ModuleHolder (a shared_ptr wrapper), see
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/pimpl.h
template <typename Block = ResidualBlock>
class ResNet : public torch::nn::ModuleHolder<ResNetImpl<Block>> {
   public:
    using torch::nn::ModuleHolder<ResNetImpl<Block>>::ModuleHolder;
};

}  // namespace nn
}  // namespace torch
