// Copyright 2020-present pytorch-cpp Authors
// Copyright ynjassionchen@gmail.com

#include <torch/torch.h>

#include "module_utils.hpp"
#include "residual_block.hpp"

namespace torch {
namespace nn {
ResidualBlockImpl::ResidualBlockImpl(int64_t in_channels, int64_t out_channels,
                                     int64_t stride,
                                     torch::nn::Sequential downsample)
    : conv1(convNxN<3>(in_channels, out_channels, stride)),
      bn1(out_channels),
      conv2(convNxN<3>(out_channels, out_channels)),
      bn2(out_channels),
      downsampler(downsample) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("relu", relu);
    register_module("conv2", conv2);
    register_module("bn2", bn2);

    if (downsampler) {
        register_module("downsampler", downsampler);
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = relu->forward(out);
    out = conv2->forward(out);
    out = bn2->forward(out);

    auto residual = downsampler ? downsampler->forward(x) : x;
    out += residual;
    out = relu->forward(out);

    return out;
}

}  // namespace nn
}  // namespace torch