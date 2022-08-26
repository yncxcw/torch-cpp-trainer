// Copyright ynjassionchen@gmail.com
#pragma once

#include <torch/torch.h>

namespace torch {
namespace nn {
template <int64_t N = 3>
torch::nn::Conv2d convNxN(int64_t in_channels, int64_t out_channels,
                          int64_t stride = 1) {
  return torch::nn::Conv2d(
      torch::nn::Conv2dOptions(in_channels, out_channels, N)
          .stride(stride)
          .padding(1)
          .bias(false));
}
}  // namespace nn
}  // namespace torch