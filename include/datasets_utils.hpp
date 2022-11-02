// Copyright ynjassionchen@gmail.com
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace torch {
namespace data {

using TensorShape = at::IntArrayRef;

void _check_tensor_shape(const TensorShape& tensor_shape) {
    std::for_each(tensor_shape.begin(), tensor_shape.end(), [](const int64_t& dim) {
        if (dim < 0) {
            throw std::runtime_error("Dim of tensor_shaoe must >=0");
        }
    });
}
}  // namespace data
}  // namespace torch