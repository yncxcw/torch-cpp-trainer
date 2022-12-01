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

inline void check_tensor_shape(const TensorShape& tensor_shape) {
    std::for_each(tensor_shape.begin(), tensor_shape.end(), [](const int64_t& dim) {
        if (dim < 0) {
            throw std::runtime_error("Dim of tensor_shaoe must >=0");
        }
    });
}

inline bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

}  // namespace data
}  // namespace torch