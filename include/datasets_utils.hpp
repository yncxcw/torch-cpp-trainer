// Copyright ynjassionchen@gmail.com
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace torch {
namespace data {

using TensorShape = std::vector<int64_t>;

void _check_tensor_shape(const TensorShape& tensor_shape) {
    std::for_each(
        tensor_shape.begin(), tensor_shape.end(), [](const int64_t& dim) {
            if (dim < 0) {
                throw std::runtime_error("Dim of tensor_shaoe must >=0");
            }
        });
}
}  // namespace data
}  // namespace torch