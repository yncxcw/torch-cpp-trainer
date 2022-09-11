// Copyright ynjassionchen@gmail.com
#pragma once

#include "datasets_utils.hpp"

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <vector>

namespace torch {
namespace data {
namespace datasets {

using namespace torch::data;

class DummyDataset : public torch::data::datasets::Dataset<DummyDataset> {
   public:
    explicit DummyDataset(const size_t length, const TensorShape feature_shape,
                          const TensorShape label_shape)
        : length(length),
          feature_shape(feature_shape),
          label_shape(label_shape) {
        _check_tensor_shape(feature_shape);
        _check_tensor_shape(label_shape);
    }

    torch::optional<size_t> size() const override { return length; }

    torch::data::Example<> get(size_t index) override {
        return torch::data::Example<>{torch::zeros(feature_shape),
                                      torch::zeros(label_shape)};
    }

   private:
    size_t length;
    TensorShape feature_shape;
    TensorShape label_shape;
};

}  // namespace datasets
}  // namespace data
}  // namespace torch