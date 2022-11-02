// Copyright ynjassionchen@gmail.com
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <string>

namespace torch {
namespace data {
namespace datasets {
class Cifar10Dataset : public torch::data::datasets::Dataset<Cifar10Dataset> {
   public:
    explicit Cifar10Dataset(const std::string& root);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

   private:
    torch::Tensor images_;
    torch::Tensor targets_;
};
}  // namespace datasets
}  // namespace data
}  // namespace torch