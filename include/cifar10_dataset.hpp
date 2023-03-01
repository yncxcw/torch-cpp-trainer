// Copyright ynjassionchen@gmail.com
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace datasets {

const constexpr int64_t kImageHeight = 32;
const constexpr int64_t kImageWidth = 32;
const constexpr int64_t kSampleSize = 3 * kImageHeight * kImageWidth + 1;

std::pair<torch::Tensor, torch::Tensor> load_cifar_bins(const std::string& root);

class Cifar10Dataset : public torch::data::datasets::Dataset<Cifar10Dataset> {
   public:
    using ExampleType = typename Dataset::ExampleType;
    using BatchType = typename std::vector<Dataset::ExampleType>;

    explicit Cifar10Dataset(const std::string& root);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    static const constexpr bool is_stateful = false;

   private:
    torch::Tensor images_;
    torch::Tensor targets_;
};
}  // namespace datasets
}  // namespace data
}  // namespace torch