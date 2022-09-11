// Copyright ynjassionchen@gmail.com
#pragma once

#include <functional>
#include <memory>

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "module_utils.hpp"

namespace torch {

template <typename Dataset,
          typename Sampler = torch::data::samplers::RandomSampler>
class Trainer {
   public:
    explicit Trainer(
        torch::Device device, torch::nn::AnyModule model,
        std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>>
            ptr_dataloader,
        std::unique_ptr<torch::optim::Optimizer> ptr_optimizer,
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)>
            loss_function);

    // Trainer is statefull, so delee copy constructor and assignment operator.
    Trainer(const Trainer&) = delete;

    Trainer& operator=(const Trainer&) = delete;

    void train(size_t epochs);

   private:
    torch::Device device;
    // This makes interface generic, to be safe, use move constructor.
    torch::nn::AnyModule model;
    // Dataloader instance
    std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>>
        ptr_dataloader;
    // Optimizer instance
    std::unique_ptr<torch::optim::Optimizer> ptr_optimizer;
    // loss function
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_function;
};

}  // namespace torch