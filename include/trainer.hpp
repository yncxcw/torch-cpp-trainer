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

enum class Mode { TRAIN, EVAL };

template <typename TaskFactory>
class Trainer {
   public:
    using Dataset = typename TaskFactory::TaskDataset;
    using Sampler = typename TaskFactory::TaskSampler;
    using ExampleType = typename TaskFactory::ExampleType;
    using BatchType = typename TaskFactory::BatchType;

    explicit Trainer(
        torch::Device device, torch::nn::AnyModule model,
        std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>> ptr_dataloader,
        std::unique_ptr<torch::optim::Optimizer> ptr_optimizer,
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_function,
        std::function<ExampleType(std::vector<ExampleType>)> collate_function,
        torch::Mode mode = torch::Mode::TRAIN)
        : device(device),
          model(model),
          ptr_dataloader(std::move(ptr_dataloader)),
          ptr_optimizer(std::move(ptr_optimizer)),
          loss_function(loss_function),
          collate_function(collate_function),
          mode(mode) {
        ptr_model = model.ptr();
        if (mode == torch::Mode::TRAIN) {
            ptr_model->train();
        } else {
            ptr_model->eval();
        }
        ptr_model->to(device, /*non_blocking*/ true);
    }

    // Trainer is statefull, so delee copy constructor and assignment operator.
    Trainer(const Trainer&) = delete;

    Trainer& operator=(const Trainer&) = delete;

    void train(size_t epochs) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Start training for " << epochs << " epochs on device " << device.str();

        for (size_t epoch = 0; epoch < epochs; epoch++) {
            double running_loss = 0.0;
            size_t step = 0;

            for (auto& batch : *ptr_dataloader) {
                auto collated_batch = collate_function(batch);
                auto data = collated_batch.data.to(device);
                auto target = collated_batch.target.to(device);

                // Forward pass
                auto output = model.forward(data);

                // Compute loss
                torch::Tensor loss = loss_function(output, target);

                ptr_optimizer->zero_grad();
                // Backward pass
                loss.backward();
                // Update weights
                ptr_optimizer->step();

                std::cout << "Step " << step << " loss " << loss.item() << std::endl;
                step++;
            }
            // TODO Add callbacks.
            std::cout << "Epoch " << epoch << " finished." << std::endl;
        }
    }

    void inference(torch::Tensor input) {}

   private:
    torch::Device device;
    // This makes interface generic, to be safe, use move constructor.
    torch::nn::AnyModule model;
    // Underlying pointer of model
    std::shared_ptr<torch::nn::Module> ptr_model;
    // Dataloader instance
    std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>> ptr_dataloader;
    // Optimizer instance
    std::unique_ptr<torch::optim::Optimizer> ptr_optimizer;
    // loss function
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_function;
    // function to collate a vector of batches.
    std::function<ExampleType(std::vector<ExampleType>)> collate_function;
    // Mode of the trainer
    torch::Mode mode;
};

}  // namespace torch