// Copyright ynjassionchen@gmail.com
#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "cifar10_dataset.hpp"
#include "resnet.hpp"
#include "task_factory.hpp"

using TaskFactory = torch::train::TaskFactory<torch::data::datasets::Cifar10Dataset,
                                              torch::data::samplers::RandomSampler>;

using ResNet = torch::nn::ResNet<>;

class Cifar10Factory : public TaskFactory {
   public:
    Cifar10Factory(const std::string& name, const size_t batch_size, const std::string& results_dir,
                   const size_t num_workers, const std::string dataset_folder,
                   const double learning_rate)
        : TaskFactory(name, batch_size, num_workers, results_dir, learning_rate),
          m_dataset_folder(dataset_folder) {}

    torch::data::datasets::Cifar10Dataset make_dataset() override {
        return torch::data::datasets::Cifar10Dataset(m_dataset_folder);
    }

    torch::nn::AnyModule make_model() {
        std::array<int64_t, 3> layers{2, 2, 2};
        return torch::nn::AnyModule(ResNet(layers, m_num_classes));
    }

    std::unique_ptr<torch::optim::Optimizer> make_optimizer(torch::nn::AnyModule model) override {
        // The ptr() returns a shared_ptr of torch::nn::Module, but why ?
        return std::make_unique<torch::optim::Adam>(model.ptr()->parameters(),
                                                    torch::optim::AdamOptions(learning_rate()));
    }

    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> make_loss_function() override {
        auto loss_function = [](torch::Tensor input, torch::Tensor target) -> torch::Tensor {
            return torch::nn::functional::cross_entropy(input, target);
        };

        return loss_function;
    }

    std::function<ExampleType(std::vector<ExampleType>)> make_collate_function() override {
        auto collate_function = [](std::vector<ExampleType> batch) -> ExampleType {
            std::vector<torch::Tensor> datas;
            std::vector<torch::Tensor> targets;
            for (const auto& sample : batch) {
                datas.push_back(sample.data);
                targets.push_back(sample.target);
            }
            return ExampleType{torch::stack(std::move(datas)), torch::stack(std::move(targets))};
        };
        return collate_function;
    }

   private:
    std::string m_dataset_folder;
    int64_t m_num_classes{10};
};