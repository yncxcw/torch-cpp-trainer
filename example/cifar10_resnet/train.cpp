#include "cifar10_resnet_factory.hpp"
#include "trainer.hpp"

#include <iostream>
#include <memory>

// using Trainer =
//     torch::Trainer<torch::data::datasets::Cifar10Dataset, torch::data::samplers::RandomSampler>;

int main() {
    auto factory_ptr = std::make_unique<Cifar10Factory>("cifar10", 10, "/tmp", 5, "/tmp", 0.1);
    return 0;
}