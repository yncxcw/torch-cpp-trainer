#include "cifar10_resnet_factory.hpp"
#include "trainer.hpp"

#include <iostream>
#include <memory>

using Trainer =
    torch::Trainer<torch::data::datasets::Cifar10Dataset, torch::data::samplers::RandomSampler>;

int main() {
    auto factory_ptr = std::make_unique<Cifar10Factory>(
        "cifar10", 10, "/tmp/cifar10/log", 5, "/tmp/cifar10/cifar-10-batches-bin/train", 0.1);
    auto model = factory_ptr->make_model();

    Trainer trainer(
        /* device           */
        torch::kCUDA,
        /* model            */
        factory_ptr->make_model(),
        /* dataloader       */
        std::move(factory_ptr->make_dataloader()),
        /* optimizer        */
        std::move(factory_ptr->make_optimizer(model)),
        /* loss_function    */
        factory_ptr->make_loss_function(),
        /* collate_function */
        factory_ptr->make_collate_function());

    trainer.train(10);

    return 0;
}