// Copyright ynjassionchen@gmail.com

#include "trainer.hpp"
#include "dummy_dataset.hpp"
#include "module_utils.hpp"

namespace torch {

template <typename Dataset, typename Sampler>
Trainer<Dataset, Sampler>::Trainer(
    torch::Device device, torch::nn::AnyModule model,
    std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>> ptr_dataloader,
    std::unique_ptr<torch::optim::Optimizer> ptr_optimizer,
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_function)
    : device(device),
      model(model),
      ptr_dataloader(std::move(ptr_dataloader)),
      ptr_optimizer(std::move(ptr_optimizer)),
      loss_function(loss_function) {}

template <typename Dataset, typename Sampler>
void Trainer<Dataset, Sampler>::train(size_t epochs) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Start training for " << epochs << " epochs on device " << device.str();

    // Get the underlying ptr pointig to the torch::nn::Module
    std::shared_ptr<torch::nn::Module> ptr_model = model.ptr();
    ptr_model->train();
    ptr_model->to(device, /*non_blocking*/ true);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double running_loss = 0.0;
        size_t step = 0;

        for (auto& batch : *ptr_dataloader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model.forward(data);

            // Compute loss
            auto loss = loss_function(output, target);

            ptr_optimizer->zero_grad();
            // Backward pass
            loss.backward();
            // Update weights
            ptr_optimizer->step();

            std::cout << "Step " << step << " loss " << loss << std::endl;
            step++;
        }
        // TODO Add callbacks.
        std::cout << "Epoch " << epoch << " finished." << std::endl;
    }
}

// This is a bit annoying, should we only keep trainer.hpp.
// template class Trainer<torch::data::datasets::Cifar10Dataset,
// torch::data::samplers::RandomSampler>; template class
// Trainer<torch::data::datasets::DummyDataset, torch::data::samplers::RandomSampler>;
}  // namespace torch