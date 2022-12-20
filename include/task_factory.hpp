// Copyright ynjassionchen@gmail.com
#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/torch.h>

namespace torch {
namespace train {

template <typename Dataset, typename Sampler>
class TaskFactory {
   public:
    using ExampleType = typename Dataset::ExampleType;

    explicit TaskFactory(const std::string& name, const size_t batch_size, const size_t num_workers,
                         const std::string& results_dir, const double learning_rate)
        : m_name(name),
          m_batch_size(batch_size),
          m_num_workers(num_workers),
          m_results_dir(results_dir),
          m_learning_rate(learning_rate) {}

    ~TaskFactory() {}

    // It doesn't make sense to build a new factory from another.
    TaskFactory(const TaskFactory& other) = delete;

    std::string name() const { return m_name; }

    std::string results_dir() const { return m_results_dir; }

    size_t batch_size() const { return m_batch_size; }

    size_t num_workers() const { return m_num_workers; }

    double learning_rate() const { return m_learning_rate; }

    // interface
    virtual torch::nn::AnyModule make_model() = 0;

    std::unique_ptr<torch::data::StatelessDataLoader<Dataset, Sampler>> make_dataloader() {
        auto dataset = make_dataset();
        auto dataset_size = dataset.size();
        TORCH_CHECK(dataset_size.has_value(), "dataset must contains samples")
        auto sampler = make_sampler(*dataset_size);
        return torch::data::make_data_loader<Dataset, Sampler>(dataset, sampler, m_batch_size);
    };

    Sampler make_sampler(size_t size) { return Sampler(size); }

    virtual Dataset make_dataset() = 0;

    virtual std::function<ExampleType(std::vector<ExampleType>)> make_collate_function() = 0;

    virtual std::unique_ptr<torch::optim::Optimizer> make_optimizer(torch::nn::AnyModule model) = 0;

    virtual std::function<torch::Tensor(torch::Tensor, torch::Tensor)> make_loss_function() = 0;

   private:
    std::string m_name;
    std::string m_results_dir;
    size_t m_batch_size;
    size_t m_num_workers;
    double m_learning_rate;
};  // namespace train
}  // namespace train

}  // namespace torch