// Copyright ynjassionchen@gmail.com
#pragma once
#include <string>

namespace torch {
namespace train {

class TaskFactory {
   public:
    explicit TaskFactory(const std::string& name, const size_t batch_size,
                         const size_t num_workers,
                         const std::string& results_dir)
        : m_name(name),
          m_batch_size(batch_size),
          m_num_workers(num_workers),
          m_results_dir(results_dir) {}

    ~TaskFactory() {}

    // It doesn't make sense to build a new factory from other.
    TaskFactory(const TaskFactory& other) = delete;

    std::string name() const { return m_name; }

    std::string results_dir() const { return m_results_dir; }

    size_t batch_size() const { return m_batch_size; }

    size_t num_workers() const { return m_num_workers; }

   private:
    std::string m_name;
    std::string m_results_dir;
    size_t m_batch_size;
    size_t m_num_workers;
};
}  // namespace train

}  // namespace torch