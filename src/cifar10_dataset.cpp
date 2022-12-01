
// Copyright ynjassionchen@gmail.com
#include "cifar10_dataset.hpp"
#include "datasets_utils.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace torch {
namespace data {
namespace datasets {

namespace fs = std::filesystem;

std::pair<torch::Tensor, torch::Tensor> load_cifar_bins(const std::string& root) {
    std::vector<char> buffer;
    // Finds all fils whose suffix is `.bin` in folder root
    const fs::path root_folder{root};
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(root_folder)) {
        const fs::path file_path = entry.path();
        const std::string file_name = file_path.filename().string();
        if (entry.is_regular_file() && ends_with(file_name, ".bin")) {
            size_t size = fs::file_size(file_path, ec);
            TORCH_CHECK(!ec, "Error when getting file size for " + file_name);
            std::cout << "reading file " << file_name << " size " << size << std::endl;
            std::ifstream fin(file_path, std::ios::binary);
            buffer.insert(buffer.end(), std::istreambuf_iterator<char>(fin), {});
        }
    }

    TORCH_CHECK(buffer.size() % kSampleSize == 0, "unexpcetd file size.");

    // Load the char array into tensor.
    size_t num_samples = buffer.size() / kSampleSize;
    // Note, torch::empty returns uninitialized data (like random data).
    auto images = torch::empty({static_cast<int64_t>(num_samples), 3, kImageHeight, kImageWidth},
                               torch::kByte);
    auto targets = torch::empty(num_samples, torch::kByte);
    for (size_t i = 0; i < num_samples; i++) {
        size_t start_index = i * kSampleSize;
        targets[i] = buffer[start_index];
        size_t image_start = start_index + 1;
        size_t image_end = image_start + kImageHeight * kImageHeight * 3;
        std::copy(buffer.begin() + image_start, buffer.begin() + image_end,
                  reinterpret_cast<char*>(images[i].data_ptr()));
    }

    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}

Cifar10Dataset::Cifar10Dataset(const std::string& root) {
    auto [first, end] = load_cifar_bins(root);
    images_ = std::move(first);
    targets_ = std::move(end);
}

torch::data::Example<> Cifar10Dataset::get(const size_t index) {
    return torch::data::Example<>{images_[index], targets_[index]};
}

torch::optional<size_t> Cifar10Dataset::size() const { return images_.size(0); }

}  // namespace datasets
}  // namespace data
}  // namespace torch