
// Copyright ynjassionchen@gmail.com
#include "cifar10_dataset.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace torch {
namespace data {
namespace datasets {

namespace fs = std::filesystem;

const constexpr int64_t kImageHeight = 32;
const constexpr int64_t kImageWidth = 32;
const constexpr int64_t kSampleSie = kImageHeight * kImageWidth + 1;

bool _ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

std::pair<torch::Tensor, torch::Tensor> _load_cifar_bins(const std::string& root) {
    std::vector<char> buffer;
    // Finds all fils whose suffix is `.bin` in folder root
    const fs::path root_folder{root};
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(root_folder)) {
        const fs::path file_path = entry.path();
        const std::string file_name = file_path.filename().string();
        if (entry.is_regular_file() && _ends_with(file_name, ".bin")) {
            size_t size = fs::file_size(file_path, ec);
            if (ec) {
                throw std::runtime_error("Error when getting file size for " + file_name);
                std::ifstream fin(file_path, std::ios::binary);
                buffer.insert(buffer.end(), std::istreambuf_iterator<char>(fin), {});
            }
        }
    }

    TORCH_CHECK(buffer.size() % 3073 != 0, "unexpcetd file size.");

    // Load the char array into tensor.
    size_t num_samples = buffer.size() / 3073;
    auto images = torch::empty({static_cast<int64_t>(num_samples), 3, kImageHeight, kImageWidth},
                               torch::kByte);
    auto targets = torch::empty(num_samples, torch::kByte);
    for (size_t i = 0; i < num_samples; i++) {
        size_t start_index = i * kSampleSie;
        targets[i] = buffer[start_index];
        size_t image_start = start_index + 1;
        size_t image_end = image_start + kImageHeight * kImageHeight * 3;
        std::copy(buffer.begin() + image_start, buffer.begin() + image_end,
                  reinterpret_cast<char*>(images[i].data_ptr()));
    }

    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}

Cifar10Dataset::Cifar10Dataset(const std::string& root) {
    auto [first, end] = _load_cifar_bins(root);
    images_ = std::move(first);
    targets_ = std::move(end);
}

torch::data::Example<> Cifar10Dataset::get(size_t index) {
    return torch::data::Example<>{images_[index], targets_[index]};
}

torch::optional<size_t> Cifar10Dataset::size() const { return images_.size(0); }

}  // namespace datasets
}  // namespace data
}  // namespace torch