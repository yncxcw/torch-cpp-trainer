// Copyright ynjassionchen@gmail.com
#include "cifar10_dataset.hpp"
#include "datasets_utils.hpp"
#include "dummy_dataset.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class Cifar10DatasetTest : public ::testing::Test {
   protected:
    Cifar10DatasetTest() {
        tmp_folder = fs::temp_directory_path() / "torch_train_test";
        fs::create_directories(tmp_folder);
        for (const auto& dataset : {"dataset1.bin", "dataset2.bin"}) {
            auto dataset_path = tmp_folder / dataset;
            _write_samples(dataset_path, 2);
        }
    }

    fs::path get_tmp_folder() { return tmp_folder; }

    // Two test bind, each with 2 samples.
    int16_t get_num_samples() { return 4; }

    ~Cifar10DatasetTest() override { fs::remove_all(tmp_folder); }

   private:
    fs::path tmp_folder;
    void _write_samples(const fs::path& dataset, const size_t& num) {
        // Write samples(32 x 32 + 1 byte).
        std::ofstream fout(dataset, std::ios::binary | std::ios::app);
        for (size_t i = 0; i < num; i++) {
            auto buffer = std::make_unique<char[]>(torch::data::datasets::kSampleSize);
            std::memset(buffer.get(), 0, torch::data::datasets::kSampleSize * sizeof(char));
            fout.write(buffer.get(), torch::data::datasets::kSampleSize);
        }

        if (fout.bad()) {
            throw std::runtime_error("Error withe writing samples for " + dataset.generic_string());
        }
    }
};

TEST(TestDummyDataset, TestLoad) {
    size_t batch_size = 8;
    auto tensor_generator = [](torch::data::TensorShape shape) -> torch::Tensor {
        return torch::zeros(shape);
    };
    auto dataset = torch::data::datasets::DummyDataset(100, {3, 10, 10}, {1}, tensor_generator);
    auto dataloader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, batch_size);

    size_t batch_count = 0;
    for (auto& batch : *dataloader) {
        batch_count++;
    }
    EXPECT_EQ(batch_count, (100 + 7) / 8);
    EXPECT_EQ(1, 0);
}

TEST_F(Cifar10DatasetTest, TestEndWith) {
    EXPECT_EQ(torch::data::ends_with("abc/def/gh.bin", ".bin"), true);
    EXPECT_EQ(torch::data::ends_with(".bin", ".bin"), true);
    EXPECT_EQ(torch::data::ends_with("abc/def/gh.bis", ".bin"), false);
    EXPECT_EQ(torch::data::ends_with(".bis", ".bin"), false);
    std::cout << "Done Cifar10DatasetTest" << std::endl;
}

TEST_F(Cifar10DatasetTest, TestLoadCifarBin) {
    auto test_folder = get_tmp_folder();
    auto [images, targets] = torch::data::datasets::load_cifar_bins(test_folder);
    std::vector<int64_t> expected_image_shape{get_num_samples(), 3,
                                              torch::data::datasets::kImageHeight,
                                              torch::data::datasets::kImageWidth};
    EXPECT_EQ(images.sizes(), expected_image_shape);
    // The test bin file is all 0
    EXPECT_EQ(images.equal(torch::zeros(expected_image_shape,
                                        torch::TensorOptions().dtype(torch::kFloat32))),
              true);
    EXPECT_EQ(images.dtype(), torch::kFloat32);

    std::vector<int64_t> expected_target_shape{get_num_samples()};
    EXPECT_EQ(targets.sizes(), expected_target_shape);
    EXPECT_EQ(targets.dtype(), torch::kInt64);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}