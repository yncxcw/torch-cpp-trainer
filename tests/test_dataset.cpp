// Copyright ynjassionchen@gmail.com
#include "../include/datasets_utils.hpp"
#include "../include/dummy_dataset.hpp"

#include <gtest/gtest.h>

TEST(TestDummyDataset, TestLoad) {
    size_t batch_size = 8;
    auto tensor_generator =
        [](torch::data::TensorShape shape) -> torch::Tensor {
        return torch::zeros(shape);
    };
    auto dataset = torch::data::datasets::DummyDataset(100, {3, 10, 10}, {1},
                                                       tensor_generator);
    auto dataloader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            dataset, batch_size);

    size_t batch_count = 0;
    for (auto& batch : *dataloader) {
        batch_count++;
    }
    EXPECT_EQ(batch_count, 100 / 8);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}