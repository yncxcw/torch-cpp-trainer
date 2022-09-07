// Copyright ynjassionchen@gmail.com
#include "../include/dummy_dataset.hpp"

#include <gtest/gtest.h>

TEST(TestDummyDataset, TestLoad) {
    auto dataset = torch::data::datasets::DummyDataset(100, {3, 10, 10}, {1});
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}