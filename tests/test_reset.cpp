// Copyright ynjassionchen@gmail.com
#include "residual_block.hpp"
#include "resnet.hpp"

#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <vector>

using ResNet = torch::nn::ResNet<>;

TEST(TestResidualBlock, TestOutputShape) {
    auto block = torch::nn::ResidualBlock(3, 3, 1);
    auto output_tensor = block(torch::ones({10, 3, 20, 30}));
    auto tensor_shape = output_tensor.sizes();
    auto expect_shape = std::vector<int64_t>{10, 3, 20, 30};
    EXPECT_EQ(tensor_shape, expect_shape);
}

TEST(TestResnet, TestForward) {
    auto model_maker = []() -> torch::nn::AnyModule {
        return torch::nn::AnyModule(ResNet(std::array<int64_t, 3>({2, 2, 2}), 10));
    };

    auto model = model_maker();
    auto output_tensor = model.forward(torch::ones({10, 3, 32, 32}));
    auto tensor_shape = output_tensor.sizes();
    auto expected_shape = std::vector<int64_t>{10, 10};
    EXPECT_EQ(tensor_shape, expected_shape);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}