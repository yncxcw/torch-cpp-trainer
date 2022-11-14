// Copyright ynjassionchen@gmail.com
#include "residual_block.hpp"
#include "resnet.hpp"

#include <gtest/gtest.h>
#include <memory>

TEST(TestResidualBlock, TestOutputShape) {
    auto block = torch::nn::ResidualBlock(3, 3, 1);
    auto output_tensor = block(torch::ones({10, 3, 20, 30}));
    auto tensor_shape = output_tensor.sizes();
    auto expect_shape = std::vector<int64_t>{10, 3, 20, 30};
    EXPECT_EQ(tensor_shape, expect_shape);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}