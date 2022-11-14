// Copyright ynjassionchen@gmail.com
#include "module_utils.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(TestCov3x3, TestOutputShape) {
    auto conv3x3 = torch::nn::convNxN<3>(3, 20, 1);
    auto output_tensor = conv3x3(torch::ones({10, 3, 10, 10}));
    auto tensor_shape = output_tensor.sizes();
    auto expect_shape = std::vector<int64_t>{10, 20, 10, 10};
    EXPECT_EQ(tensor_shape, expect_shape);
}

TEST(TestCov1x1, TestOutputShape) {
    auto conv3x3 = torch::nn::convNxN<1>(3, 20, 1);
    auto output_tensor = conv3x3(torch::ones({10, 3, 10, 10}));
    auto tensor_shape = output_tensor.sizes();
    auto expect_shape = std::vector<int64_t>{10, 20, 12, 12};
    EXPECT_EQ(tensor_shape, expect_shape);
}

TEST(TestCov2x2, TestOutputShape) {
    auto conv3x3 = torch::nn::convNxN<2>(3, 20, 1);
    auto output_tensor = conv3x3(torch::ones({10, 3, 10, 10}));
    auto tensor_shape = output_tensor.sizes();
    auto expect_shape = std::vector<int64_t>{10, 20, 11, 11};
    EXPECT_EQ(tensor_shape, expect_shape);
}

TEST(TestAnyModel, TestForward) {
    auto model_maker = []() -> torch::nn::AnyModule {
        return torch::nn::AnyModule(torch::nn::convNxN<2>(3, 20, 1));
    };

    auto model = model_maker();
    auto output_tensor = model.forward(torch::ones({10, 3, 10, 10}));
    auto tensor_shape = output_tensor.sizes();
    auto expect_shape = std::vector<int64_t>{10, 20, 11, 11};
    EXPECT_EQ(tensor_shape, expect_shape);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}