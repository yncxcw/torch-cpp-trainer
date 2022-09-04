// Copyright ynjassionchen@gmail.com
#include "../include/module_utils.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(TestCovNxN, TestOutputShape) {
  auto conv3x3 = torch::nn::convNxN<3>(3, 20, 1);
  auto output_tensor = conv3x3(torch::ones({10, 3, 10, 10}));
  auto tensor_shape = output_tensor.sizes();
  auto expect_shape = std::vector<int64_t>{10, 20, 10, 10};
  EXPECT_EQ(tensor_shape, expect_shape);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}