// Copyright ynjassionchen@gmail.com
#include "residual_block.hpp"
#include "resnet.hpp"

#include <gtest/gtest.h>
#include <array>
#include <map>
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
    std::vector<torch::Tensor> model_weights(model.ptr()->parameters());
    auto output_tensor = model.forward(torch::ones({10, 3, 32, 32}));
    auto tensor_shape = output_tensor.sizes();
    auto expected_shape = std::vector<int64_t>{10, 10};
    EXPECT_EQ(tensor_shape, expected_shape);
}

TEST(TestResnet, TestCheckPoint) {
    auto model_to_save = torch::nn::AnyModule(ResNet(std::array<int64_t, 3>({2,2,2}), 10));
    std::shared_ptr<torch::nn::Module> ptr_model_save = model_to_save.ptr(); 
    torch::save(ptr_model_save, "/tmp/model.pt");

    auto named_parameter_save = ptr_model_save->named_parameters(); 
    auto model_to_load = torch::nn::AnyModule(ResNet(std::array<int64_t, 3>({2,2,2}), 10));
    std::shared_ptr<torch::nn::Module> ptr_model_load= model_to_load.ptr(); 
    torch::load(ptr_model_load, std::string{"/tmp/model.pt"});

    auto named_parameter_load = ptr_model_load->named_parameters();
    for(auto iter=named_parameter_load.begin(); iter != named_parameter_load.end(); iter++) {
        EXPECT_TRUE(named_parameter_save.find(iter->key()) != nullptr);
        EXPECT_TRUE(torch::equal(named_parameter_save[iter->key()], iter->value()));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}