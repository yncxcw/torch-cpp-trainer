#include <torch/torch.h>
#include <iostream>

#include "residual_block.hpp"

int main() {
    torch::nn::ResidualBlock model(10, 20, 3);
    std::cout << "hello world" << std::endl;
    return 0;
}
