#include <iostream>

#include "Tensor.h"


int main() {
    Tensor tensor({2, 2}) ;
    tensor({1, 1}) = 1.0f;
    tensor({0, 0}) = 2.0f;

    tensor.reshape_({2, 2});
    tensor.unsqueeze_(2);

    std::cout << tensor << std::endl;
}