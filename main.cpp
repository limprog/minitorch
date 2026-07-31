#include <iostream>

#include "Tensor.h"


int main() {
    Tensor tensor({2, 2});
    tensor({1, 1}) = 1.0f;

    std::cout << tensor({1, 1}) << std::endl;


}