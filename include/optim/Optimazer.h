#pragma once

#include <Tensor.h>

class Optimizer {
protected:
    std::vector<Tensor*> parameters_;

public:
    Optimizer() = delete;
    Optimizer(std::vector<Tensor*> parameters);

    void zero_grad();
};
