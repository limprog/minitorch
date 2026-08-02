#pragma once



#include <nn/Module.h>
#include <Tensor.h>


class ReLU : public Module {
public

    Tensor forward(const Tensor& input) const;
}