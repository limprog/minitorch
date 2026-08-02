#pragma once


#include <nn/Module.h>
#include <Tensor.h>


class Leaky_ReLU : public Module {
private:
    float alpha_;
public:
    Leaky_ReLU(float alpha = 0.1f);


    Tensor forward(const Tensor& input) const;
};