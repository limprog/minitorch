#pragma once

#include <nn/ReLU.h>
#include <nn/Linear.h>
#include <nn/Module.h>
#include <Tensor.h>


class SimpleModel : public Module {
private:
    Linear linear1_;
    Linear linear2_;
    ReLU relu_;
public:
    SimpleModel(int in_feachers, int out_feachers = 1);

    Tensor forward(const Tensor& input) const;

};