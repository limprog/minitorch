#include <Tensor.h>
#include <nn/SimpleModel.h>
#include <nn/Module.h>
#include <nn/ReLU.h>
#include <nn/Linear.h>

SimpleModel::SimpleModel(int in_feachers, int out_feachers)
  : Module(),
    linear1_(in_feachers, 64),
    linear2_(64, out_feachers),
    relu_() {
    add_child(linear1_);
    add_child(linear2_);
    add_child(relu_);
}


Tensor SimpleModel::forward(const Tensor &input) const{
    Tensor x = linear1_(input);
    x = relu_(x);
    return linear2_(x);
}