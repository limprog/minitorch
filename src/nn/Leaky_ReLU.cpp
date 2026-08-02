#include <nn/Module.h>
#include <nn/Leaky_ReLU.h>
#include <Tensor.h>


Leaky_ReLU::Leaky_ReLU(float alpha): alpha_(alpha), Module() {}


Tensor Leaky_ReLU::forward(const Tensor& input) const{
    return input.leaky_relu(alpha_);
}
