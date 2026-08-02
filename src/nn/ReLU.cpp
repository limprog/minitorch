#include <nn/Module.h>
#include <Tensor.h>
#include <nn/ReLU.h>



Tensor ReLU::forward(const Tensor &input) const{
    return input.relu();
}


