#include <Tensor.h>
#include <optim/Optimazer.h>

Optimizer::Optimizer(std::vector<Tensor*> parameters)
   : parameters_(parameters) {}


void Optimizer::zero_grad() {
   for (std::size_t i = 0; i < parameters_.size(); ++i) {
      parameters_[i]->zero_grad();
   }
}

