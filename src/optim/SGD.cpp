#include <Tensor.h>
#include <optim/SGD.h>
#include <optim/Optimazer.h>


SGD::SGD(std::vector<Tensor*> parameters, float learning_rate)
    : Optimizer(parameters), learning_rate_(learning_rate) {}


void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor* par = parameters_[i];

        if (par == nullptr) {
            continue;
        }

        if (!par->has_grad()) {
            continue;
        }

        par->sub_(par->grad(), learning_rate_);
    }

}
