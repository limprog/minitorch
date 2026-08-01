//
// Created by limprog on 8/1/26.
//


#include "Tensor.h"

#include <algorithm>
#include <regex>

void Tensor::accumulate_grad(const Tensor &grad) const{
    if (!node_->grad.has_value()) {
        node_->grad = grad.detach();
    } else {
        node_->grad = *node_->grad + grad;
    }
}


bool Tensor::requires_grad() const {
    return node_ != nullptr;
}