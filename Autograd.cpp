//
// Created by limprog on 8/1/26.
//


#include "Tensor.h"

#include <algorithm>
#include <regex>

void Tensor::accumulate_grad(const Tensor &grad) const{
    if (node_ == nullptr) {
        throw std::runtime_error("Tensor::accumulate_grad: grad not requerd");
    }
    1
    if (!node_->grad.has_value()) {
        node_->grad = grad.detach();
    } else {
        node_->grad = (*node_->grad + grad).detach();
    }
}

void Tensor::accumulate_grad(const std::shared_ptr<AutogradNode> &node, const Tensor &grad)  {
    if (!node) {
        throw std::runtime_error("Tensor::accumulate_grad: grad not requerd");
    }

    if (!node->grad.has_value()) {
        node->grad = grad.detach();
    } else {
        node->grad = (*node->grad + grad).detach();
    }
}


bool Tensor::requires_grad() const {
    return node_ != nullptr;
}
