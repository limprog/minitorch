//
// Created by limprog on 8/1/26.
//


#include "Tensor.h"

#include <algorithm>
#include <regex>
#include <unordered_set>

void Tensor::accumulate_grad(const Tensor &grad) const{
    if (node_ == nullptr) {
        throw std::runtime_error("Tensor::accumulate_grad: grad not requerd");
    }
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

bool Tensor::has_grad() const {
    if (node_ == nullptr) {
        return false;
    }
    return node_->grad.has_value();
}


void Tensor::zero_grad() {
    if (node_ == nullptr) {
        throw std::runtime_error("Tensor::zero_grad: grad not requerd");
    }
    node_->grad.reset();
}


void Tensor::build_path(const std::shared_ptr<AutogradNode> &node, std::unordered_set<AutogradNode*> &visited, std::vector<std::shared_ptr<AutogradNode>> &path) {
    if (!node || visited.contains(node.get())) {
        return;
    }

    visited.insert(node.get());

    for (const auto& parent : node->parents) {
        build_path(parent, visited, path);
    }

    path.push_back(node);
}



void Tensor::backward() const {
    if (!requires_grad()) {
        throw std::runtime_error("Tensor::backward: grad not requerd");
    }

    // temp only for scalar
    if (shape_.size() != 1 || shape_[0] != 1) {
        throw std::runtime_error("Tensor::backward: tensor must be scalar");
    }

    node_->grad = Tensor({1}, 1, false);

    std::vector<std::shared_ptr<AutogradNode>> path;
    std::unordered_set<AutogradNode*> visited;

    build_path(this->node_, visited, path);

    for (int i = path.size() - 1; i >= 0; i--) {
        const auto& node = path[i];

        if (node->grad.has_value() &&
            node->backward_fn) {
            node->backward_fn(*node->grad);
        }
    }
}


Tensor Tensor::grad() const {
    if (!requires_grad()) {
        throw std::runtime_error("Tensor::grad: grad not requerd");
    }

    if (!node_->grad.has_value()) {
        throw std::runtime_error("Tensor::grad: gradient has not been computed");
    }

    return node_->grad->copy().detach();
}

