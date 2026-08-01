//
// Created by limprog on 8/1/26.
//

#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <regex>


Tensor Tensor::operator+(const Tensor& other) const{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+: shape should be the same");
    }

    if (!is_contiguous() || !other.is_contiguous()) {
        throw std::invalid_argument("Tensor::operator+: tensor should be contiguous");
    }

    Tensor result(shape_);

    for (std::size_t i = 0; i < storage_->data_.size(); i++) {
        result.storage_->data_[i] = storage_->data_[i] + other.storage_->data_[i];
    }

    if (requires_grad() || other.requires_grad()) {
        const auto A_node = this->node_;
        const auto B_node = other.node_;

        result.node_ = std::make_shared<AutogradNode>();

        if (A_node) {
            result.node_->parents.push_back(A_node);
        }

        if (B_node) {
            result.node_->parents.push_back(B_node);
        }

        result.node_->backward_fn =
            [A_node, B_node](const Tensor& grad) {
                if (A_node) accumulate_grad(A_node, grad);
                if (B_node) accumulate_grad(B_node, grad);
            };
    } else {
        result.node_.reset();
    }

    return result;
}


Tensor Tensor::operator*(const Tensor& other) const{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator*: shape should be the same");
    }
    if (!is_contiguous() || !other.is_contiguous()) {
        throw std::invalid_argument("Tensor::operator*: tensor should be contiguous");
    }

    Tensor result(other.shape_);

    for (std::size_t i = 0; i < storage_->data_.size(); i++) {
        result.storage_->data_[i] = storage_->data_[i] * other.storage_->data_[i];
    }

    if (requires_grad() || other.requires_grad()) {
        const auto A_node = this->node_;
        const auto B_node = other.node_;

        const auto A = *this;
        const auto B = other;

        if (A_node) {
            result.node_->parents.push_back(A_node);
        }

        if (B_node) {
            result.node_->parents.push_back(B_node);
        }

        result.node_->backward_fn =
            [A, B](const Tensor& grad_out) {
                if (A.requires_grad()) {
                    Tensor grad_a = grad_out * B.detach();
                    A.accumulate_grad(grad_a);
                }
                if (B.requires_grad()) {
                    Tensor grad_b = grad_out * A.detach();
                    B.accumulate_grad(grad_b);
                }
            };
    } else {
        result.node_.reset();
    }

    return result;
}