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

    Tensor result(other.shape_);

    for (std::size_t i = 0; i < storage_->data_.size(); i++) {
        result.storage_->data_[i] = storage_->data_[i] + other.storage_->data_[i];
    }

    if (requires_grad() || other.requires_grad()) {
        ;
    }

    return result;
}


Tensor Tensor::operator*(const Tensor& other) const{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+: shape should be the same");
    }

    Tensor result(other.shape_);

    for (std::size_t i = 0; i < storage_->data_.size(); i++) {
        result.storage_->data_[i] = storage_->data_[i] * other.storage_->data_[i];
    }

    return result;
}