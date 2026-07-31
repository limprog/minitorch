//
// Created by limprog on 7/31/26.
//

#include "Tensor.h"

#include <stdexcept>


std::size_t Tensor::flatten_index(std::vector<std::size_t> indices) const {
    if (shape_.size() != indices.size()) {
        throw std::invalid_argument("Tensor::flatten_index: indices should be the same size");
    }

    std::size_t flat_index = 0;

    for (std::size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i]) {
            throw std::invalid_argument("Tensor::flatten_index: indices should be smaller than Tensor::shape");
        }

        flat_index += indices[i] * this->strides_[i];
    }

    return flat_index;
}



void Tensor::calculate_strides() {
    this->strides_.resize(shape_.size());

    std::size_t stride = 1;

    for (std::size_t i = shape_.size() - 1; i-- > 0; ) {
        strides_[i] = stride;
        stride *= shape_[i];
    }

}


Tensor::Tensor(const std::initializer_list<std::size_t>& shape, float value)
    : shape_(shape)
{

    std::size_t temp = 1;

    for (std::size_t i = 0; i < shape.size(); i++) {
        temp *= shape_[i];
    }

    data_.resize(temp);
    std::fill(data_.begin(), data_.end(), value);
    shape_ = shape;

    calculate_strides();

}


float& Tensor::operator()(const std::initializer_list<std::size_t>& indices) {
    std::vector<std::size_t> index(indices);
    return data_[flatten_index(index)];

}

const std::vector<std::size_t>& Tensor::get_shape() {
    return shape_;
}


