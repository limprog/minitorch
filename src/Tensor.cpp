//
// Created by limprog on 7/31/26.
//

#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <numeric>
#include <functional>
#include <algorithm>
#include <regex>


Tensor::Tensor(const std::initializer_list<std::size_t>& shape, float value, bool requires_grad)
    : shape_(shape)
{

    std::size_t size = std::accumulate(
        shape_.begin(),
        shape_.end(),
        std::size_t{1},
        std::multiplies<std::size_t>()
        );

    storage_ = std::make_shared<Storage>();
    storage_->data_.resize(size);
    std::fill(storage_->data_.begin(), storage_->data_.end(), value);

    shape_ = shape;
    if (requires_grad)
        node_ = std::make_shared<AutogradNode>();

    calculate_strides();


}


Tensor::Tensor(const std::vector<std::size_t>& shape, float value, bool requires_grad)
    : shape_(shape)
{

    std::size_t size = std::accumulate(
        shape_.begin(),
        shape_.end(),
        std::size_t{1},
        std::multiplies<std::size_t>()
        );

    storage_ = std::make_shared<Storage>();

    storage_->data_.resize(size);
    std::fill(storage_->data_.begin(), storage_->data_.end(), value);
    shape_ = shape;
    if (requires_grad)
        node_ = std::make_shared<AutogradNode>();

    calculate_strides();
}




std::size_t Tensor::flatten_index(std::vector<std::size_t> indices) const {
    if (shape_.size() != indices.size()) {
        throw std::invalid_argument("Tensor::flatten_index: indices should be the same size");
    }

    std::size_t flat_index = 0;

    for (std::size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i]) {
            throw std::invalid_argument("Tensor::flatten_index: indices should be smaller than Tensor::shape");
        }

        flat_index += indices[i] * strides_[i];
    }

    return flat_index;
}



std::size_t Tensor::flatten_index(std::vector<std::size_t> indices, std::vector<std::size_t> strides) {
    if (strides.size() != indices.size()) {
        throw std::invalid_argument("Tensor::flatten_index: indices should be the same size");
    }

    std::size_t flat_index = 0;

    for (std::size_t i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }

    return flat_index;
}


std::vector<std::size_t> Tensor::get_logic_index(std::size_t index) const {
    if (index >= numel()) {
        throw std::invalid_argument("Tensor::unflatten_index: index out of range");
    }

    std::size_t flat = index;
    std::vector<std::size_t> result;
    result.resize(shape_.size());


    for (std::size_t dim = shape_.size(); dim-- > 0;) {
        result[dim] = flat % shape_[dim];
        flat /= shape_[dim];
    }

    return result;
}


Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor result = Tensor(shape_, 0.0f, false);
    std::vector<std::size_t> indices;


    for (std::size_t i = 0; i < result.numel(); i++) {
        indices = get_logic_index(i);
        result.storage_->data_[i] = storage_->data_[flatten_index(indices)];
    }

    if (requires_grad()) {
        auto parent_node = node_;

        result.node_ = std::make_shared<AutogradNode>();
        result.node_->parents.push_back(parent_node);

        result.node_->backward_fn = [parent_node](const Tensor& grad_out) {
            accumulate_grad(parent_node, grad_out);
        };
    }

    return result;
}


void Tensor::contiguous_() {
    if (is_contiguous()) {
        return;
    }

    std::vector<float> new_data(numel());

    for (std::size_t i = 0; i < numel(); ++i) {
        const auto indices = get_logic_index(i);

        new_data[i] =
            storage_->data_[flatten_index(indices)];
    }


    auto new_storage = std::make_shared<Storage>();
    new_storage->data_ = std::move(storage_->data_);

    storage_ = std::move(new_storage);

    calculate_strides();
}




std::size_t Tensor::batch_offset(std::size_t batch_index) const {
    std::size_t offset = 0;

    for (std::size_t i = shape_.size() - 2; i-- > 0;) {
        const std::size_t index = batch_index % shape_[i];
        batch_index /= shape_[i];

        offset += index * strides_[i];
    }

    return offset;
}



void Tensor::calculate_strides() {
    strides_.resize(shape_.size());

    std::size_t stride = 1;

    for (std::size_t i = shape_.size(); i-- > 0; ) {
        strides_[i] = stride;
        stride *= shape_[i];
    }

}


bool Tensor::is_contiguous() const {
    std::size_t stride = 1;

    for (std::size_t i = shape_.size(); i-- > 0; ) {
        if (strides_[i] != stride) {
            return false;
        }
        stride *= shape_[i];
    }

    return true;
}





float& Tensor::operator()(const std::initializer_list<std::size_t>& indices) {
    std::vector<std::size_t> index(indices);
    return storage_->data_[flatten_index(index)];

}


const float& Tensor::operator()(const std::initializer_list<std::size_t>& indices) const {
    std::vector<std::size_t> index(indices);
    return storage_->data_[flatten_index(index)];

}


const std::vector<std::size_t>& Tensor::shape() {
    return shape_;
}


void Tensor::flat_index_fill(std::size_t index, float value) {
    storage_->data_[index] = value;
}



Tensor& Tensor::fill_(float value) {
    std::fill(storage_->data_.begin(), storage_->data_.end(), value);
    return *this;
}



Tensor Tensor::copy() const {
    Tensor result = *this;

    result.storage_= std::make_shared<Storage>(*storage_);

    if (requires_grad()) {
        const auto A_node = node_;

        result.node_ = std::make_shared<AutogradNode>();
        result.node_->parents.push_back(A_node);

        result.node_->backward_fn =[A_node](const Tensor& grad) {
            accumulate_grad(A_node, grad);
        };
    } else {
        result.node_.reset();
    }

    return result;
}


Tensor Tensor::copy_for_operation() const {
    Tensor result = *this;

    result.storage_ = std::make_shared<Storage>(*storage_);
    result.node_ = std::make_shared<AutogradNode>();
    return result;
}



Tensor Tensor::detach() const {
    Tensor result = *this;

    result.node_.reset();

    return result;
}


std::size_t Tensor::numel() const {

    std::size_t result = 1;
    for (std::size_t i = 0; i < shape_.size(); i++) {
        result *= shape_[i];
    }
    return result;
}



std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[";

    for (std::size_t i = 0; i < tensor.shape_.size(); i++) {
        os << tensor.shape_[i];


        if (i + 1 < tensor.shape_.size()) {
            os << ", ";
        }
    }

    os << "], data=[";
    for (std::size_t i = 0; i < tensor.storage_->data_.size(); i++) {
        os << tensor.storage_->data_[i];

        if (i + 1 < tensor.storage_->data_.size()) {
            os << ", ";
        }
    }

    os << "], requires_grad=";
    os << tensor.requires_grad() << ")";
    return os;
}