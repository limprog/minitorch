//
// Created by limprog on 8/1/26.
//

#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <numeric>
#include <functional>
#include <algorithm>
#include <regex>



Tensor& Tensor::unsqueeze_(std::size_t axis) {
    if (axis > shape_.size()) {
        throw std::invalid_argument("Tensor::unsqueeze_: axis must be less than Tensor::shape");
    }
    std::size_t new_straide;

    if (axis == shape_.size()) {
        new_straide = 1;
    } else {
        new_straide = strides_[axis] * shape_[axis];
    }

    shape_.insert(shape_.begin() + axis, 1);
    strides_.insert(strides_.begin() + axis, new_straide);

    return *this;
}


Tensor Tensor::unsqueeze(std::size_t axis) const{
    Tensor result = copy_for_operation();
    result.unsqueeze_(axis);

    if (requires_grad()) {
        const auto A_node = this->node_;

        result.node_->parents.push_back(A_node);

        result.node_->backward_fn = [axis, A_node](const Tensor& grad_out) {
            accumulate_grad(A_node, grad_out.squeeze(axis));
        };
    } else {
        result.node_.reset();
    }
    return result;
}


Tensor& Tensor::squeeze_(std::size_t axis) {
    if (axis >= shape_.size()) {
        throw std::invalid_argument("Tensor::squeeze: axis must be less than Tensor::shape");
    }

    if (shape_[axis] != 1) {
        throw std::invalid_argument("Tensor::squeeze: selected axis must have size 1");
    }
    shape_.erase(shape_.begin() + axis);
    strides_.erase(strides_.begin() + axis);

    return *this;
}


Tensor Tensor::squeeze(std::size_t axis) const {
    Tensor result = copy_for_operation();
    result.squeeze_(axis);

    if (requires_grad()) {
        const auto A_node = this->node_;

        result.node_->parents.push_back(A_node);

        result.node_->backward_fn = [axis, A_node](const Tensor& grad_out) {
            accumulate_grad(A_node, grad_out.unsqueeze(axis));
        };
    } else {
        result.node_.reset();
    }
    return result;
}


Tensor Tensor::transpose(std::size_t dim1, std::size_t dim2) const {
    Tensor result = *this;

    std::swap(result.shape_[dim1], result.shape_[dim2]);
    std::swap(result.strides_[dim1], result.strides_[dim2]);

    if (requires_grad()) {
        const auto parent = *this;

        result.node_ = std::make_shared<AutogradNode>();
        result.node_->parents = {parent.node_};

        result.node_->backward_fn =
            [parent, dim1, dim2](const Tensor& grad_out) {
                Tensor grad = grad_out.transpose(dim1, dim2);

                parent.accumulate_grad(grad);
        };

    } else {
        result.node_.reset();
    }
    return result;

}


Tensor Tensor::T() const {
    if (shape_.size() < 2) {
        throw std::invalid_argument("Tensor::T: shape must be at least two");
    }
    return transpose(shape_.size() - 2, shape_.size() - 1);
}



Tensor& Tensor::reshape_(std::initializer_list<std::size_t> shape) {
    std::size_t new_size = std::accumulate(
        shape.begin(),
        shape.end(),
        std::size_t{1},
        std::multiplies<std::size_t>());
    if (new_size != storage_->data_.size()) {
        throw std::invalid_argument("Tensor::reshape: new_size should be equal to storege_->data_.size()");
    }

    if (!is_contiguous()) {
        throw std::invalid_argument("Tensor::reshape: tensor must be contiguoes");
    }


    shape_ = shape;
    calculate_strides();

    return *this;
}


Tensor& Tensor::reshape_(std::vector<std::size_t> shape) {
    std::size_t new_size = std::accumulate(
        shape.begin(),
        shape.end(),
        std::size_t{1},
        std::multiplies<std::size_t>());
    if (new_size != storage_->data_.size()) {
        throw std::invalid_argument("Tensor::reshape: new_size should be equal to storege_->data_.size()");
    }

    if (!is_contiguous()) {
        throw std::invalid_argument("Tensor::reshape: tensor must be contiguoes");
    }


    shape_ = shape;
    calculate_strides();

    return *this;
}


Tensor Tensor::reshape(std::initializer_list<std::size_t> shape) const{
    Tensor result = copy_for_operation();
    result.reshape_(shape);

    const std::vector<std::size_t> new_shape(shape);
    const std::vector<std::size_t> old_shape = shape_;
    const auto parent = node_;

    if (requires_grad()) {
      result.node_->parents = {parent};

      result.node_->backward_fn = [parent, old_shape](const Tensor& grad_out) {
          accumulate_grad(parent, grad_out.reshape(old_shape));

      };
    } else {
      result.node_.reset();
    }
    return result;
}


Tensor Tensor::reshape(std::vector<std::size_t> shape) const{
    Tensor result = copy_for_operation();
    result.reshape_(shape);

    const std::vector<std::size_t> new_shape(shape);
    const std::vector<std::size_t> old_shape = shape_;
    const auto parent = node_;

    if (requires_grad()) {
        result.node_->parents = {parent};

        result.node_->backward_fn = [parent, old_shape](const Tensor& grad_out) {
            accumulate_grad(parent, grad_out.reshape(old_shape));

        };
    } else {
      result.node_.reset();
    }
    return result;
}