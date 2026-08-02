//
// Created by limprog on 8/1/26.
//

#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <regex>


std::size_t Tensor::broadcast_index(
        const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& stride) {
    const std::size_t rank_diff = indices.size() - shape.size();

    std::size_t offset = 0;
    for (std::size_t i = 0; i < shape.size(); i++) {
        const std::size_t index = shape[i] == 1 ? 0 : indices[i + rank_diff];
        offset += stride[i] * index;
    }

    return offset;

}


std::vector<std::size_t> Tensor::broadcast_shape(const std::vector<std::size_t> &a_shape, const std::vector<std::size_t> &b_shape) {
    std::size_t rank = std::max(a_shape.size(), b_shape.size());

    std::vector<std::size_t> result(rank);

    for (std::size_t i = 0; i < rank; i++) {
        std::size_t a_dim = i + 1 > a_shape.size() ? 1 : a_shape[a_shape.size() - i - 1];
        std::size_t b_dim = i + 1 > b_shape.size() ? 1 : b_shape[b_shape.size() - i - 1];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("Tensor::broadcast_shape does not support this shapes");
        }

        result[rank - i - 1] = std::max(a_dim, b_dim);
    }
    return result;
}

Tensor Tensor::binary_operation_kernel(const Tensor &other, const std::function<float(float, float)> &operation) const {
    const auto new_shape = broadcast_shape(shape_, other.shape_);

    Tensor result(new_shape, 0.0f, false);

    std::vector<std::size_t> indices(new_shape.size());

    for (std::size_t flat = 0; flat < result.storage_->data_.size(); flat++) {
        std::size_t temp = flat;

        for (std::size_t dim = new_shape.size(); dim-- > 0; ) {
            indices[dim] = temp % new_shape[dim];
            temp /= new_shape[dim];
        }

        const std::size_t index_a = broadcast_index(indices, shape_, strides_);
        const std::size_t index_b = broadcast_index(indices, other.shape_, other.strides_);

        result.storage_->data_[flat] = operation(storage_->data_[index_a], other.storage_->data_[index_b]);
    }

    return result;
}


Tensor Tensor::sum_to_shape_without_grad(const Tensor &tensor, const std::vector<std::size_t> &new_shape) {
    if (broadcast_shape(new_shape, tensor.shape_) != tensor.shape_) {
        throw std::invalid_argument(
            "Tensor::sum_to_shape_without_grad: incompatible shapes"
        );
    }

    if (!tensor.is_contiguous())
        throw std::invalid_argument("Tensor::sum_to_shape_without_grad: tensor is not contiguous");

    Tensor result(new_shape, 0.0f, false);

    std::vector<std::size_t> indices(tensor.shape_.size());

    for (std::size_t flat = 0; flat < tensor.storage_->data_.size(); flat++) {
        std::size_t temp = flat;
        for (std::size_t dim = tensor.shape_.size(); dim-- > 0; ) {
            indices[dim] = temp % tensor.shape_[dim];
            temp /= tensor.shape_[dim];
        }
        const std::size_t index = broadcast_index(indices, result.shape_, result.strides_);

        result.storage_->data_[index] += tensor.storage_->data_[flat];
    }
    return result;
}



Tensor Tensor::operator+(const Tensor& other) const{
    Tensor result = binary_operation_kernel(other, [](float lhs, float rhs){return lhs + rhs;});

    if (requires_grad() || other.requires_grad()) {
        result.node_ = std::make_shared<AutogradNode>();

        const auto A_node = this->node_;
        const auto B_node = other.node_;

        const auto A_shape = this->shape_;
        const auto B_shape = other.shape_;

        if (A_node) {
            result.node_->parents.push_back(A_node);
        }

        if (B_node) {
            result.node_->parents.push_back(B_node);
        }

        result.node_->backward_fn =
            [A_node, B_node, A_shape, B_shape](const Tensor& grad) {
                if (A_node) {
                    Tensor grad_A = sum_to_shape_without_grad(grad, A_shape);
                    accumulate_grad(A_node, grad_A);
                }
                if (B_node) {
                    Tensor grad_B = sum_to_shape_without_grad(grad, B_shape);
                    accumulate_grad(B_node, grad_B);
                }
            };
    }

    return result;
}


Tensor Tensor::operator-() const {
    return (*this) * -1.0f;
}


Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result = binary_operation_kernel(other, [](float lhs, float rhs){return lhs - rhs;});


    if (requires_grad() || other.requires_grad()) {
        const auto A_node = this->node_;
        const auto B_node = other.node_;

        const auto A_shape = this->shape_;
        const auto B_shape = other.shape_;

        result.node_ = std::make_shared<AutogradNode>();

        if (A_node) {
            result.node_->parents.push_back(A_node);
        }

        if (B_node) {
            result.node_->parents.push_back(B_node);
        }

        result.node_->backward_fn =
            [A_node, B_node, A_shape, B_shape](const Tensor& grad) {
                if (A_node) {
                    Tensor grad_A = sum_to_shape_without_grad(grad, A_shape);
                    accumulate_grad(A_node, grad_A);
                }
                if (B_node) {
                    Tensor grad_B = sum_to_shape_without_grad(-grad, B_shape);
                    accumulate_grad(B_node, grad_B);
                };
        };
    }

    return result;
}



Tensor Tensor::operator*(const Tensor& other) const{
    Tensor result = binary_operation_kernel(other, [](float lhs, float rhs){return lhs * rhs;});

    if (requires_grad() || other.requires_grad()) {
        const auto A_node = this->node_;
        const auto B_node = other.node_;

        const auto A_data = this->detach();
        const auto B_data = other.detach();
        result.node_ = std::make_shared<AutogradNode>();
        if (A_node) {
            result.node_->parents.push_back(A_node);
        }

        if (B_node) {
            result.node_->parents.push_back(B_node);
        }

        result.node_->backward_fn =
            [A_node, B_node, A_data, B_data](const Tensor& grad_out) {
                if (A_node) {
                    Tensor grad_A = grad_out * B_data;
                    grad_A = sum_to_shape_without_grad(grad_A, A_data.shape_);
                    accumulate_grad(A_node, grad_A);
                }
                if (B_node) {
                    Tensor grad_b = grad_out * A_data;
                    grad_b = sum_to_shape_without_grad(grad_b, B_data.shape_);
                    accumulate_grad(B_node, grad_b);
                }
            };
    }

    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result = copy_for_operation();

    for (std::size_t i = 0; i < storage_->data_.size(); i++) {
        result.storage_->data_[i] = storage_->data_[i] * scalar;
    }

    if (requires_grad()) {
        const auto A_node = this->node_;
        result.node_->parents.push_back(A_node);

        result.node_->backward_fn = [scalar, A_node](const Tensor& grad_out) {
            accumulate_grad(A_node, grad_out * scalar);
        };
    } else {
        result.node_.reset();
    }

    return result;
}


Tensor Tensor::sum() const {
    Tensor result({1}, 0, requires_grad());

    for (std::size_t i = 0; i < storage_->data_.size(); i++) {
        result.storage_->data_[0] += storage_->data_[i];
    }

    if (requires_grad()) {
        const auto A_shape = this->shape_;
        const auto A_node = this->node_;

        result.node_->parents.push_back(A_node);
        result.node_->backward_fn = [A_node, A_shape](const Tensor& grad_out) {
            Tensor grad = Tensor(A_shape, grad_out.storage_->data_[0], false);
            accumulate_grad(A_node, grad);
        };
    }
    else {
        result.node_.reset();
    }
    return result;
}


Tensor Tensor::mean() const {
    return sum() * (1.0f / static_cast<float>(storage_->data_.size()));
}


Tensor Tensor::where(const std::function<bool(float)>& condition, float if_true, float if_false) const {
    Tensor result(shape_, 0, false);

    for (std::size_t i = 0; i < result.numel(); i++) {
        if (condition(result.storage_->data_[i])) {
            result.storage_->data_[i] = if_true;
        } else {
            result.storage_->data_[i] = if_false;
        }
    }
    return result;
}


Tensor Tensor::relu() const{
    Tensor result = copy_for_operation();

    for (std::size_t i = 0; i < result.numel(); i++) {
        if (result.storage_->data_[i] <= 0) {
            result.storage_->data_[i] = 0;
        }
    }

    if (requires_grad()) {
        const auto A_node = this->node_;
        const auto A_data = this->detach();

        result.node_->parents.push_back(A_node);
        result.node_->backward_fn = [A_node, A_data](const Tensor& grad_out) {
            const auto grad_mask = A_data.where([](float x){return x > 0;}, 1, 0);
            accumulate_grad(A_node, grad_mask*grad_out);
        };
    } else {
        result.node_.reset();
    }

    return result;
}


Tensor Tensor::leaky_relu(float alpha) const {
    Tensor result = copy_for_operation();

    for (std::size_t i = 0; i < result.numel(); i++) {
        if (result.storage_->data_[i] <= 0) {
            result.storage_->data_[i] *= alpha;
        }
    }
    if (requires_grad()) {
        const auto A_node = this->node_;
        const auto A_data = this->detach();

        result.node_->parents.push_back(A_node);
        result.node_->backward_fn = [A_node, A_data, alpha](const Tensor& grad_out) {
            auto grad_mask = A_data.where([](float x){return x > 0;}, 1, alpha);
            accumulate_grad(A_node, grad_mask * grad_out);
        };
    }
    return result;
}



Tensor operator*(float scalar, const Tensor& other) {
    return other * scalar;
}


