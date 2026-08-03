//
// Created by limprog on 8/1/26.
//

#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <regex>
#include <valarray>


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

    for (std::size_t flat = 0; flat < result.numel(); flat++) {
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

    Tensor result(new_shape, 0.0f, false);

    std::vector<std::size_t> indices(tensor.shape_.size());

    for (std::size_t flat = 0; flat < tensor.numel(); flat++) {
        std::size_t temp = flat;
        for (std::size_t dim = tensor.shape_.size(); dim-- > 0; ) {
            indices[dim] = temp % tensor.shape_[dim];
            temp /= tensor.shape_[dim];
        }
        const std::size_t index = broadcast_index(indices, result.shape_, result.strides_);

        result.storage_->data_[index] += tensor.storage_->data_[Tensor::flatten_index(indices, tensor.strides_)];
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


Tensor Tensor::operator/(const Tensor& other) const {
    for (float i : other.storage_->data_) {
        if (i == 0.0f)
            throw std::invalid_argument("Tensor::operator/: division by zero");
    }

    Tensor result = binary_operation_kernel(other, [](float lhs, float rhs){return lhs / rhs;});

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

        result.node_->backward_fn = [A_node, B_node, A_data, B_data](const Tensor& grad_out) {
            if (A_node) {
                Tensor grad_A = grad_out / B_data;
                grad_A = sum_to_shape_without_grad(grad_A, A_data.shape_);
                accumulate_grad(A_node, grad_A);
            }
            if (B_node) {
                Tensor grad_b = -grad_out * (A_data / B_data / B_data);
                grad_b = sum_to_shape_without_grad(grad_b, B_data.shape_);
                accumulate_grad(B_node, grad_b);
            }
        };
    }

    return  result;
}


Tensor Tensor::operator*(float scalar) const {
    Tensor result = copy_for_operation();

    for (std::size_t i = 0; i < numel(); i++) {
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


Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::invalid_argument(
            "Tensor::operator/: division by zero"
        );
    }

    return *this * (1.0f / scalar);
}


Tensor operator/(float scalar, const Tensor& tensor) {
    Tensor result = tensor.copy_for_operation();

    for (std::size_t i = 0; i < result.numel(); i++) {
        result.storage_->data_[i] = scalar / result.storage_->data_[i];
    }

    if (tensor.requires_grad()) {
        const auto A_node = tensor.node_;
        const auto A_data = tensor.detach();

        result.node_->parents.push_back(A_node);

        result.node_->backward_fn = [scalar, A_node, A_data](const Tensor& grad_out) {
            Tensor::accumulate_grad(A_node, grad_out * (-scalar) / A_data / A_data);
        };
    } else {
        result.node_.reset();
    }

    return result;
}


void Tensor::sub_(const Tensor &other, float alpha) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::sub_: shape mismatch");
    }

    if (!is_contiguous()) {
        throw std::runtime_error(
            "Tensor::sub_: in-place update requires contiguous tensor"
        );
    }

    Tensor other_sub = other.detach().contiguous();

    for (std::size_t i = 0; i < numel(); i++) {
        storage_->data_[i] -= other_sub.storage_->data_[i] *  alpha;
    }
}



Tensor Tensor::sum() const {
    Tensor result({1}, 0, requires_grad());

    for (std::size_t i = 0; i < numel(); i++) {
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
    return sum() * (1.0f / static_cast<float>(numel()));
}


Tensor Tensor::exp() const{
    Tensor result = copy_for_operation();

    for (std::size_t i = 0; i < numel(); i++) {
        result.storage_->data_[i] = std::exp(result.storage_->data_[i]);
    }

    if (requires_grad()) {
        const auto parent_node = this->node_;
        const auto parent_exp = result.detach();

        result.node_ = std::make_shared<AutogradNode>();

        result.node_->parents.push_back(parent_node);
        result.node_->backward_fn = [parent_node, parent_exp](const Tensor& grad_out) {
            accumulate_grad(parent_node, grad_out * parent_exp);
        };
    }else {
        result.node_.reset();
    }

    return result;
}


Tensor Tensor::log() const {
    Tensor result = copy_for_operation();

    for (std::size_t i = 0; i < numel(); i++) {
        result.storage_->data_[i] = std::log(result.storage_->data_[i]);
    }

    if (requires_grad()) {
        const auto parent_node = this->node_;
        const auto parent_data = this->detach();

        result.node_ = std::make_shared<AutogradNode>();
        result.node_->parents.push_back(parent_node);

        result.node_->backward_fn = [parent_node, parent_data](const Tensor& grad_out) {
            accumulate_grad(parent_node, grad_out * (1 / parent_data));
        };
    }else {
        result.node_.reset();
    }

    return result;
}


Tensor Tensor::where(const std::function<bool(float)>& condition, float if_true, float if_false) const {
    Tensor result(shape_, 0, false);

    for (std::size_t i = 0; i < result.numel(); i++) {
        if (condition(storage_->data_[i])) {
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


Tensor Tensor::sum(std::size_t dim, bool keep_dim) const {
    if (dim >= shape_.size()) {
        throw std::invalid_argument("Tensor::sum: dimension out of range");
    }

    auto new_shape = shape_;
    new_shape[dim] = 1;
    Tensor result = sum_to_shape_without_grad(*this, new_shape);
    if (!keep_dim) {
        result = result.squeeze(dim);
    }
    if (requires_grad()) {
        const auto A_node = this->node_;
        const auto old_shape = shape_;

        result.node_ = std::make_shared<AutogradNode>();
        result.node_->parents.push_back(A_node);

        result.node_->backward_fn = [A_node, old_shape, dim, keep_dim](const Tensor& grad_out) {
            Tensor grad_A = grad_out.detach();
            if (!keep_dim) {
                grad_A = grad_out.unsqueeze(dim);
            }
            accumulate_grad(A_node, grad_A.expand(old_shape).contiguous());
        };
    }
    return result;
}


Tensor Tensor::max(std::size_t dim, bool keep_dim) const{
    if (dim >= shape_.size()) {
        throw std::invalid_argument("Tensor::sum: dimension out of range");
    }

    auto new_shape = shape_;
    new_shape[dim] = 1;

    Tensor result(new_shape, -std::numeric_limits<float>::infinity(), false);

    std::vector<std::size_t> indices(result.shape_.size());

    for (std::size_t flat = 0; flat < numel(); flat++) {
        indices = get_logic_index(flat);
        const std::size_t index = broadcast_index(indices, result.shape_, result.strides_);

        result.storage_->data_[index] = std::max(storage_->data_[flatten_index(indices)], result.storage_->data_[index]);
    }
    if (!keep_dim) {
        result = result.squeeze(dim);
    }
    return result;
}


Tensor Tensor::min(std::size_t dim, bool keep_dim) const{
    if (dim >= shape_.size()) {
        throw std::invalid_argument("Tensor::sum: dimension out of range");
    }

    auto new_shape = shape_;
    new_shape[dim] = 1;

    Tensor result(new_shape, std::numeric_limits<float>::infinity(), false);

    std::vector<std::size_t> indices(result.shape_.size());

    for (std::size_t flat = 0; flat < numel(); flat++) {
        indices = get_logic_index(flat);
        const std::size_t index = broadcast_index(indices, result.shape_, result.strides_);

        result.storage_->data_[index] = std::min(storage_->data_[flatten_index(indices)], result.storage_->data_[index]);
    }
    if (!keep_dim) {
        result = result.squeeze(dim);
    }
    return result;
}
