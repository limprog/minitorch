//
// Created by limprog on 8/1/26.
//


#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <regex>


void Tensor::matmul_matrix_kernel(const Tensor &other, Tensor &result, std::size_t offset_a, std::size_t offset_b,
    std::size_t offset_result, std::size_t rows, std::size_t cols, std::size_t inner) const {
    const std::size_t lhs_dim = shape_.size();
    const std::size_t rhs_dim = other.shape_.size();

    const std::size_t lhs_row_stride =
        lhs_dim == 1 ? 0 : strides_[lhs_dim - 2];

    const std::size_t lhs_col_stride =
        strides_[lhs_dim - 1];

    const std::size_t rhs_row_stride =
        rhs_dim == 1
            ? other.strides_[0]
            : other.strides_[rhs_dim - 2];

    const std::size_t rhs_col_stride =
        rhs_dim == 1
            ? 0
            : other.strides_[rhs_dim - 1];

    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            float sum = 0;
            for (std::size_t k = 0; k < inner; k++) {
                sum += storage_->data_[offset_a + i * lhs_row_stride + k * lhs_col_stride] *
                    other.storage_->data_[offset_b + k * rhs_row_stride + j*rhs_col_stride];
            }
            result.storage_->data_[offset_result + i * cols + j] = sum;
        }
    }
}


void Tensor::batched_matmul(const Tensor& other,
        Tensor& result,
        std::size_t rows,
        std::size_t cols,
        std::size_t inner,
        std::size_t batch_count) const {
    for (std::size_t batch = 0; batch < batch_count; batch++) {
        matmul_matrix_kernel(
            other,
            result,
            batch_offset(batch),
            other.batch_offset(batch),
            batch * cols * rows,
            rows,
            cols,
            inner);
    }
}


Tensor Tensor::matmul(const Tensor &other) const {
    const std::size_t lhs_dim = shape_.size();
    const std::size_t rhs_dim = other.shape_.size();


    if (lhs_dim == 2 && rhs_dim == 2) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }
        Tensor result({shape_[0], other.shape_[1]});
        matmul_matrix_kernel(other, result, 0, 0, 0, shape_[0], other.shape_[1], shape_[1]);

        if (requires_grad() || other.requires_grad()) {
            const auto A_out = this->detach();
            const auto B_out = other.detach();

            const auto A_node = this->node_;
            const auto B_node = other.node_;


            if (A_node){
                result.node_->parents.push_back(A_node);
            }
            if (B_node){
                result.node_->parents.push_back(B_node);
            }

            result.node_->backward_fn = [A_node, B_node, A_out, B_out](const Tensor &grad_out) {
                Tensor grad_A = grad_out.matmul(B_out.T());
                Tensor grad_B = A_out.T().matmul(grad_out);

                if (A_node) accumulate_grad(A_node, grad_A);
                if (B_node) accumulate_grad(B_node, grad_B);
            };
        } else {
          result.node_.reset();
        }

        return result;
    }

    if (lhs_dim== 2 && rhs_dim == 1) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }

        Tensor result({shape_[0]});
        matmul_matrix_kernel(other, result, 0, 0, 0, shape_[0], 1, shape_[1]);

        if (requires_grad() || other.requires_grad()) {
            const auto A_out = this->detach();
            const auto B_out = other.detach();

            const auto A_node = this->node_;
            const auto B_node = other.node_;

            if (A_node){
                result.node_->parents.push_back(A_node);
            }
            if (B_node){
                result.node_->parents.push_back(B_node);
            }

            result.node_->backward_fn = [A_node, B_node, A_out, B_out](const Tensor &grad_out) {
                Tensor grad_A = grad_out.unsqueeze(1).matmul(B_out.unsqueeze(0));
                Tensor grad_B = A_out.T().matmul(grad_out);

                if (A_node) accumulate_grad(A_node, grad_A);
                if (B_node) accumulate_grad(B_node, grad_B);
            };
        } else {
          result.node_.reset();
        }
        return result;

    }

    if (lhs_dim == 1 && rhs_dim == 2) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }

        Tensor result({other.shape_[1]});
        matmul_matrix_kernel(other, result, 0, 0, 0, 1, other.shape_[1], shape_[0]);

        if (requires_grad() || other.requires_grad()) {
            const auto A_out = this->detach();
            const auto B_out = other.detach();

            const auto A_node = this->node_;
            const auto B_node = other.node_;

            if (A_node){
                result.node_->parents.push_back(A_node);
            }
            if (B_node){
                result.node_->parents.push_back(B_node);
            }

            result.node_->backward_fn = [A_node, B_node, A_out, B_out](const Tensor &grad_out) {
                Tensor grad_A = grad_out.matmul(B_out.T());
                Tensor grad_B = A_out.unsqueeze(1).matmul(grad_out.unsqueeze(0));

                if (A_node) accumulate_grad(A_node, grad_A);
                if (B_node) accumulate_grad(B_node, grad_B);
            };
        } else {
            result.node_.reset();
        }

        return result;
    }

    if (lhs_dim == 1 && rhs_dim == 1) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }

        Tensor result({1});
        float sum = 0;

        for (std::size_t i = 0; i < other.shape_[0]; i++) {
            sum += other.storage_->data_[i] * storage_->data_[i];
        }

        result.storage_->data_[0] = sum;

        if (requires_grad() || other.requires_grad()) {
            const auto A_out = this->detach();
            const auto B_out = other.detach();

            const auto A_node = this->node_;
            const auto B_node = other.node_;


            if (A_node){
                result.node_->parents.push_back(A_node);
            }
            if (B_node){
                result.node_->parents.push_back(B_node);
            }

            result.node_->backward_fn = [A_node, B_node, A_out, B_out](const Tensor &grad_out) {
                Tensor grad_A = grad_out.storage_->data_[0] * B_out;
                Tensor grad_B = grad_out.storage_->data_[0] * A_out;

                if (A_node) accumulate_grad(A_node, grad_A);
                if (B_node) accumulate_grad(B_node, grad_B);
            };
        } else {
            result.node_.reset();
        }

        return result;
    }

    if (lhs_dim != rhs_dim) {
        throw std::invalid_argument("Tensor::matmul: shape should be the same");
    }

    const std::size_t rows = shape_[lhs_dim - 2];
    const std::size_t cols = other.shape_[rhs_dim - 1];
    const std::size_t inner = shape_[lhs_dim - 1];
    const std::size_t other_inner = other.shape_[rhs_dim - 2];
    if (inner != other_inner) {
        throw std::invalid_argument(
            "Tensor::matmul: inner matrix dimensions must match"
        );
    }

    std::size_t batch_count = 1;
    for (size_t i = 0; i < shape_.size() - 2; i++) {
        if (shape_[i] == other.shape_[i]) {
            batch_count *= other.shape_[i];
        }
        else {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }
    }

    std::vector<std::size_t> new_shape = shape_;

    new_shape.back() = cols;

    Tensor result(new_shape);

    batched_matmul(
        other,
        result,
        rows,
        cols,
        inner,
        batch_count);


    if (requires_grad() || other.requires_grad()) {
        const auto A_out = this->detach();
        const auto B_out = other.detach();

        const auto A_node = this->node_;
        const auto B_node = other.node_;


        if (A_node){
            result.node_->parents.push_back(A_node);
        }
        if (B_node){
            result.node_->parents.push_back(B_node);
        }

        result.node_->backward_fn = [A_node, B_node, A_out, B_out](const Tensor &grad_out) {
            Tensor grad_A = grad_out.matmul(B_out.T());
            Tensor grad_B = A_out.T().matmul(grad_out);

            if (A_node) accumulate_grad(A_node, grad_A);
            if (B_node) accumulate_grad(B_node, grad_B);
        };
    } else {
      result.node_.reset();
    }


    return result;

}
