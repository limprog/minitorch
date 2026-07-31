//
// Created by limprog on 7/31/26.
//

#include "Tensor.h"

#include <stdexcept>
#include <iostream>
#include <numeric>
#include <functional>
#include <algorithm>


Tensor::Tensor(const std::initializer_list<std::size_t>& shape, float value)
    : shape_(shape)
{

    std::size_t size = std::accumulate(
        shape_.begin(),
        shape_.end(),
        std::size_t{1},
        std::multiplies<std::size_t>()
        );

    data_.resize(size);
    std::fill(data_.begin(), data_.end(), value);
    shape_ = shape;

    calculate_strides();

}


Tensor::Tensor(const std::vector<std::size_t>& shape, float value)
    : shape_(shape)
{

    std::size_t size = std::accumulate(
        shape_.begin(),
        shape_.end(),
        std::size_t{1},
        std::multiplies<std::size_t>()
        );

    data_.resize(size);
    std::fill(data_.begin(), data_.end(), value);
    shape_ = shape;

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

void Tensor::matmul_matrix_kernel(const Tensor &other, Tensor &result, std::size_t offset_a, std::size_t offset_b,
    std::size_t offset_result, std::size_t rows, std::size_t cols, std::size_t inner) const {
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            float sum = 0;
            for (std::size_t k = 0; k < inner; k++) {
               sum += data_[offset_a + i * inner + k] * other.data_[offset_b + k * cols + j];
            }
            result.data_[offset_result + i * cols + j] = sum;
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
            batch * rows * inner,
            batch * cols * inner,
            batch * cols * rows,
            rows,
            cols,
            inner);
    }
}


void Tensor::calculate_strides() {
    strides_.resize(shape_.size());

    std::size_t stride = 1;

    for (std::size_t i = shape_.size(); i-- > 0; ) {
        strides_[i] = stride;
        stride *= shape_[i];
    }

}


float& Tensor::operator()(const std::initializer_list<std::size_t>& indices) {
    std::vector<std::size_t> index(indices);
    return data_[flatten_index(index)];

}


const float& Tensor::operator()(const std::initializer_list<std::size_t>& indices) const {
    std::vector<std::size_t> index(indices);
    return data_[flatten_index(index)];

}


const std::vector<std::size_t>& Tensor::get_shape() {
    return shape_;
}


Tensor& Tensor::fill_(float value) {
    std::fill(data_.begin(), data_.end(), value);
    return *this;
}


Tensor& Tensor::unsqueeze_(std::size_t axis) {
    if (axis > shape_.size()) {
        throw std::invalid_argument("Tensor::unsqueeze_: axis must be less than Tensor::shape");
    }

    shape_.insert(shape_.begin() + axis, 1);

    calculate_strides();

    return *this;
}


Tensor Tensor::unsqueeze(std::size_t axis) const{
    Tensor result(*this);

    result.unsqueeze_(axis);
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
    Tensor result = *this;
    result.squeeze_(axis);
    return result;
}



Tensor Tensor::operator+(const Tensor& other) const{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+: shape should be the same");
    }

    Tensor result(other.shape_);

    for (std::size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}


Tensor Tensor::operator*(const Tensor& other) const{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+: shape should be the same");
    }

    Tensor result(other.shape_);

    for (std::size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    return result;
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
        return result;
    }

    if (lhs_dim== 2 && rhs_dim == 1) {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }

        Tensor result({shape_[0]});
        matmul_matrix_kernel(other, result, 0, 0, 0, shape_[0], 1, shape_[1]);
        return result;

    }

    if (lhs_dim == 1 && rhs_dim == 2) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }

        Tensor result({other.shape_[1]});
        matmul_matrix_kernel(other, result, 0, 0, 0, 1, other.shape_[1], shape_[0]);
        return result;
    }

    if (lhs_dim == 1 && rhs_dim == 1) {
        if (shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Tensor::matmul: shape should be the same");
        }

        Tensor result(std::vector<std::size_t>{});
        float sum = 0;

        for (int i = 0; i < other.shape_[0]; i++) {
            sum += other.data_[i] * data_[i];
        }

        result.data_[0] = sum;
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
    return result;

}


Tensor& Tensor::reshape_(std::initializer_list<std::size_t> shape) {
    std::size_t new_size = std::accumulate(
        shape.begin(),
        shape.end(),
        std::size_t{1},
        std::multiplies<std::size_t>());
    if (new_size != data_.size()) {
        throw std::invalid_argument("Tensor::reshape: new_size should be equal to data_.size()");
    }

    shape_ = shape;
    calculate_strides();

    return *this;
}


Tensor Tensor::reshape(std::initializer_list<std::size_t> shape) const{
    Tensor result(*this);
    result.reshape_(shape);
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
    for (std::size_t i = 0; i < tensor.data_.size(); i++) {
        os << tensor.data_[i];

        if (i + 1 < tensor.data_.size()) {
            os << ", ";
        }
    }

    os << "])";
    return os;
}