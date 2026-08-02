#include "Tensor.h"
#include "random.h"

Tensor Tensor::zeros(const std::initializer_list<std::size_t> shape) {
    Tensor tensor(shape);

    return tensor;
}

Tensor Tensor::ones(const std::initializer_list<std::size_t> shape) {
    Tensor tensor(shape, 1.0f);

    return tensor;
}


Tensor Tensor::unifrom(const std::initializer_list<std::size_t> shape, float min, float max) {
    Tensor tensor(shape);

    for (std::size_t i = 0; i < tensor.numel(); i++) {
        tensor.storage_->data_[i] = rng::uniform(min, max);
    }

    return tensor;
}


Tensor Tensor::normal(const std::initializer_list<std::size_t> shape, float mean, float stddev) {
    Tensor tensor(shape);

    for (std::size_t i = 0; i < tensor.numel(); i++) {
        tensor.storage_->data_[i] = rng::normal(mean, stddev);
    }

    return tensor;
}


Tensor Tensor::zeros_like(const Tensor &tensor) {
    Tensor new_tensor(tensor.shape_);
    return new_tensor;
}


Tensor Tensor::ones_like(const Tensor &tensor) {
    Tensor new_tensor(tensor.shape_, 1.0f);

    return new_tensor;
}


Tensor Tensor::unifrom_like(const Tensor &tensor, float min, float max) {
    Tensor new_tensor(tensor.shape_);

    for (std::size_t i = 0; i < new_tensor.numel(); i++) {
        new_tensor.storage_->data_[i] = rng::uniform(min, max);
    }
    return new_tensor;
}



Tensor Tensor::normal_like(const Tensor &tensor, float mean, float stddev) {
    Tensor new_tensor(tensor.shape_);

    for (std::size_t i = 0; i < new_tensor.numel(); i++) {
        new_tensor.storage_->data_[i] = rng::normal(mean, stddev);
    }

    return new_tensor;
}
