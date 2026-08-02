#include "nn/init.h"
#include "random.h"

void init::kaiming_normal(Tensor &tensor) {
    const auto shape = tensor.shape();

    if (shape.size() != 2) {
        throw std::runtime_error("init::kaiming_normal: tensor shape must be 2-D");
    }

    std::size_t in_features = shape[0];
    std::size_t out_features = shape[1];

    float stddev = std::sqrt(2.0f / in_features);
    for (std::size_t i = 0; i < in_features * out_features; i++) {
        tensor.flat_index_fill(i, rng::normal(0, stddev));
    }

}


void init::xavier_uniform(Tensor &tensor) {
    const auto shape = tensor.shape();

    if (shape.size() != 2) {
        throw std::runtime_error("init::xavier_normal: tensor shape must be 2-D");
    }

    std::size_t in_features = shape[0];
    std::size_t out_features = shape[1];
    float min = -std::sqrt(6 / (out_features + in_features));
    float max = std::sqrt(6 / (in_features + out_features));

    for (std::size_t i = 0; i < in_features * out_features; i++) {
        tensor.flat_index_fill(i, rng::uniform(min, max));
    }
}



void init::old(Tensor &tensor) {
    const auto shape = tensor.shape();

    if (shape.size() != 2) {
        throw std::runtime_error("init::xavier_normal: tensor shape must be 2-D");
    }

    std::size_t in_features = shape[0];
    std::size_t out_features = shape[1];
    float min = -(1 / std::sqrt(out_features));
    float max = 1 / std::sqrt(out_features);

    for (std::size_t i = 0; i < in_features * out_features; i++) {
        tensor.flat_index_fill(i, rng::uniform(min, max));
    }
}

void init::zeros(Tensor &tensor) {
    const auto shape = tensor.shape();

    for (std::size_t i = 0; i < tensor.numel(); i++) {
        tensor.flat_index_fill(i, 0);
    }
}


void init::ones(Tensor &tensor) {
    const auto shape = tensor.shape();

    for (std::size_t i = 0; i < tensor.numel(); i++) {
        tensor.flat_index_fill(i, 1);
    }
}