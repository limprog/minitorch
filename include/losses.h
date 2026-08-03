#pragma once

#include <Tensor.h>

namespace corr {
    Tensor mean_squares_error(const Tensor& y_pred, const Tensor& y_target);
    Tensor cross_entropy_loss(const Tensor& y_pred, const Tensor& y_target, std::size_t class_dim);
    Tensor bce_with_logits(const Tensor& y_pred, const Tensor& y_target);
}