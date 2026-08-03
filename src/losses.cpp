#include "losses.h"

#include "Tensor.h"



Tensor corr::mean_squares_error(const Tensor &y_pred, const Tensor &y_target) {
    Tensor dist = y_pred - y_target;
    return (dist * dist).mean();
}


Tensor corr::cross_entropy_loss(const Tensor &logit, const Tensor &y_target /* must be one-hot*/, std::size_t class_dim) {
    Tensor max_value = logit.max(class_dim, true).detach();
    Tensor shifted = logit - max_value;

    Tensor log_probs = shifted - shifted.exp().sum(class_dim, true).log();

    return -(y_target * log_probs).sum(class_dim, true).mean();
}

// TODO:
Tensor corr::bce_with_logits(const Tensor &y_pred, const Tensor &y_target) {
    return y_pred.sum();
}

