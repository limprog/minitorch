#pragma once

#include <nn/Module.h>
#include <nn/init.h>
#include <Tensor.h>


class Linear : public Module {
private:
    Tensor weights_;
    Tensor bias_;
    bool use_bias_;

    void init_weight(InitType type);

public:
    Linear(std::size_t num_inputs, std::size_t num_outputs, InitType init_type = InitType::KaimingNormal, bool is_bias = true);

    Tensor forward(const Tensor& input) const;
};