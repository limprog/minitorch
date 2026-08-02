
#include <nn/Module.h>
#include <nn/Linear.h>
#include <Tensor.h>



void Linear::init_weight(InitType init_type) {
    switch (init_type) {
        case InitType::KaimingNormal:
            init::kaiming_normal(weights_);
        break;
        case InitType::XavierUniform:
            init::xavier_uniform(weights_);
        break;
        case InitType::Old:
            init::old(weights_);
        break;
        case InitType::Ones:
            init::ones(weights_);
        break;
        default:
            throw std::invalid_argument("Invalid initialization type");
    }
}


Linear::Linear(std::size_t num_inputs, std::size_t num_outputs, InitType init_type, bool is_bias_)
    : Module(),
      weights_({num_inputs, num_outputs}),
      bias_({num_outputs}, 0, is_bias_),
      use_bias_(is_bias_){

    init_weight(init_type);

    add_parameter(weights_);

    if (is_bias_)
        add_parameter(bias_);
}



Tensor Linear::forward(const Tensor &input) const {
    if (use_bias_) {
        return input.matmul(weights_)  + bias_;
    }
    return input.matmul(weights_);
}

