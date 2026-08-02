//
// Created by limprog on 8/2/26.
//

#include "nn/Module.h"
#include "Tensor.h"


void Module::add_parameter(Tensor& parameter) {
    parameters_.push_back(&parameter);
}


void Module::add_child(Module& child) {
    children_.push_back(&child);
}


std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> params = parameters_;

    for (Module* child : children_) {
        auto child_params = child->parameters();

        params.insert(params.end(),
            child_params.begin(),
            child_params.end());
    }

    return params;
}