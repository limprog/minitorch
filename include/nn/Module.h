#pragma once

#include "Tensor.h"
#include <vector>


class Module {
private:
    std::vector<Tensor*> parameters_;
    std::vector<Module*> children_;

public:

    void add_parameter(Tensor& parameter);
    void add_child(Module& child);

    std::vector<Tensor*> parameters();

    Tensor forward(const Tensor& input) const;

    Tensor operator()(const Tensor& input) const;


    Module(const Module&) = delete;
    Module() = default;


    Module& operator=(const Module&) = delete;

    Module(Module&&) = delete;
    Module& operator=(Module&&) = delete;
};

