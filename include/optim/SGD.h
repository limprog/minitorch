#pragma once

#include <Tensor.h>
#include <optim/Optimazer.h>



class SGD : public Optimizer {
private:
    float learning_rate_;
public:
    SGD(std::vector<Tensor*> parameters, float learning_rate);

    void step();

};