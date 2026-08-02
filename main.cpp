#include <iostream>

#include "nn/SimpleModel.h"



int main() {
    SimpleModel model(10, 1);
    const auto pars = model.parameters();

    for (int i = 0; i < pars.size(); ++i) {
        std::cout << *pars[i] << std::endl;
    }

}
