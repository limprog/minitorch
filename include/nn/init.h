#pragma once

#include <Tensor.h>

enum class InitType {
    KaimingNormal,
    XavierUniform,
    Old,
    Zeros,
    Ones
};



namespace init {
    void kaiming_normal(Tensor& tensor);
    void xavier_uniform(Tensor& tensor);
    void old(Tensor& tensor);
    void zeros(Tensor& tensor);
    void ones(Tensor& tensor);
}
