#include "Tensor.h"

#include <cassert>
#include <cmath>
#include <iostream>

bool close(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

int main() {
    Tensor A({2, 2}, 0.0f, true);
    A[0, 0] = 1.0f;
    A[0, 1] = 2.0f;
    A[1, 0] = 3.0f;
    A[1, 1] = 4.0f;

    Tensor B({2, 2}, 0.0f, true);
    B[0, 0] = 5.0f;
    B[0, 1] = 6.0f;
    B[1, 0] = 7.0f;
    B[1, 1] = 8.0f;

    Tensor C = A.matmul(B);
    Tensor loss = (C * C).mean();

    loss.backward();

    Tensor grad_A = A.grad();
    Tensor grad_B = B.grad();

    std::cout << "C:\n";
    std::cout << C[0, 0] << ' ' << C[0, 1] << '\n';
    std::cout << C[1, 0] << ' ' << C[1, 1] << '\n';

    std::cout << "loss = " << loss[0] << '\n';

    std::cout << "grad A:\n";
    std::cout << grad_A[0, 0] << ' ' << grad_A[0, 1] << '\n';
    std::cout << grad_A[1, 0] << ' ' << grad_A[1, 1] << '\n';

    std::cout << "grad B:\n";
    std::cout << grad_B[0, 0] << ' ' << grad_B[0, 1] << '\n';
    std::cout << grad_B[1, 0] << ' ' << grad_B[1, 1] << '\n';

    assert(close(C[0, 0], 19.0f));
    assert(close(C[0, 1], 22.0f));
    assert(close(C[1, 0], 43.0f));
    assert(close(C[1, 1], 50.0f));

    assert(close(loss[0], 1298.5f));

    assert(close(grad_A[0, 0], 113.5f));
    assert(close(grad_A[0, 1], 154.5f));
    assert(close(grad_A[1, 0], 257.5f));
    assert(close(grad_A[1, 1], 350.5f));

    assert(close(grad_B[0, 0], 74.0f));
    assert(close(grad_B[0, 1], 86.0f));
    assert(close(grad_B[1, 0], 105.0f));
    assert(close(grad_B[1, 1], 122.0f));

    std::cout << "Complex backward test passed!\n";
}