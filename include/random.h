#pragma once

#include <random>

namespace rng {

    void manual_seed(int seed);

    float normal(float mean, float std);
    float uniform(float min, float max);
}