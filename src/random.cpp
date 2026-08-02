#include "random.h"

#include <random>


namespace {
    std::mt19937_64 gen(42);
}

void rng::manual_seed(int seed) {
    gen.seed(seed);
}

float rng::normal(float mean, float std) {
    std::normal_distribution<float> distribution(mean, std);
    return distribution(gen);
}

float rng::uniform(float min, float max) {
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(gen);
}


