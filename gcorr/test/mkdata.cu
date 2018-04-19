#include <cstddef>
#include <vector>
#include <algorithm>
#include <random>

#include <cuComplex.h>

#include "mkdata.h"

// Use complex small magnitude ints to avoid rouding artefacts in testing.
std::vector<float2> random_cint_data(std::size_t n, int vmin, int vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_int_distribution<int> U(vmin, vmax);

    std::vector<float2> data(n);
    std::generate(data.begin(), data.end(), [&]() {
	float2 f;
	f.x = U(R);
	f.y = U(R);
       	return f;
    });
    return data;
}

std::vector<float2> random_cfloat_data(std::size_t n, float vmin, float vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_real_distribution<float> U(vmin, vmax);

    std::vector<float2> data(n);
    std::generate(data.begin(), data.end(), [&]() {
	float2 f;
	f.x = U(R);
	f.y = U(R);
       	return f;
    });
    return data;
}

std::vector<float> random_float_data(std::size_t n, float vmin, float vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_real_distribution<float> U(vmin, vmax);

    std::vector<float> data(n);
    std::generate(data.begin(), data.end(), [&]() { return U(R); });
    return data;
}

std::vector<int32_t> random_int_data(std::size_t n, int vmin, int vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_int_distribution<int32_t> U(vmin, vmax);

    std::vector<int32_t> data(n);
    std::generate(data.begin(), data.end(), [&]() { return U(R); });
    return data;
}

std::vector<int8_t> random_int8_data(std::size_t n, int8_t vmin, int8_t vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_int_distribution<int8_t> U(vmin, vmax);

    std::vector<int8_t> data(n);
    std::generate(data.begin(), data.end(), [&]() { return U(R); });
    return data;
}
