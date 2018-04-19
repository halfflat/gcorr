#pragma once

#include <cstddef>
#include <vector>

#include <cuComplex.h>

std::vector<float2> random_cint_data(std::size_t n, int vmin, int vmax, int seed = 12345);
std::vector<float2> random_cfloat_data(std::size_t n, float vmin, float vmax, int seed = 12345);
std::vector<float> random_float_data(std::size_t n, float vmin, float vmax, int seed = 12345);
