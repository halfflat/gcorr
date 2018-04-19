#pragma once

#include <cstddef>
#include <vector>

#include <cuComplex.h>

std::vector<float2> random_cint_data(std::size_t n, int vmin, int vmax, int seed = 12345);
std::vector<float2> random_cfloat_data(std::size_t n, float vmin, float vmax, int seed = 12345);
std::vector<float> random_float_data(std::size_t n, float vmin, float vmax, int seed = 12345);

std::vector<int32_t> random_int_data(std::size_t n, int32_t vmin, int32_t vmax, int seed = 12345);
std::vector<int8_t> random_int8_data(std::size_t n, int8_t vmin, int8_t vmax, int seed = 12345);
