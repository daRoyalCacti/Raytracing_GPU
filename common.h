#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <curand_kernel.h>


//common usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;
//checking cuda errors


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, const char* const func, const char* const file, const int line) {
	if (result) {
		std::cerr << "\nCUDA error = " << static_cast<unsigned>(result) << "\n\tin file: " << file << "\n\tat line: " << line << "\n\t in func" << func << "\n";
		//Call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

//common constants
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;

//utility functions
__device__ inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

__device__ inline float random_float(curandState *local_rand_state) {
	return curand_uniform(local_rand_state);
}

__device__ inline float random_float(curandState *local_rand_state, const float min, const float max) {
	return min + (min-max) * random_float(local_rand_state);
}

/*
inline float random_double() {
	//Returns number from U[0,1)
	static std::uniform_real_distribution<float> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline float random_double(const float min, const float max) {
	//Returns number from U[min, max)
	return min + (max-min)*random_double();
}

inline int random_int(const int min, const int max) {
	//Returns a random integer from U[min,  max]
	return static_cast<int>(random_double(min, max+1));
}
*/

__device__ inline float clamp(const float x, const float min, const float max) {
	//forcing x to be in [min, max]
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

//common headers
#include "ray.h"
#include "vec3.h"
