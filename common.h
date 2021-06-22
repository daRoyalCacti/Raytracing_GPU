#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <curand_kernel.h>

//common headers
#include "ray.h"
#include "vec3.h"
#include "stb_image_ne.h"

//testing atm
#include <thrust/copy.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/swap.h>

#include <vector>


//common usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

//checking cuda errors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, const char* const func, const char* const file, const int line) {
	if (result) {
		std::cerr << "\nCUDA error = " << static_cast<unsigned>(result) << "\n\tin file: " << file << "\n\tat line: " << line << "\n\tin func " << func << "\n";
		//Call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

//common constants
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;

//utility functions
__host__ __device__ inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

__device__ inline float random_int(curandState *local_rand_state, const int min, const int max) {
	//Returns an integer from U[min,max]
	return static_cast<int>(random_float(local_rand_state, min, max+1));
}




__device__ __host__ inline float clamp(const float x, const float min, const float max) {
	//forcing x to be in [min, max]
	if (x < min) return min;
	if (x > max) return max;
	return x;
}






//uncomment - commented for debugging
/*template <typename T>
void upload_to_device(thrust::device_vector<T> &d_vec, std::vector<T> &h_ptr) {
	//thrust::host_vector<T> h_vec = h_ptr;
	//d_vec = h_vec;
	
	d_vec = h_ptr;
}*/

template <typename T>
void upload_to_device(thrust::device_ptr<T> &d_ptr, T *h_ptr, int size) {
	std::cout << "using old copy method" << std::endl;
	d_ptr = thrust::device_malloc<T>(size);
	for (int i = 0; i < size; i++) {
		d_ptr[i] = h_ptr[i];
	}
}

template <typename T>
void upload_to_device(thrust::device_ptr<T> &d_ptr, std::vector<T> &h_ptr) {
	d_ptr = thrust::device_malloc<T>(h_ptr.size());
	thrust::copy(h_ptr.begin(), h_ptr.end(), d_ptr);	
}
