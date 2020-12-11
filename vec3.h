#pragma once

#include <iostream>
#include <curand_kernel.h>
//#include <cmath>

//using std::sqrt;

//are sort of out of place here
__device__ inline float random_float(curandState *local_rand_state) {
	return curand_uniform(local_rand_state);
}

__device__ inline float random_float(curandState *local_rand_state, const float min, const float max) {
	return min + (max-min) * random_float(local_rand_state);
}


struct vec3 {
	float e[3];
	__host__ __device__ vec3() : e{0,0,0} {}
	__host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

	__host__ __device__ float x() const {return e[0];}
	__host__ __device__ float y() const {return e[1];}
	__host__ __device__ float z() const {return e[2];}

	__host__ __device__ vec3 operator -() const {return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ float operator [](int i) const {return e[i];}
	__host__ __device__ float& operator [](int i) {return e[i];}

	__host__ __device__ vec3& operator += (const vec3 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator *= (const float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	__host__ __device__ vec3& operator /= (const float t) {
		return *this *= 1/t;
	}

	__host__ __device__ inline float length() const {
		return sqrtf(length_squared());
	}

	__host__ __device__ inline float length_squared() const {
		return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
	}

	__device__ inline static vec3 random(curandState *s) {	//static because does not need a particular vec3
		return vec3(random_float(s), random_float(s), random_float(s));
	}


	__device__ inline static vec3 random(curandState *s, const float min, const float max) {
		return vec3(random_float(s, min, max), random_float(s, min, max), random_float(s, min, max));
	}


	__device__ inline bool near_zero() const {
		//Returns true if the vector is near 0 in all constituent dimensions
		const float s = 1e-6;	//what is considered 'near 0'
		return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s);
	}
};


//Type aliases
using point3 = vec3;
using color = vec3;



//vec3 Utility Functions

inline std::ostream& operator << (std::ostream &out, const vec3 &v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator + (const vec3 &u, const vec3 &v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator - (const vec3 &u, const vec3 &v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator * (const vec3 &u, const vec3 &v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator * (const float t, const vec3 &v) {
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator * (const vec3 &v, const float t) {
	return t * v;
}

__host__ __device__ inline vec3 operator / (const vec3 &v, const float t) {
	return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
	return vec3( u.e[1] * v.e[2] - u.e[2] * v.e[1],  u.e[2] * v.e[0] - u.e[0] * v.e[2],  u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3 v) {
	return v / v.length();
}


__device__ inline vec3 random_in_unit_sphere(curandState *s) {
	while (true) {
		const auto p = vec3::random(s, -1,1);
		if (p.length_squared() < 1) return p;
	}
}

__device__ inline vec3 random_in_unit_disk(curandState *s) {
	while (true) {
		const auto p = vec3(random_float(s, -1,1), random_float(s, -1,1), 0);
		if (p.length_squared() < 1) return p;
	}
}

__device__ inline vec3 random_unit_vector(curandState *s) {
	return unit_vector(random_in_unit_sphere(s));
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
	//reflects a vector v about another vector n
	return v - 2 * dot(v,n)*n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, const float etai_over_etat) {
	//Computes the refracted ray of light passing through a dielectric material using snell's law
	const auto cos_theta = min(dot(-uv, n), 1.0);	
	const vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
	const vec3 r_out_parallel = -sqrt(abs(1.0 - r_out_perp.length_squared() )) * n;
	return r_out_perp + r_out_parallel;
}


