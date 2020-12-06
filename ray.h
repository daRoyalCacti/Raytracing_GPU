#pragma once

#include "vec3.h"

struct ray {
	point3 orig;
	vec3 dir;
	double tm;	//time the ray exists at
	
	__device__ ray() {}
	__device__ ray(const point3& origin, const vec3& direction, const double time = 0.0) : orig(origin), dir(direction), tm(time) {}

	__device__ inline point3 origin() const {return orig;}
	__device__ inline vec3 direction() const {return dir;}
	__device__ inline double time() const {return tm;}

	__device__ point3 at(const double t) const {
		return orig + t*dir;
	}
};
