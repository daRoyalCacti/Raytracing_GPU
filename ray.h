#pragma once

#include "vec3.h"

struct ray {
	point3 orig;
	vec3 dir;
	float tm;	//time the ray exists at
	
	__device__ ray() {}
	__device__ ray(point3 origin, vec3 direction, const float time = 0.0) : orig(origin), dir(direction), tm(time) {}

	__device__ inline point3 origin() const {return orig;}
	__device__ inline vec3 direction() const {return dir;}
	__device__ inline float time() const {return tm;}

	__device__ point3 at(const float t) const {
		return orig + t*dir;
	}
};
