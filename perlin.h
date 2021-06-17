#pragma once

#include "common.h"

//#include <algorithm>
//#include <numeric>

class perlin {
	static const int point_count = 256;
	vec3* ranvec;		//array to hold the noise 
				//was original a float array, now vec3 to help with smoothing the noise
	int* perm_x;		//permutations for the x-direction
	int* perm_y;
	int* perm_z;

	__device__ static int* perlin_generate_perm(curandState *s) {	//creates a permutation of the numbers from 0 to point_count
		auto p = new int[point_count];

		for (int i = 0; i < point_count; i++)
			p[i] = i;

		permute(s, p, point_count);

		return p;
	}

	__host__ static int* perlin_generate_perm() {	//creates a permutation of the numbers from 0 to point_count
		auto p = new int[point_count];

		//for (int i = 0; i < point_count; i++)
		//	p[i] = i;
		std::iota(&p[0], &p[point_count], 0);

		permute(p, point_count);

		return p;
	}

	__device__ static void permute(curandState *s, int* p, const int n) {	//creates a permutatoin of an integer array of length n 
		for (int i = n-1; i>0; i--) {	//switches p[i] with some p[n] for n<i
			const int target = random_int(s, 0,i);
			const int temp = p[i];
			p[i] = p[target];
			p[target] = temp;
		}
	}

	__host__ static void permute(int* p, const int n) {	//creates a permutation of an integer array of length n
		for (int i = n-1; i>0; i--) {	//switches p[i] with some p[n] for n<i
			const int target = random_int(0,i);
			std::swap(p[i], p[target]);
		}
	}


	__host__ __device__ static float perlin_interp(const vec3 c[2][2][2], const float u, const float v, const float w) {
		//first using a Hermite cubic to smooth the results
		const auto uu = u*u*(3-2*u);
		const auto vv = v*v*(3-2*v);
		const auto ww = w*w*(3-2*w);

		auto accum = 0.0f;

		for (int i=0; i<2; i++)
			for (int j=0; j<2; j++)
				for (int k=0; k<2; k++) {
					const vec3 weight_v(u-i, v-j, w-k);
					accum +=(i*uu + (1-i)*(1-uu)) *	//regual trilinar interpolation
						(j*vv + (1-j)*(1-vv)) *
						(k*ww + (1-k)*(1-ww)) *
						dot(c[i][j][k], weight_v);	//with a slight modification

				}

		return accum;
	}


	public:
	//__device__ perlin() {}

	__device__ perlin(curandState *s) {
		ranvec = new vec3[point_count];
		for (int i = 0; i < point_count; i++) {
			ranvec[i] = unit_vector(vec3::random(s, -1,1));
		}

		perm_x = new int[point_count];
		perm_y = new int[point_count];
		perm_z = new int[point_count];

		perm_x = perlin_generate_perm(s);
		perm_y = perlin_generate_perm(s);
		perm_z = perlin_generate_perm(s);
	}

	__host__ __device__ ~perlin() {
		delete [] ranvec;
		delete [] perm_x;
		delete [] perm_y;
		delete [] perm_z;
	}

	__host__ perlin() {
		ranvec = new vec3[point_count];
		for (int i = 0; i < point_count; i++) {
			ranvec[i] = unit_vector(vec3::random(-1,1));
		}

		perm_x = perlin_generate_perm();
		perm_y = perlin_generate_perm();
		perm_z = perlin_generate_perm();
	}

	__host__ __device__ float noise(const point3& p) const {
		//scrambling (using a hash) the random numbers (all point_count of them) to remove tiling
		auto u = p.x() - floorf(p.x());	//the decimal part of p.x
		auto v = p.y() - floorf(p.y());
		auto w = p.z() - floorf(p.z());

		const auto i = static_cast<int>(floorf(p.x()));	//used in the scambling
		const auto j = static_cast<int>(floorf(p.y()));
		const auto k = static_cast<int>(floorf(p.z()));
		
		vec3 c[2][2][2];


		//smothing out the result using linear interplation
		for (int di=0; di<2; di++) {
			for (int dj=0; dj<2; dj++) {
				for (int dk=0; dk<2; dk++) {
					c[di][dj][dk] = ranvec[	//the scrambling for the current point and the a points 1 and 2 integer steps in each direction
						perm_x[(i+di)&255] ^	// - other points are required for the linear interpolation
						perm_y[(j+dj)&255] ^
						perm_z[(k+dk)&255]
					];
				}
			}
		}

		return perlin_interp(c, u, v, w);	//the actual linear interpolation
	}

	__host__ __device__ float turb(const point3& p, const int depth=7) const {
		auto accum = 0.0;
		auto temp_p = p;
		auto weight = 1.0;

		for (int i = 0; i < depth; i++) {
			accum += weight * noise(temp_p);	//the actual noise
			weight *= 0.5;				//progressive additions of noise have less impact overall
			temp_p *= 2;				//so the noise is not all at the same place
		}

		return abs(accum);
	}
};
