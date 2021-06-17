#pragma once

#include "common.h"

#include "hittable.h"
#include "material.h"
#include "texture.h"

struct constant_medium : public hittable {
	hittable *boundary;
	material *phase_function;
	float neg_inv_density;		//required to move info from constructor to hit
	
	//d for density
	constant_medium(hittable *b, const float d, texturez *a) 
		: boundary(b), neg_inv_density(-1/d) {
		phase_function = new isotropic(a);
		};

	constant_medium(hittable *b, const float d, color c)
		: boundary(b), neg_inv_density(-1/d) {
		phase_function = new isotropic(c);
		};

	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const override;

	virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		return boundary->bounding_box(time0, time1, output_box);
	}
};


//algorithm only works for convex shape
__device__ bool constant_medium::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const {
	hit_record rec1, rec2;
	
	//if the ray ever hits the boundary
	if (!boundary->hit(r, -infinity, infinity, rec1, s))
		return false;

	//if the ray hits the boundary after every hitting the boundary (i.e. the time found above)
	// - so the ray is on a trajectory that will enter the medium and exit it
	if (!boundary->hit(r, rec1.t+0.0001f, infinity, rec2, s))
		return false;

	if (rec1.t < t_min) rec1.t = t_min;	//if the entry collision happens before the min time
	if (rec2.t > t_max) rec2.t = t_max;	//if the exit collision happens after the max time

	if (rec1.t >= rec2.t)	//if the entry happens after or at the same time as the exit
		return false;

	if (rec1.t < 0)
		rec1.t = 0;

	const auto ray_length = r.direction().length();
	const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
	const auto hit_distance = neg_inv_density * log(random_float(s));	//randomly deciding if the ray should leave the medium

	if (hit_distance > distance_inside_boundary)	//if the randomly chosen distance is greater than the distance from the boudnary to the ray origin
		return false;

	rec.t = rec1.t + hit_distance / ray_length;	//t_f = t_i + d/s
	rec.p = r.at(rec.t);

	rec.normal = vec3(1,0,0);	//arbitrary
	rec.front_face = true;		//arbitrary
	rec.mat_ptr = phase_function;

	return true;
}
