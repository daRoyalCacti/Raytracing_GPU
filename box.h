#pragma once

#include "common.h"

#include "aarect.h"
#include "hittable_list.h"

struct box : public hittable {
	point3 box_min;		//min and max define the corners of the box
	point3 box_max;
	hittable_list sides;

	__host__ box() {sides.alloc(6);}
	//__host__ box(const point3& p0, const point3& p1, material *ptr);

	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const override {
		return sides.hit(r, t_min, t_max, rec, s);
	}

	__device__ void create(const point3& p0, const point3& p1, material *ptr);

	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		output_box = aabb(box_min, box_max);
		return true;
	}
};

//__host__ box::box(const point3& p0, const point3& p1, material *ptr) : box_min(p0), box_max(p1) {
//	sides.alloc(6);
//}


__device__ void box::create(const point3& p0, const point3& p1, material *ptr) {
	sides.add(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
	sides.add(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

	sides.add(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
	sides.add(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));
	
	sides.add(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
	sides.add(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}
