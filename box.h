#pragma once

#include "common.h"

#include "aarect.h"
#include "hittable_list.h"

struct box : public hittable {
	point3 box_min;		//min and max define the corners of the box
	point3 box_max;
	hittable_list *sides;
	hittable ** temp_hittable;	//for constructing the box

	box(point3 p0, point3 p1, material *ptr) : box_min(p0), box_max(p1) {
		temp_hittable = new hittable*[6];
		
		temp_hittable[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
		temp_hittable[1] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

		temp_hittable[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
		temp_hittable[3] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);
		
		temp_hittable[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
		temp_hittable[5] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);

		sides = new hittable_list(temp_hittable, 6);
	}

	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const override {
		//printf("box hit start\n");
		return sides->hit(r, t_min, t_max, rec, s);
	}

	void create(const point3& p0, const point3& p1, material *ptr);

	virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		output_box = aabb(box_min, box_max);
		return true;
	}
};

