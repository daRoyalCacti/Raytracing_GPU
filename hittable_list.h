#pragma once

#include "hittable.h"
//#include "aabb.h"


struct hittable_list : public hittable {
	hittable** objects;
	int list_size;

	__device__ hittable_list() {}
	__device__ hittable_list(hittable **l, int n) :objects(l), list_size(n)  {}


	__device__ ~hittable_list() {
		delete objects;
	}

	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;
};

__device__ bool hittable_list::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;
	
	//testing to see if ray hits anyobject between the given times
	for (int i = 0; i < list_size; i++) {	//iterating through all objects that could be hit
		if (objects[i]->hit(r, t_min, closest_so_far, temp_rec, s)) {//checking to see if the ray hit the object
			//printf("%i/%i\n", i, list_size-1);
			hit_anything = true;
			closest_so_far = temp_rec.t;	//keep lowering max time to find closest object
			rec = temp_rec;
		}
	}

	return hit_anything;
}


__device__ bool hittable_list::bounding_box(const float time0, const float time1, aabb& output_box) const {
	if (list_size == 0) return false;	//no objects to create bounding boxes for

	aabb temp_box;
	bool first_box = true;

	for (int i = 0; i < list_size; i++) {
		if (!objects[i]->bounding_box(time0, time1, temp_box)) return false;	//if bounding box returns false,  this function returns false
											//this also makes temp_box the bounding box for object
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box); 	//creates a bounding box around the previous large bounding box and 
												// the bounding box for the current object
												//If there is no previous large bounding box, output box is just the bounding box
												// for the object
		first_box = false;
	}
	
	return true;
}
