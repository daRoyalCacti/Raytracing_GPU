//is a tree of bounding boxes
#pragma once

//#include <algorithm>

#include "common.h"

#include <functional>

#include "hittable.h"
#include "hittable_list.h"
#include "box.h"

struct bvh_node : public hittable {
	hittable *left;	//left and right nodes on the tree
	hittable *right;
	aabb box;			//the box for the current node

	__device__ bvh_node();
	__device__ bvh_node(hittable_list& list,  const float time0, const float time1, curandState *s) {
		bvh_node(list.objects, 0, list.list_size, time0, time1, s);
	}
	__device__ bvh_node(hittable** src_objects, const size_t start, const size_t end, const float time0, const float time1, curandState *s);

	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;
};

__device__ bool bvh_node::bounding_box(const float time0, const float time1, aabb& output_box) const {
	output_box = box;
	return true;
}

__device__ bool bvh_node::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	if (!box.hit(r, t_min, t_max)) return false;	//if it didn't hit the large bounding box

	const bool hit_left = left->hit(r, t_min, t_max, rec, s);	//did the ray hit the left hittable
	const bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec, s);	//did the ray hit the right hittable
											//if the ray hit the left, check to make sure it hit the right before the left
											// - so rec is set correctly
	return hit_left || hit_right;
}

__device__ inline bool box_compare(const hittable* a, const hittable *b, const int axis) {
	aabb box_a;
	aabb box_b;

	if (!a->bounding_box(0,0,box_a) || !b->bounding_box(0,0,box_b))
		return false;	//cannot cerr on GPU
		//std::cerr << "No bouning box in bvh_node constructor.\n";

	return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ bool box_x_compare (const hittable* a, const hittable* b) {
	return box_compare(a, b, 0);
}


__device__ inline bool box_y_compare (const hittable *a, const hittable *b) {
	return box_compare(a, b, 1);
}


__device__ inline bool box_z_compare (const hittable *a, const hittable *b) {
	return box_compare(a, b, 2);
}



//__device__ void d_sort(hittable** arr, int low, int high, std::function<__device__ bool (const hittable*, const hittable*)>) {
//
//}


__device__ bvh_node::bvh_node(hittable** src_objects,  const size_t start, const size_t end, const float time0, const float time1, curandState *s) {
	auto objects = src_objects;	//Create a modifiable array for the sorce scene objects

	const auto axis = random_int(s,0,2);
	const auto comparator =   (axis==0) ? box_x_compare	//used to sort boxes into close and far to a given axis
				: (axis==1) ? box_y_compare
					    : box_z_compare;
	
	const size_t object_span = end - start;		//the number of objects the node is conned to

	if (object_span == 1) {		//only 1 object on node
		//put the object in both left and right
		left = right = objects[start];
	} else if (object_span == 2) {	//2 objects on node

		if (comparator(objects[start], objects[start+1])) {	//if the first object is closer to the random axis than the second object
			left = objects[start];
			right = objects[start+1];
		} else {
			left = objects[start+1];
			right = objects[start];
		}

	} else {
		//std::sort(objects[0] + start, objects[0] + end, comparator);
		//d_sort(objects, start, end, comparator);
		comparator(objects[0]+start, objects[0]+end);

		const auto mid = start + object_span /2;
		left = new bvh_node(objects, start, mid, time0, time1, s);
		right = new bvh_node(objects, mid, end, time0, time1, s);
	}

	aabb box_left, box_right;
	
	if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right) )
		return;	//cannot cerr on GPU
		//std::cerr << "No bounding box in bvh_node constructor.\n";
	
	box = surrounding_box(box_left, box_right);
}



