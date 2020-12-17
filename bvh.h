#pragma once

#include "common.h"

#include "hittable.h"
#include "hittable_list.h"
#include "box.h"

struct node_info {
	bool end;
	int num, left, right, parent;
};

#define cpu

int num_bvh_nodes(int n) {
	return pow(2, ceil(log2(n))) -1 +n;
}

struct bvh_nodez : public hittable {
#ifndef cpu
	hittable* hittables;
#endif
	node_info* info;
	int n;	//number of objects associated to the tree

	//__device__ bvh_node();
#ifdef cpu
	bvh_nodez(int num_obj);
#else
	__device__ bvh_nodez(int num_obj);
#endif

#ifdef cpu
	int num_nodes(){
		return pow(2, ceil(log2(n))) -1 +n;
	}
#else
	__device__ int num_nodes(){
		return static_cast<int>(powf(2, ceilf(log2f(n)) ) -1 +n);
	}
#endif

#ifdef cpu
	int index_at(int row, int col) {
		return pow(2, row) - 1 + col;	
	}
#else
	__device__ int index_at(int row, int col) {
		return powf(2,row) - 1 + col;	
	}
#endif

#ifndef cpu
	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;
#endif
};




#ifdef cpu
 bvh_nodez::bvh_nodez(int num_obj) : n(num_obj) {
	 const int num_ne_rows = ceil(log2(n));
	 info = new node_info[num_nodes()];
#else
__device__ bvh_nodez::bvh_nodez(int num_obj) : n(num_obj) {
	const int num_ne_rows = ceilf(log2f(n));
#endif
	
	int index;
	int k;
	//for tree nodes that are above the last 2 rows
	bool first = true, left = true, right = false;
	for (int row = 0; row < num_ne_rows -1; row++) {
#ifdef cpu
		for (int col = 0; col < pow(2, row); col++) {
#else
		for (int col = 0; col < powf(2, row); col++) {
#endif
			index = index_at(row, col);
			if (first) {
				first = false;
				info[0].num = n;
				info[0].left = 1;
				info[0].right = 2;
				info[0].parent = -1;	//no parent
				info[0].end = false;
				info[info[0].left].parent = 0;
				info[info[0].right].parent = 0;
			} else {
#ifdef cpu
				k = floor( info[ info[index].parent].num / 2.0f  );
#else
				k = floorf(info[ info[index].parent].num / 2.0f  );
#endif
				if (left) {
					left = false;
					right = true;

					info[index].num = k;
				} else if (right) {
					right = false;
					left = true;
					info[index].num = k + (info[info[index].parent].num%2);
				}

				info[index].left = index_at(row + 1, 2*col);
				info[index].right = index_at(row + 1, 2*col + 1);
				info[index].end = false;

				info[ info[index].left ].parent = index;
				info[ info[index].right].parent = index;
			}
		}
	}

	//for second to last row and last row
	int row = num_ne_rows-1;	//taking the last non end row
	left = true;	//should always be true
	right = false;	// -- just making sure
#ifdef cpu
	int counter = pow(2, num_ne_rows) - 1;	//total number of non-end nodes -- starting on the end row
#else 
	int counter = powf(2, num_ne_rows) -1;	//total number of non-end nodes -- starting on the end row
#endif

#ifdef cpu
	for (int col = 0; col < pow(2, row); col++) {
#else
	for (int col = 0; col < powf(2, row); col++) {
#endif
		index = index_at(row, col);
#ifdef cpu
		k = floor( info[ info[index].parent].num / 2.0f  );
#else
		k = floorf(info[ info[index].parent].num / 2.0f  );
#endif
		if (left) {
			left = false;
			right = true;
			info[index].num = k; 
		} else if (right) {
			right = false;
			left = true;
			info[index].num = k + (info[info[index].parent].num%2);
		}
		info[index].end = false;
		if (info[index].num == 1) {	//1 object
			info[index].left = counter;
			info[index].right = counter;
		} else if (info[index].num==2) {
			info[index].left = counter;
			info[index].right = ++counter;
		} else {
			std::cerr << "not second last row: " << index << std::endl;
		}
		counter++;

		info[ info[index].left].parent = index;
		info[ info[index].right].parent =index;

		//switching now to last row
		int old_index = index;
		index = info[old_index].left;
#ifdef cpu
		k = floor( info[ info[index].parent].num / 2.0f  );
#else
		k = floorf(info[ info[index].parent].num / 2.0f  );
#endif
		if (k+info[info[index].parent].num%2 != 1) {
			std::cerr << "last row does not contain 1 object: " <<index << std::endl;
		}
		
		info[index].num = 1;
		info[index].end = true;
		info[index].left = -1;	//no children
		info[index].right = -1;

		if (index != info[old_index].right) {	//for when a second last row node is connected to 2 last row nodes
			index = info[old_index].right;
#ifdef cpu
			k = floor( info[ info[index].parent].num / 2.0f  );
#else
			k = floorf(info[ info[index].parent].num / 2.0f  );
#endif
			if (k != 1) {
				std::cerr << "last row does not contain 1 object" << std::endl;
			}

			info[index].num = 1;
			info[index].end = true;
			info[index].left = -1;	//no children
			info[index].right = -1;
		}
	}
}


#ifndef cpu
__device__ bool bvh_nodez::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	return false;
}
__device__ bool bvh_nodez::bounding_box(const float time0, const float time1, aabb& output_box) const {
	return false;
}

#endif
