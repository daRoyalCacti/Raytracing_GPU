
#pragma once

#include "common.h"

#include "hittable.h"
#include "hittable_list.h"
#include "box.h"

struct node_info {
	bool end;
	int num, left, right, parent;
	int* ids;	//ids of the objs that a node bounds

	node_info() = default;
	__host__ __device__ node_info(const bool end_, const int num_, const int left_, const int right_, const int parent_, int* ids_) {
		end = end_;
		num = num_;
		left = left_;
		right = right_;
		parent = parent_;
		ids = ids_;
	}
};


int num_bvh_nodes(int n) {
	return pow(2, floor(log2(n))+1) -1 +n;
}
__device__ int num_bvh_nodes_d(int n) {
	return pow(2, floor(log2f(n))+1) -1 +n;
}


struct hittable_id {
	hittable* obj;
	int index;
};


unsigned size_of_bvh(int n) {
	//n is the number of objects
	//returns the approximate size of the object in bytes
	unsigned current = 0;
	current += n * sizeof(hittable*);		//the raw objects
	current += (num_bvh_nodes(n) - n) * sizeof(aabb*);	//number of bounding boxes
	current += ceil(log2(n))*n * sizeof(int);	//ids of objects per node
	current += 3*n * sizeof(int);			//the sorted ids array
	current += num_bvh_nodes(n) * sizeof(node_info);	//the info stored per node
	return current;
}


//U must be a hittable
template <typename U> 
struct bvh_node : public hittable {
	U** objs;	//the actual objects
	bvh_node(){}
	bvh_node(U** &hits, int num_obj, const float time0, const float time1);
	node_info* info;
	int n;	//number of objects associated to the tree
	aabb* bounds;	//the bounding boxes for each node of the tree
	int* obj_s[3];	//the sorted indices of the objects based on an axis

	bvh_node(int num_obj);

	int num_nodes() const {
		return static_cast<int>(powf(2, floor(log2f(n)+1) ) -1 +n);
	}


	__device__ __host__ bvh_node(U** objs_, node_info* info_, const int n_, aabb* bounds_, int* obj_s0, int* obj_s1, int* obj_s2) {
		objs = objs_;
		info = info_;
		n = n_;
		bounds = bounds_;
		obj_s[0] = obj_s0;
		obj_s[1] = obj_s1;
		obj_s[2] = obj_s2;
	}

	__host__ __device__ int index_at(int row, int col) const {
		return powf(2,row) - 1 + col;	
	}


	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;

	template <typename T>
	void cpy_constit_d(T* d_ptr) const override {
		int* objs0_d, *objs1_d, *objs2_d;
		aabb* bounds_d;
		U** objects_dg;
		U** objects_d = new U*[n];
		node_info* info_d;
		int** ids_d = new int*[num_bvh_nodes(n)]; 	//only storing the first value of the host array


		cudaMalloc((void**)&objs0_d, n * sizeof(int));
		cudaMalloc((void**)&objs1_d, n * sizeof(int));
		cudaMalloc((void**)&objs2_d, n * sizeof(int));
		cudaMalloc((void**)&bounds_d, (num_bvh_nodes(n) - n) * sizeof(aabb) );
		//cudaMalloc((void**)&objects_d,n * sizeof(U*));
		cudaMalloc((void**)&objects_dg, n*sizeof(U*));
		for (size_t i = 0; i < n; i++) {
			//std::cout << i << "\n";
			cudaMalloc((void**)&objects_d[i], sizeof(U));
		}
		cudaMalloc((void**)&info_d, num_bvh_nodes(n) * sizeof(node_info));
		for (size_t i = 0; i < num_bvh_nodes(n); i++) {
			cudaMalloc((void**)&ids_d[i], sizeof(int));
		}


		//moving obj_s[0] inside the bvh_node
		checkCudaErrors(cudaMemcpy(objs0_d, obj_s[0], n*sizeof(int), cudaMemcpyHostToDevice));
		//moving obj_s[1] inside the bvh_node
		checkCudaErrors(cudaMemcpy(objs1_d, obj_s[1], n*sizeof(int), cudaMemcpyHostToDevice));
		//moving obj_s[2] inside the bvh_node
		checkCudaErrors(cudaMemcpy(objs2_d, obj_s[2], n*sizeof(int), cudaMemcpyHostToDevice));
		//moving bounds inside the bvh_node
		checkCudaErrors(cudaMemcpy(bounds_d, bounds, (num_bvh_nodes(n) - n) * sizeof(aabb) , cudaMemcpyHostToDevice));
		//the actual triangle objects in the bvh_node
		//checkCudaErrors(cudaMemcpy(objects_d, objs, n * sizeof(U*) , cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(objects_dg, objs, n * sizeof(U**) , cudaMemcpyHostToDevice));
		for (size_t i = 0; i < n; i++) {
			//std::cout << i << "/" << n << "\n";
				checkCudaErrors(cudaMemcpy(objects_d[i], objs[i], sizeof(U) , cudaMemcpyHostToDevice));
		}
		//node info for bvh_node
		checkCudaErrors(cudaMemcpy(info_d, info, num_bvh_nodes(n) * sizeof(node_info) , cudaMemcpyHostToDevice));
			//ids for the node info
			for (size_t i = 0; i < num_bvh_nodes(n); i++) {
				checkCudaErrors(cudaMemcpy(ids_d[i], info[i].ids, sizeof(int) , cudaMemcpyHostToDevice));
			}

		checkCudaErrors(cudaMemcpy(&(d_ptr->obj_s[0]), &(objs0_d), sizeof(int*),   cudaMemcpyDefault)); 	
		checkCudaErrors(cudaMemcpy(&(d_ptr->obj_s[1]), &(objs1_d), sizeof(int*),   cudaMemcpyDefault)); 	
		checkCudaErrors(cudaMemcpy(&(d_ptr->obj_s[2]), &(objs2_d), sizeof(int*),   cudaMemcpyDefault)); 	
		checkCudaErrors(cudaMemcpy(&(d_ptr->bounds), &bounds_d, sizeof(aabb*), cudaMemcpyDefault)); 
		//checkCudaErrors(cudaMemcpy(&(d_ptr->objs), &objects_d, sizeof(U**), cudaMemcpyDefault)); 
		checkCudaErrors(cudaMemcpy(&(d_ptr->objs), &objects_dg, sizeof(U**), cudaMemcpyDefault)); 
		for (size_t i = 0; i < n; i++) {
			//std::cout << i << "\n";
			//std::cout << &(d_ptr->objs[i]) << "\n";
			//checkCudaErrors(cudaMemcpy(&(d_ptr->objs[i]), &objects_d[i], sizeof(int*), cudaMemcpyDefault)); 
			checkCudaErrors(cudaMemcpy(&(objects_dg[i]), &objects_d[i], sizeof(int*), cudaMemcpyDefault)); 
		}
		checkCudaErrors(cudaMemcpy(&(d_ptr->info), &info_d, sizeof(node_info*), cudaMemcpyDefault)); 
		for (size_t i = 0; i < num_bvh_nodes(n); i++) {
			checkCudaErrors(cudaMemcpy(&(info_d[i].ids), &ids_d[i], sizeof(int*), cudaMemcpyDefault)); 
		}


		
		for (size_t i = 0; i < n; i++) {
			//std::cout << objs[i] << std::endl;
			//std::cout << objects_d[i] << std::endl;
			objs[i]->cpy_constit_d(objects_d[i]); 	
		}

	}
};




template <typename U>
bvh_node<U>::bvh_node(int num_obj) : n(num_obj) {
	const int num_ne_rows = ceilf(log2f(n));
	info = new node_info[num_nodes()];
	
	int index;
	int k;
	//for tree nodes that are above the last 2 rows
	bool first = true, left = true, right = false;
	for (int row = 0; row < num_ne_rows -1; row++) {
		for (int col = 0; col < powf(2, row); col++) {
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
				k = floorf(info[ info[index].parent].num / 2.0f  );
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
	int counter = powf(2, num_ne_rows) -1;	//total number of non-end nodes -- starting on the end row

	for (int col = 0; col < powf(2, row); col++) {
		index = index_at(row, col);
		k = floorf(info[ info[index].parent].num / 2.0f  );
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
			printf("not second last row: %i\n", index);
		}
		counter++;

		info[ info[index].left].parent = index;
		info[ info[index].right].parent =index;

		//switching now to last row
		int old_index = index;
		index = info[old_index].left;
		k = floorf(info[ info[index].parent].num / 2.0f  );
		if (k+info[info[index].parent].num%2 != 1) {
			printf("last row does not contain 1 object: %i\n", index);
		}
		
		info[index].num = 1;
		info[index].end = true;
		info[index].left = -1;	//no children
		info[index].right = -1;

		if (index != info[old_index].right) {	//for when a second last row node is connected to 2 last row nodes
			index = info[old_index].right;
			k = floorf(info[ info[index].parent].num / 2.0f  );
			if (k != 1) {
				printf("last row does not contain 1 object: %i\n", index);
			}

			info[index].num = 1;
			info[index].end = true;
			info[index].left = -1;	//no children
			info[index].right = -1;
		}
	}


}


inline bool box_compare(hittable_id &a, hittable_id &b, const int axis) {
	aabb box_a;
	aabb box_b;

	if (!a.obj->bounding_box(0,0,box_a) || !b.obj->bounding_box(0,0,box_b))
		std::cerr << "No bounding box in bvh_node constructor.\n";

	return box_a.min().e[axis] < box_b.min().e[axis];
}

bool box_x_compare (hittable_id &a, hittable_id &b) {
	return box_compare(a, b, 0);
}


inline bool box_y_compare (hittable_id &a, hittable_id &b) {
	return box_compare(a, b, 1);
}


inline bool box_z_compare (hittable_id &a, hittable_id &b) {
	return box_compare(a, b, 2);
}

template <typename U>
bvh_node<U>::bvh_node(U** &hits, int  num_obj, const float time0, const float time1) : bvh_node<U>(num_obj) {

	objs = hits;
	
	
	//first filling the sorted arrays
	obj_s[0] = new int[n];
	obj_s[1] = new int[n];
	obj_s[2] = new int[n];


	//should be updated to use std::sort
	/*merge_sort(objs, n, obj_s[0], 0);
	merge_sort(objs, n, obj_s[1], 1);
	merge_sort(objs, n, obj_s[2], 2); */
	hittable_id* temp_objs = new hittable_id[n];
	for (unsigned k = 0; k < n; k++) {
		temp_objs[k].obj = objs[k];
		temp_objs[k].index = k;
	}

	hittable_id* temp_objs1 = temp_objs;
	hittable_id* temp_objs2 = temp_objs;
	hittable_id* temp_objs3 = temp_objs;

	std::sort(temp_objs1, temp_objs1+n, box_x_compare);
	std::sort(temp_objs2, temp_objs2+n, box_y_compare);
	std::sort(temp_objs3, temp_objs3+n, box_z_compare);

	for (unsigned i = 0; i  < n; i++) {
		obj_s[0][i] = temp_objs1[i].index;
		obj_s[1][i] = temp_objs2[i].index;
		obj_s[2][i] = temp_objs3[i].index;
	}



	//the first node contains all objects
	info[0].ids = new int[n];
	for (int i = 0; i < n; i++)
		info[0].ids[i] = i;
	for (int node = 1; node < num_nodes(); node++)
		info[node].ids = new int[info[node].num];
	//filling the nodes with the info about the objects they bound
	for (int node = 0; node < num_nodes() - n; node++) {	//iterating through all nodes less the last row (each node modifies the ids of its children)
		int axis = random_int(0, 2);			

		//setting the ids of the objects passed down to the children
		
		//first sorting the objects by the axis
		//	iterating throuhg the sorted array of objects in their sorted order
		//	the ids that appear first in the sorted array get added to the children nodes in order
		//	thus splitting the objects between the children based on their ordering along a given axis
		int counterl = 0;
		int counterr = 0;
		int counter = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < info[node].num; j++) {
				if (obj_s[axis][i] == info[node].ids[j]) {
					if (counter < info[info[node].left].num) {
						info[info[node].left].ids[counterl++] = info[node].ids[j];
						counter++;
						goto test_goto_point;
					} else {
						info[info[node].right].ids[counterr++] = info[node].ids[j];
						counter++;
						goto test_goto_point;
					}

				}
			}
test_goto_point:
		}
	}	
	
	//creating the bounding boxes
	bounds = new aabb[num_nodes() - n];
	
	//first for the second last row
	const int num_ne_rows = ceilf(log2f(n));
	aabb temp_box1, temp_box2;
	for (int node = index_at(num_ne_rows - 1, 0); node < index_at(num_ne_rows,0); node++) {
		if (info[node].num == 1) {
			objs[info[info[node].left].ids[0]]->bounding_box(time0, time1, bounds[node]);
		} else {
			objs[info[info[node].left].ids[0]]->bounding_box(time0,time1, temp_box1);
			objs[info[info[node].right].ids[0]]->bounding_box(time0, time1, temp_box2);
			bounds[node] = surrounding_box(temp_box1, temp_box2);
		}
	}

	//for the rest of the rows
	for (int node = index_at(num_ne_rows - 1, 0) - 1; node >= 0; node--) {	//running through the rest of the nodes backwards
		bounds[node] = surrounding_box(bounds[info[node].left], bounds[info[node].right]);
	}


}

template <typename U>
__device__ bool bvh_node<U>::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	//printf("vvv\n");

	
	int row_cntr = 0;	//counter for the current row
	int* col_cntr = new int[ceilf(log2f(n))];	//counts the current column for each row

	bool once_hit = false;

	for (int i = 0; i < sizeof(col_cntr)/sizeof(int); i++) {
		col_cntr[i] = 0;
	}

	bool end = false;

	int current_index;

	float current_hit_time = infinity; 
	hit_record temp_rec;

	
	while (!end) {
		//printf("adsfa\n");
		current_index = index_at(row_cntr, col_cntr[row_cntr]);

		if (info[current_index].end) {	//if at an end node
			if (info[info[current_index].parent].num == 1) {
				if (objs[info[current_index].ids[0]]->hit(r, t_min, t_max, temp_rec, s) ) {	//if an end node is hit
					if (temp_rec.t < current_hit_time) {
						current_hit_time = temp_rec.t;
						rec = temp_rec;
						once_hit = true;
					}
				}
			} else {	//currently checking an end node that has 2 objects attacted to the parent node
				if (objs[info[current_index].ids[0]]->hit(r, t_min, t_max, temp_rec, s) ) {
					if (temp_rec.t < current_hit_time) {
						current_hit_time = temp_rec.t;
						rec = temp_rec;
						once_hit = true;
					} 
				}
				if (objs[info[current_index+1].ids[0]]->hit(r, t_min, t_max, temp_rec, s) ) {
					if (temp_rec.t < current_hit_time) {
						current_hit_time = temp_rec.t;
						rec = temp_rec;
						once_hit = true;
					}
				}

			}

			//done checking end nodes
			// time to move back to regular nodes
			row_cntr--;	//moving up
			if (col_cntr[row_cntr] %2 == 0) {	//at the first child node
				col_cntr[row_cntr]++;	//move to the second child node
			} else {	//at the second child node
				while (col_cntr[--row_cntr]%2==1) {}	//keep moving up until get back to a left node			
				col_cntr[row_cntr]++;			//the move right 1
			}
		} else {	//not an end node
			if (bounds[current_index].hit(r, t_min, t_max) ) {	//ray hit a bounding box
				row_cntr++;	//move down a row
				col_cntr[row_cntr] = info[current_index].left - index_at(row_cntr, 0);
			} else { 	//ray did not hit a bounding box
				if (col_cntr[row_cntr]%2 == 0) { 	//at the first child node
					col_cntr[row_cntr]++;
				} else {	//at the second child node
					while(col_cntr[--row_cntr]%2 == 1) {}	//keep moving up until get back to a left node				
					col_cntr[row_cntr]++;			//then move right
				}
			}
		}


		if (col_cntr[row_cntr] >= powf(2, row_cntr)) {
			end = true;
		}
	}

	//printf("uuu\n");

	delete [] col_cntr;

	if (once_hit) {	//if at lest 1 object was hit
		return true;
	} else {
		return false;
	}
}

template <typename U>
bool bvh_node<U>::bounding_box(const float time0, const float time1, aabb& output_box) const {
	output_box = bounds[0];	//bounding box for the entire tree was already calculated
	return true;
}


