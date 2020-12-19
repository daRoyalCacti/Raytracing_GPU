#pragma once

#include "common.h"

#include "hittable.h"
#include "hittable_list.h"
#include "box.h"

#define newa

struct node_info {
	bool end;
	int num, left, right, parent;
	int* ids;	//ids of the objs that a node bounds
};


int num_bvh_nodes(int n) {
	return pow(2, ceil(log2(n))) -1 +n;
}


struct hittable_id {
	hittable* obj;
	int index;
};


//helper function to be used in merge sort
__device__ hittable_id* merge(hittable_id* A, int L1, int R1, int L2, int R2, int axis) {
	//A array to be sorted
	//L1 the start of the first part
	//R1 the end of the first part
	//L2 the start of the second part
	//R2 the end of the second part
	int index = 0;
#ifdef newa
	hittable_id* temp = new hittable_id[R1 - L1 + R2 - L2+1];
#endif
	aabb box1, box2;
	
	while (L1 <= R1 && L2 <= R2) {
		A[L1].obj->bounding_box(0, 0, box1);
		A[L2].obj->bounding_box(0, 0, box2);
		//if (A[L1] <= A[L2]) {
		if (box1.min().e[axis] <= box2.min().e[axis]) {
			temp[index] = A[L1];
			index++;
			L1++;
		} else {
			temp[index] = A[L2];
			index++;
			L2++;
		}
	}

	while (L1 <= R1) {	//if L2 <= R2 becomes false, L1 <= R1 does not necessarily have to be false
		temp[index] = A[L1];
		index++;
		L1++;
	}

	while (L2 <= R2) {
		temp[index] = A[L2];
		index++;
		L2++;
	}

	return temp;	
}



__device__ void merge_sort(hittable** A, int n, int* O, int axis) {
	//A input array
	//n size of input array
	//O output array

#ifdef newa	
	hittable_id* Out = new hittable_id[n];
#endif
	
	//might need ===================================
	for (int k = 0; k < n; k++) { 
		Out[k].obj = A[k];
		Out[k].index = k;
	}
	//=============================================


	int len = 1;
	int i, L1, R1, L2, R2;
	
	while (len < n) {
		//std::cout << "len = " << len << std::endl;
		i = 0;
		while (i < n) {
			//std::cout << "i = " << i << std::endl;
			L1 = i;
			R1 = i + len-1;
			L2 = i + len;
			R2 = i + 2*len - 1;

			if (L2 >= n) 
				break;

			if (R2 >= n)
				R2 = n-1;

			auto temp = merge(Out, L1, R1, L2, R2, axis);
			
			for (int j=0; j < R2-L1+1; j++) { 
				//std::cout << "i+j " << i+j << std::endl;
				Out[i+j] = temp[j];
			}

			i += 2*len;
		}
		len *= 2;
	}

	for (int i = 0; i < n; i++) {
		O[i] = Out[i].index;
	}
	
}


int size_of_bvh(int n) {
	//n is the number of objects
	//returns the approximate size of the object in bytes
	int current = 0;
	current += n * sizeof(hittable*);		//the raw objects
	current += (num_bvh_nodes(n) - n) * sizeof(aabb*);	//number of bounding boxes
	current += ceil(log2(n))*n * sizeof(int);	//ids of objects per node
	current += 3*n * sizeof(int);			//the sorted ids array
	current += num_bvh_nodes(n) * sizeof(node_info);	//the info stored per node
	return current;
}



struct bvh_node : public hittable {
	hittable** objs;	//the actual objects
	__device__ bvh_node(hittable** hits, int num_obj, const float time0, const float time1, curandState *s);
	node_info* info;
	int n;	//number of objects associated to the tree
	aabb* bounds;	//the bounding boxes for each node of the tree
	int* obj_s[3];	//the sorted indices of the objects based on an axis

	//__device__ bvh_node();
	__device__ bvh_node(int num_obj);

	__device__ int num_nodes() const {
		return static_cast<int>(powf(2, ceilf(log2f(n)) ) -1 +n);
	}



	__device__ int index_at(int row, int col) const {
		return powf(2,row) - 1 + col;	
	}


	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;
};





__device__ bvh_node::bvh_node(int num_obj) : n(num_obj) {
	const int num_ne_rows = ceilf(log2f(n));
#ifdef newa
	info = new node_info[num_nodes()];
#endif
	
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
			//std::cerr << "not second last row: " << index << std::endl;
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
			//std::cerr << "last row does not contain 1 object: " <<index << std::endl;
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
				//std::cerr << "last row does not contain 1 object" << std::endl;
			}

			info[index].num = 1;
			info[index].end = true;
			info[index].left = -1;	//no children
			info[index].right = -1;
		}
	}


}


__device__ bvh_node::bvh_node(hittable** hits, int  num_obj, const float time0, const float time1, curandState *s) : bvh_node(num_obj) {
	
#ifdef newa
	objs = new hittable*[num_obj];
#endif


	objs = hits;	//not sure if will work
	//first filling the sorted arrays
	
#ifdef newa
	obj_s[0] = new int[n];
	obj_s[1] = new int[n];
	obj_s[2] = new int[n];

	merge_sort(objs, n, obj_s[0], 0);
	merge_sort(objs, n, obj_s[1], 1);
	merge_sort(objs, n, obj_s[2], 2);
#endif

	//the first node contains all objects
#ifdef newa
	info[0].ids = new int[n];
#endif
	for (int i = 0; i < n; i++)
		info[0].ids[i] = i;

	//printf("\n%i %i\n", info[0].left, info[0].right);
#ifdef newa
	for (int node = 1; node < num_nodes(); node++)
		info[node].ids = new int[info[node].num];
#endif
	//filling the nodes with the info about the objects they bound
	for (int node = 0; node < num_nodes() - n; node++) {	//iterating through all nodes less the last row (each node modifies the ids of its children)
		int axis = random_int(s, 0, 2);			

		//setting the ids of the objects passed down to the children
		
		//first sorting the objects by the axis
		//	iterating throuhg the sorted array of objects in their sorted order
		//	the ids that appear first in the sorted array get added to the children nodes in order
		//	thus splitting the objects between the children based on their ordering along a given axis
		int counterl = 0;
		int counterr = 0;
		int counter = 0;
		//bool should_break;
		for (int i = 0; i < n; i++) {
			//should_break = false;
			for (int j = 0; j < info[node].num /*&& !should_break*/; j++) {
				if (obj_s[axis][i] == info[node].ids[j]) {
					//printf("\n\n%i\n%i\n\n", info[node].left, info[node].num);
					//printf("counter = %i\n", counter);
					if (counter < info[info[node].left].num) {
						//printf("left %i\n", info[node].left);
						//printf("node = %i, i = %i, j = %i, counter = %i, %i, %i\n", node, i, j, counter, info[info[node].left].num, info[node].ids[j]);
						info[info[node].left].ids[counterl++] = info[node].ids[j];
						//should_break = true;
						//break;
						counter++;
						goto test_goto_point;
					} else {
						//printf("right %i\n", info[node].right);						
						info[info[node].right].ids[counterr++] = info[node].ids[j];
						//should_break = true;
						//break;
						counter++;
						goto test_goto_point;
					}
					//printf("axis = %i, node = %i, obj = %i, info = %i, i = %i, j = %i, counterl = %i, counterr = %i, counter = %i\n", axis, node, obj_s[axis][i], info[node].ids[j], i, j, counterl, counterr, counter);


				}
			}
test_goto_point:
		}
	}	
	
	//creating the bounding boxes
#ifdef newa
	bounds = new aabb[num_nodes() - n];
#endif
	
	//first for the second last row
	const int num_ne_rows = ceilf(log2f(n));
	aabb temp_box1, temp_box2;
	for (int node = index_at(num_ne_rows - 1, 0); node < index_at(num_ne_rows,0); node++) {
		if (info[node].num == 1) {
			objs[info[info[node].left].ids[0]]->bounding_box(0, 0, bounds[node]);
			//printf("%f\n", objs[info[node].left.ids[0]]->radius);
			//printf("1, %i\n", info[info[node].left].ids[0]);
		} else {
			objs[info[info[node].left].ids[0]]->bounding_box(0,0, temp_box1);
			objs[info[info[node].right].ids[0]]->bounding_box(0,0, temp_box2);
			bounds[node] = surrounding_box(temp_box1, temp_box2);
			//printf("2, %i %i\n", info[info[node].left].ids[0], info[info[node].right].ids[0]);
		}
		printf("\n (%f, %f, %f),   (%f, %f, %f)\n", bounds[node].min().x(), bounds[node].min().y(), bounds[node].min().z(), bounds[node].max().x(), bounds[node].max().y(), bounds[node].max().z() );
	}

	//for the rest of the rows
	for (int node = index_at(num_ne_rows - 2, 0); node <= 0; node++) {	//running through the rest of the nodes backwards
		bounds[node] = surrounding_box(bounds[info[node].left], bounds[info[node].right]);
	}

	//const int index =index_at(num_ne_rows - 2, 0);
	//printf("%f, %f\n", bounds[index].min().x(), bounds[index].max().x() );

}

__device__ bool bvh_node::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	
#ifdef newa
	int* lr = new int[ceilf(log2f(n))];	//number of non end rows
#else
	int* lr;
#endif

	
	int row_cntr = 0;	//counter for the current row
#ifdef newa
	int* col_cntr = new int[ceilf(log2f(n))];	//counts the current column for each row
#else
	int* col_cntr;
#endif
	bool once_hit = false;

	for (int i = 0; i < sizeof(col_cntr)/sizeof(int); i++) {
		col_cntr[i] = 0;
		lr[i] = 0;
	}

	bool end = false;

	int current_index;

	float current_hit_time = infinity; 
	hit_record temp_rec;

	
	while (!end) {
		current_index = index_at(row_cntr, col_cntr[row_cntr]);
		//printf("%i\n", current_index);

		if (info[current_index].end) {	//if at an end node
			if (info[info[current_index].parent].num == 1) {
				if (objs[info[current_index].ids[0]]->hit(r, t_min, t_max, temp_rec, s) ) {	//if an end node is hit
					if (temp_rec.t < current_hit_time) {
						current_hit_time = temp_rec.t;
						rec = temp_rec;
						once_hit = true;
					} else {	//there was a hit but it was after the current hit
						row_cntr--;		//move up 
						col_cntr[row_cntr]++;	//and across
					}
				} else {	//there was no hit
					row_cntr--;		//move up
					col_cntr[row_cntr]++;	//and across
				}
			} else {	//currently checking an end node that has 2 objects attacted to the parent node
				if (objs[info[current_index].ids[0]]->hit(r, t_min, t_max, temp_rec, s) ) {
					if (temp_rec.t < current_hit_time) {
						current_hit_time = temp_rec.t;
						rec = temp_rec;
						once_hit = true;
					} 
				}
				if (objs[info[current_index].ids[1]]->hit(r, t_min, t_max, temp_rec, s) ) {
					if (temp_rec.t < current_hit_time) {
						current_hit_time = temp_rec.t;
						rec = temp_rec;
						once_hit = true;
					} else {
						row_cntr--;		//move up
						col_cntr[row_cntr]++;	//and across
					}
				}

			}
		} else {	//not an end node
			if (bounds[current_index].hit(r, t_min, t_max) ) {	//ray hit a bounding box
				printf("a hit!\n");
				row_cntr++;	//move down a row
				col_cntr[row_cntr] = info[current_index].left;
			} else { 	//ray did not hit a bounding box
				if (lr[row_cntr] == 0) {	//at the first child node
					lr[row_cntr] = 1;	
					col_cntr[row_cntr]++;	//move to the second child node
				} else {	//at the second child node
					lr[row_cntr] = 0;	//resetting the variable
					while (lr[--row_cntr] == 1) {//keep moving up until get back to a left node
						lr[row_cntr] = 0;	//resetting the variable
					}						
					col_cntr[row_cntr]++;
				}
			}
		}


		if (col_cntr[row_cntr] > powf(2, row_cntr)) {
			end = true;
		}
	}


#ifdef newa
	delete [] lr;
	delete [] col_cntr;
#endif
	//printf("%i\n", once_hit);
	if (once_hit) {
		return true;
	} else {
		return false;
	}




	//return false;
}
__device__ bool bvh_node::bounding_box(const float time0, const float time1, aabb& output_box) const {
	output_box = bounds[0];
	return true;
}

