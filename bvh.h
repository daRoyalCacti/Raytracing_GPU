#pragma once

#include "common.h"

#include "hittable.h"
#include "hittable_list.h"
#include "box.h"

struct node_info {
	bool end;
	int num, left, right, parent;
};


int num_bvh_nodes(int n) {
	return pow(2, ceil(log2(n))) -1 +n;
}




//helper function to be used in merge sort
__device__ hittable* merge(const hittable* A, int L1, int R1, int L2, int R2, int axis) {
	//A array to be sorted
	//L1 the start of the first part
	//R1 the end of the first part
	//L2 the start of the second part
	//R2 the end of the second part
	int index = 0;
	hittable* temp = new hittable[R1 - L1 + R2 - L2+1];
	aabb box1, box2;
	
	while (L1 <= R1 && L2 <= R2) {
		A[L1].bounding_box(0, 0, box1);
		A[L2].bounding_box(0, 0, box2);
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



__device__ void merge_sort(const hittable* A, int n, hittable* O, int axis) {
	//A input array
	//n size of input array
	//O output array
	
	//might need ===================================
	/*for (int k = 0; k < n; k++) { 
		O[k] = A[k];
	}*/
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

			auto temp = merge(O, L1, R1, L2, R2, axis);
			
			for (int j=0; j < R2-L1+1; j++) { 
				//std::cout << "i+j " << i+j << std::endl;
				O[i+j] = temp[j];
			}

			i += 2*len;
		}
		len *= 2;
	}
}


struct bvh_node : public hittable {
	hittable* hittables;
	__device__ bvh_node(hittable* hits, int num_obj, const float time0, const float time1, curandState *s);
	node_info* info;
	int n;	//number of objects associated to the tree

	//__device__ bvh_node();
	__device__ bvh_node(int num_obj);

	__device__ int num_nodes(){
		return static_cast<int>(powf(2, ceilf(log2f(n)) ) -1 +n);
	}



	__device__ int index_at(int row, int col) {
		return powf(2,row) - 1 + col;	
	}


	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;
};





__device__ bvh_node::bvh_node(int num_obj) : n(num_obj) {
	const int num_ne_rows = ceilf(log2f(n));
	
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


__device__ bvh_node::bvh_node(hittable* hits, int  num_obj, const float time0, const float time1, curandState *s) : bvh_node(num_obj) {

}

__device__ bool bvh_node::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	return false;
}
__device__ bool bvh_node::bounding_box(const float time0, const float time1, aabb& output_box) const {
	return false;
}

