#pragma once

#include "hittable.h"
#include "hittable_list.h"
#include "camera.h"
#include "common.h"
#include "sphere.h"



__global__ void free_world(hittable ** d_list, hittable **d_world, camera **d_camera, int no_hittables) {
	for (int i = 0; i < no_hittables; i++) {
		delete *(d_list+i);
	}
	//delete *(d_list);
	//delete *(d_list+1);
	delete *d_world;
	delete *d_camera;
}



struct scene {
	int no_hittables = -1;
	float aspect = 0;

	hittable **d_list;
	hittable **d_world;
	camera   **d_camera;	

	__host__ scene(float a) : aspect(a) {}

	__host__ ~scene() {
		free_world<<<1,1>>>(d_list,d_world,d_camera, no_hittables);
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaFree(d_list));
		checkCudaErrors(cudaFree(d_world));
		checkCudaErrors(cudaFree(d_camera));
	}


	//__global__ virtual void create_world(hittable **d_list, hittable **d_world, camera **d_camera);
};


__global__ void create_basic_world(hittable **d_list, hittable **d_world, camera **d_camera, int no_hittables) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		//should make no_hittables number of hittables
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0, 1, 0)));
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0, 0, 1)));


		*d_world    = new hittable_list(d_list, no_hittables);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 40, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );	//needs to take the aspect ratio
	}
}


struct basic_scene : public scene {
	
	
	__host__ basic_scene(int aspec) : scene(aspec) {
		no_hittables = 2;

		checkCudaErrors(cudaMalloc((void**)&d_list, no_hittables*sizeof(hittable*) ));	
		checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*) ));

		checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*) ));

		create_basic_world<<<1,1>>>(d_list, d_world, d_camera, no_hittables);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};
