#pragma once

#include "hittable.h"
#include "hittable_list.h"
#include "camera.h"
#include "common.h"
#include "sphere.h"
#include "material.h"
#include "texture.h"


enum class background_color {sky, black};

__global__ void free_world(hittable ** d_list, hittable **d_world, camera **d_camera, int no_hittables) {
	for (int i = 0; i < no_hittables; i++) {
		delete *(d_list+i);
	}
	//delete *(d_list);
	//delete *(d_list+1);
	delete *d_world;
	delete *d_camera;
}


__global__ void world_init(curandState *rand_State) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		curand_init(1984, 0, 0, rand_State);
	}
}



struct scene {
	int no_hittables = -1;
	float aspect = 0;
	color background;

	hittable **d_list;
	hittable **d_world;
	camera   **d_camera;	

	scene(float a, int h, background_color bc) : aspect(a), no_hittables(h) {
		set_background(bc);

		checkCudaErrors(cudaMalloc((void**)&d_list, no_hittables*sizeof(hittable*) ));	
		checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*) ));

		checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*) ));
	}

	~scene() {
		free_world<<<1,1>>>(d_list,d_world,d_camera, no_hittables);
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaFree(d_list));
		checkCudaErrors(cudaFree(d_world));
		checkCudaErrors(cudaFree(d_camera));
	}

	inline void set_background(const background_color& col) {
		if (col == background_color::sky) {
			background = color(0.7f, 0.8f, 1.0f);
		} else if (col == background_color::black) {
			background = color(0.0f, 0.0f, 0.0f);
		} else {
			background = color(1.0, 0.0, 1.0);	//terrible color for debugging
		}
	}


	//__global__ virtual void create_world(hittable **d_list, hittable **d_world, camera **d_camera);
};


__global__ void create_basic_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0, 1, 0)));
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0, 0, 1)));

		*d_world    = new hittable_list(d_list,2);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 40, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );
	}
	
}


struct basic_scene : public scene {
	basic_scene() : scene(16.0f/9.0f, 2, background_color::sky) {
		create_basic_world<<<1,1>>>(d_list, d_world, d_camera);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};





__global__ void create_first_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism

		const auto material_ground = new lambertian(color(0.8, 0.8, 0));
		const auto material_center = new lambertian(color(0.1, 0.2, 0.3));
		const auto material_left   = new dielectric(2.5);
		const auto material_right  = new metal(color(0.8, 0.6, 0.2), 0.2);
		const auto material_front  = new dielectric(2);

		*(d_list+0) = new sphere(point3( 0.0,-100.5,-1.0), 100.0, material_ground);
		*(d_list+1) = new sphere(point3( 0.0,   0.0,-1.0),   0.5, material_center);
		*(d_list+2) = new sphere(point3(-1.0,   0.0,-1.0),   0.5, material_left);
		*(d_list+3) = new sphere(point3( 1.0,   0.0,-1.0),   0.5, material_right);
		//having a sphere with a negative radius allows for the creation of hollow spheres
		*(d_list+4) = new sphere(point3( 0.0,   1.0,-0.75),  0.25, material_front);
		*(d_list+5) = new sphere(point3( 0.0,   1.0,-0.75), -0.25, material_front);

		*d_world    = new hittable_list(d_list, 6);
		*d_camera   = new camera(vec3(-2,2,-3), vec3(0,0,-1), vec3(0,1,0), 20, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );

	}

}


struct first_scene : public scene {
	first_scene() : scene(16.0f/9.0f, 6, background_color::sky) {
		create_first_world<<<1,1>>>(d_list, d_world, d_camera);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};




__global__ void create_big_world1(hittable **d_list, hittable **d_world, camera **d_camera, curandState* rs) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		//hittable **d_list;
		//*d_list = new hittable[488];

		//hittable_list **obj;

		int counter = 0;

		const auto checker = new checker_texture(color(0.2f, 0.3f, 0.1f), color(0.9f, 0.9f, 0.9f));
		const auto ground_material = new lambertian(checker);
		d_list[counter++] = new sphere(point3(0,-1000,0), 1000, ground_material);

		for (int a = -11; a<11; a++) {	//centers of spheres x
			for (int b = -11; b<11; b++) {	//centers of spheres y
				const auto choose_mat = random_float(rs);	//random number to decide what material to use
				const point3 center(a + 0.9f*random_float(rs), 0.2f, b + 0.9f*random_float(rs));

				if ((center - point3(4, 0.2, 0)).length_squared() > 0.9f*0.9f) {	//not points where main balls go
					material* sphere_material;

					if (choose_mat < 0.8) {
						//diffuse (has moving spheres)
						const auto albedo = color::random(rs) * color::random(rs);
						sphere_material = new lambertian(albedo);	
						const auto center2 = center + vec3(0, random_float(rs, 0, 0.5f), 0);	//spheres moving downwards at random speeds				
						d_list[counter++] = new moving_sphere(center, center2, 0.0f, 1.0f, 0.2f, sphere_material);
					} else if (choose_mat < 0.95) {
						//metal
						const auto albedo = color::random(rs, 0.5f, 1);
						const auto fuzz = random_float(rs, 0, 0.5f);
						sphere_material = new metal(albedo, fuzz);
						d_list[counter++] = new sphere(center, 0.2f, sphere_material);
					} else {
						//glass
						sphere_material = new dielectric(1.5f);
						d_list[counter++] = new sphere(center, 0.2f, sphere_material);	
					}
					
				} else {
					d_list[counter++] = new sphere(vec3(10000, -10000, 10000), 0.00001f,  new lambertian(vec3(0,0,0)) );
				}

			}
		}

		//main balls
		const auto material1 = new dielectric(1.5f);
		d_list[counter++] = new sphere(point3(0, 1, 0), 1.0f, material1);

		const auto material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
		d_list[counter++] = new sphere(point3(-4, 1, 0), 1.0f, material2);

		const auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
		d_list[counter++] = new sphere(point3(4, 1, 0), 1.0f, material3);
		
		*d_world = new hittable_list(d_list, 488);


		*d_camera   = new camera(vec3(13.0f,2.0f,-3.0f), vec3(0.0f,0.0f,0.0f), vec3(0,1,0), 20, 16.0f/9.0f, 0.1f, 10.0f, 0, 1 );

	}
}


struct big_scene1 : public scene {
	big_scene1() : scene(16.0f/9.0f, 488, background_color::sky) {
		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		world_init<<<1,1>>>(rand_state);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		create_big_world1<<<1,1>>>(d_list, d_world, d_camera, rand_state);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};

