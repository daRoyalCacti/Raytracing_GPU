#pragma once

#include "hittable.h"
#include "hittable_list.h"
#include "camera.h"
#include "common.h"
#include "sphere.h"
#include "material.h"
#include "moving_sphere.h"
#include "texture.h"
#include "bvh.h"
#include "constant_medium.h"


enum class background_color {sky, black};

__global__ void free_world(hittable ** d_list, hittable **d_world, camera **d_camera, int no_hittables) {
	for (int i = 0; i < no_hittables; i++) {
		delete *(d_list+i);
	}
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

	scene(float a, background_color bc) : aspect(a) {
		set_background(bc);
		checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable) ));
		checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*) ));
	}

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


__global__ void create_big_world1(hittable **d_list1, hittable **d_world, camera **d_camera, curandState* rs, hittable **d_list) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism

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
		
		*d_list1 = new bvh_node(d_list, 488, 0, 1, rs);
		*d_world = new hittable_list(d_list1, 1);


		*d_camera   = new camera(vec3(13.0f,2.0f,-3.0f), vec3(0.0f,0.0f,0.0f), vec3(0,1,0), 20, 16.0f/9.0f, 0.1f, 10.0f, 0, 1 );

	}
}


struct big_scene1 : public scene {
	big_scene1() : scene(16.0f/9.0f, background_color::sky) {
		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		hittable** d_list_temp;
		checkCudaErrors(cudaMalloc( (void**)&d_list_temp, 488*sizeof(hittable*) ));
			
		checkCudaErrors(cudaMalloc((void**)&d_list, size_of_bvh(22*22+4) ));

		no_hittables = 1;
		create_big_world1<<<1,1>>>(d_list, d_world, d_camera, rand_state, d_list_temp);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};


__global__ void create_two_spheres_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		const auto checker = new checker_texture(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

		d_list[0] = new sphere(point3(0,-10,0), 10, new lambertian(checker) );
		d_list[1] = new sphere(point3(0, 10,0), 10, new lambertian(checker) );
		
		*d_world    = new hittable_list(d_list, 2);
		*d_camera   = new camera(vec3(13.0f, 2.0f, 3.0f), vec3(0.0f,0.0f,0.0f), vec3(0,1,0), 20, 16.0f/9.0f, 0.1f, 10.0f, 0, 1 );
	}
}

struct two_spheres_scene : public scene {
	two_spheres_scene() : scene(16.0f/9.0f, 2, background_color::sky) {
		create_two_spheres_world<<<1,1>>>(d_list, d_world, d_camera);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};




__global__ void create_two_perlin_spheres_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState *s) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		const auto pertex1 = new marble_texture(s, 4);
		const auto pertex2 = new turbulent_texture(s, 5);

		d_list[0] = new sphere(point3(0,-1000,0), 1000, new lambertian(pertex1) );
		d_list[1] = new sphere(point3(0,    2,0),    2, new lambertian(pertex2) );

		*d_world    = new hittable_list(d_list, 2);
		*d_camera   = new camera(vec3(13.0f, 2.0f, 3.0f), vec3(0.0f,0.0f,0.0f), vec3(0,1,0), 20, 16.0f/9.0f, 0.1f, 10.0f, 0, 1 );
	}
}

struct two_perlin_spheres_scene : public scene {
	two_perlin_spheres_scene() : scene(16.0f/9.0f, 2, background_color::sky) {
		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));
		
		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		create_two_perlin_spheres_world<<<1,1>>>(d_list, d_world, d_camera, rand_state);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};



__global__ void create_earth_world(hittable **d_list, hittable **d_world, camera **d_camera, 
		unsigned char* imdata, int *widths, int *heights, int* bytes_per_pixels) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		const auto earth_texture = new image_texture(imdata, widths, heights, bytes_per_pixels, 0);

		const auto earth_surface = new lambertian(earth_texture);
		d_list[0] = new sphere(point3(0,0,0), 2, earth_surface);

		const auto difflight = new diffuse_light(color(4.0, 4.0, 4.0));
		d_list[1] = new xy_rect(-5, 5, -3, 3, 6, difflight);


		*d_world    = new hittable_list(d_list, 2);
		*d_camera   = new camera(vec3(13.0f, 0.0f, 3.0f), vec3(0.0f,0.0f,0.0f), vec3(0,1,0), 20, 16.0f/9.0f, 0.1f, 10.0f, 0, 1 );
	}
}

struct earth_scene : public scene {
	earth_scene() : scene(16.0f/9.0f, background_color::black) {
		
		thrust::device_ptr<unsigned char> imdata;
		thrust::device_ptr<int> imwidths;
		thrust::device_ptr<int> imhs;
		thrust::device_ptr<int> imch;
		
		std::vector<const char*> image_locs;
		image_locs.push_back("../textures/earthmap.jpg");
		
		make_image(image_locs, imdata, imwidths, imhs, imch);


		checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*) ));

		create_earth_world<<<1,1>>>(d_list, d_world, d_camera, 
				thrust::raw_pointer_cast(imdata),
				thrust::raw_pointer_cast(imwidths),
				thrust::raw_pointer_cast(imhs),
				thrust::raw_pointer_cast(imch) );
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created

	}
};


__global__ void create_cornell_box_world(hittable **d_list, hittable **d_world, camera **d_camera) {
		const auto red = new lambertian(color(0.65, 0.05, 0.05));
		const auto white = new lambertian(color(0.73, 0.73, 0.73));
		const auto green = new lambertian(color(0.12, 0.45, 0.15));
		const auto light = new diffuse_light(color(15, 15, 15));	//very bright light

		d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);	//left wall
		d_list[1] = new yz_rect(0, 555, 0, 555, 0  , red  );	//right wall

		d_list[2] = new xz_rect(213, 343, 227, 332, 554, light);	//small light on roof

		d_list[3] = new xz_rect(0, 555, 0, 555, 0  , white);	//floor
		d_list[4] = new xz_rect(0, 555, 0, 555, 555, white);	//roof

		d_list[5] = new xy_rect(0, 555, 0, 555, 555, white);	//back wall
	

		//big box
		d_list[6] = new box(point3(0, 0, 0), point3(165, 330, 165), white);
		d_list[6] = new rotate_y(d_list[6], 15);
		d_list[6] = new translate(d_list[6], vec3(265, 0, 295));

		//small box
		d_list[7] = new box(point3(0, 0, 0), point3(165, 165, 165), white);
		d_list[7] = new rotate_y(d_list[7], -18);
		d_list[7] = new translate(d_list[7], vec3(130, 0, 65) );
				
		*d_world    = new hittable_list(d_list, 8);
		*d_camera   = new camera(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0,1,0), 40, 1.0f, 0.0f, 10.0f, 0, 1 );
}

struct cornell_box_scene : public scene {
	cornell_box_scene() : scene(1.0f, 18, background_color::black) {
		create_cornell_box_world<<<1,1>>>(d_list, d_world, d_camera);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};




__global__ void create_cornell_smoke_box_world(hittable **d_list, hittable **d_world, camera **d_camera) {
		const auto red = new lambertian(color(0.65, 0.05, 0.05));
		const auto white = new lambertian(color(0.73, 0.73, 0.73));
		const auto green = new lambertian(color(0.12, 0.45, 0.15));
		const auto light = new diffuse_light(color(15, 15, 15));	//very bright light

		d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);	//left wall
		d_list[1] = new yz_rect(0, 555, 0, 555, 0  , red  );	//right wall

		d_list[2] = new xz_rect(113, 443, 127, 432, 554, light);	//light on roof

		d_list[3] = new xz_rect(0, 555, 0, 555, 0  , white);	//floor
		d_list[4] = new xz_rect(0, 555, 0, 555, 555, white);	//roof

		d_list[5] = new xy_rect(0, 555, 0, 555, 555, white);	//back wall


		//big box
		d_list[6] = new box(point3(0, 0, 0), point3(165, 330, 165), white);
		d_list[6] = new rotate_y(d_list[6], 15);
		d_list[6] = new translate(d_list[6], vec3(265, 0, 295));
		d_list[6] = new constant_medium(d_list[6], 0.01, color(0,0,0) );
		//world.add(make_shared<constant_medium>(box1, 0.01, color(0,0,0)) );

		//small box
		d_list[7] = new box(point3(0, 0, 0), point3(165, 165, 165), white);
		d_list[7] = new rotate_y(d_list[7], -18);
		d_list[7] = new translate(d_list[7], vec3(130, 0, 65) );	
		d_list[7] = new constant_medium(d_list[7], 0.01, color(1,1,1) );
		//world.add(make_shared<constant_medium>(box2, 0.01, color(1,1,1)));

		*d_world    = new hittable_list(d_list, 8);
		*d_camera   = new camera(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0,1,0), 40, 1.0f, 0.0f, 10.0f, 0, 1 );
}

struct cornell_smoke_box_scene : public scene {
	cornell_smoke_box_scene() : scene(1.0f, 18, background_color::black) {
		create_cornell_smoke_box_world<<<1,1>>>(d_list, d_world, d_camera);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};
