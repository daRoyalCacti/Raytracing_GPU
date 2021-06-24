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
#include "triangle.h"
#include "triangle_mesh.h"


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


/*
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

		//small box
		d_list[7] = new box(point3(0, 0, 0), point3(165, 165, 165), white);
		d_list[7] = new rotate_y(d_list[7], -18);
		d_list[7] = new translate(d_list[7], vec3(130, 0, 65) );	
		d_list[7] = new constant_medium(d_list[7], 0.01, color(1,1,1) );

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




__global__ void create_triangle_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		*(d_list)   = new triangle(vec3(-0.5f, 0, 0), vec3(0, 1, 10), vec3(0.f, 0, 0), 0, 0, 0, 1, 1, 0, new lambertian(vec3(0, 1, 0)) );
			//sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0, 1, 0)));
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0, 0, 1)));

		*d_world    = new hittable_list(d_list,2);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 40, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );
	}
	
}


struct triangle_scene : public scene {
	triangle_scene() : scene(16.0f/9.0f, 2, background_color::sky) {
		create_triangle_world<<<1,1>>>(d_list, d_world, d_camera);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};



__global__ void create_triangles_world(hittable **d_list, hittable **d_world, camera **d_camera, hittable** temp_list, curandState* s) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		//hittable** temp_list;
		//temp_list = new hittable*[4];

		temp_list[0] = new triangle(vec3(-0.5f, 0, 0), vec3(0, 1, 10), vec3(0.5f, 0, 0), 0, 0, 0, 1, 1, 0, new lambertian(vec3(0, 1, 0)) );
		temp_list[1] = new triangle(vec3(0.5f, 0, 0), vec3(0, 1, 10), vec3(0.5f, 1, 0), 0, 0, 0, 1, 1, 0, new lambertian(vec3(1, 1, 0)) );
		temp_list[2] = new triangle(vec3(1.5f, 0, 0), vec3(0, 2, 10), vec3(1.5f, 1, 0), 0, 0, 0, 1, 1, 0, new lambertian(vec3(1, 1, 1)) );
		temp_list[3] = new triangle(vec3(1.5f, 0, 0), vec3(1.5f, 1, 10), vec3(1.5f, 0, 2), 0, 0, 0, 1, 1, 0, new lambertian(vec3(1, 1, 1)) );


		*(d_list) = new triangle_mesh(temp_list, 4, 0, 1, s);
		//*(d_list) = new bvh_node(temp_list, 4, 0, 1, s);
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0, 0, 1)));

		*d_world    = new hittable_list(d_list, 2);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 40, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );
	}
	
}


struct triangles_scene : public scene {
	triangles_scene() : scene(16.0f/9.0f,  background_color::sky) {
		no_hittables = 2;

		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		hittable** temp_list;
		checkCudaErrors(cudaMalloc( (void**)&temp_list, 4*sizeof(hittable*) ));

		checkCudaErrors(cudaMalloc((void**)&d_list, sizeof(hittable*) + size_of_bvh(4)) );	

		create_triangles_world<<<1,1>>>(d_list, d_world, d_camera, temp_list, rand_state);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	}
};
*/

__global__ void create_door_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState* s, triangle_mesh<lambertian<image_texture>>* mesh) {
//__global__ void create_crate_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState* s, hittable* mesh) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		
		//assumes there is only 1 texture and 1 material
		const auto texture = new image_texture(mesh->tris->objs[0]->mp->albedo->data, mesh->tris->objs[0]->mp->albedo->width, mesh->tris->objs[0]->mp->albedo->height, mesh->tris->objs[0]->mp->albedo->bytes_per_pixel);
		const auto material = new lambertian(texture);
		
		const auto n = mesh->n;
		auto objects = new triangle<lambertian<image_texture>>*[n];
		for (size_t i = 0; i < n; i++) {
			objects[i] = new triangle(material, mesh->tris->objs[i]->vertex0, mesh->tris->objs[i]->vertex1, mesh->tris->objs[i]->vertex2, mesh->tris->objs[i]->u_0, mesh->tris->objs[i]->v_0, mesh->tris->objs[i]->u_1, mesh->tris->objs[i]->v_1, mesh->tris->objs[i]->u_2, mesh->tris->objs[i]->v_2, mesh->tris->objs[i]->S, mesh->tris->objs[i]->T, mesh->tris->objs[i]->v0, mesh->tris->objs[i]->v1, mesh->tris->objs[i]->d00, mesh->tris->objs[i]->d01, mesh->tris->objs[i]->d11, mesh->tris->objs[i]->invDenom, mesh->tris->objs[i]->vertex_normals, mesh->tris->objs[i]->normal0, mesh->tris->objs[i]->normal1, mesh->tris->objs[i]->normal2);
		}
		const auto info = new node_info[ceil(log2f(n))*n];
		for (size_t i = 0; i < ceil(log2f(n))*n; i++) {
			info[i] = node_info(mesh->tris->info[i].end, mesh->tris->info[i].num, mesh->tris->info[i].left, mesh->tris->info[i].right, mesh->tris->info[i].parent, mesh->tris->info[i].ids);
		}

		const auto bounds = new aabb[num_bvh_nodes_d(n) - n];
		for (size_t i = 0; i < num_bvh_nodes_d(n) - n; i++) {
			bounds[i] = aabb(mesh->tris->bounds[i].minimum, mesh->tris->bounds[i].maximum);
		}

		
		//const auto tris_ = new bvh_node(mesh->tris->objs, mesh->tris->info, mesh->tris->n, mesh->tris->bounds, mesh->tris->obj_s[0], mesh->tris->obj_s[1], mesh->tris->obj_s[2]);
		//const auto tris_ = new bvh_node(objects, mesh->tris->info, mesh->tris->n, mesh->tris->bounds, mesh->tris->obj_s[0], mesh->tris->obj_s[1], mesh->tris->obj_s[2]);
		const auto tris_ = new bvh_node(objects, info, mesh->tris->n, bounds, mesh->tris->obj_s[0], mesh->tris->obj_s[1], mesh->tris->obj_s[2]);

		//d_list[0] = new triangle_mesh(mesh->tris, mesh->n);
		d_list[0] = new triangle_mesh(tris_, mesh->n);//mesh; //new triangle_mesh(obj_list, obj_sizes, 0, 1, s, 0);
		d_list[1] = new sphere(vec3(0,-100,-1), 100, new lambertian<solid_color>(vec3(0, 1, 0)));
		
		/*d_list[0] = new sphere(vec3(0,-100,-1), 100, new lambertian<solid_color>(vec3(0, 1, 0)));
		d_list[1] = *mesh; //new triangle_mesh(obj_list, obj_sizes, 0, 1, s, 0);
	*/

		
		/*printf("mesh->n: %i\n", (*mesh)->n); 
		printf("mesh->tris->n: %i\n", (*mesh)->tris->n); 
		printf("mesh->tris->obj_s[0][1]: %i\n", (*mesh)->tris->obj_s[0][1]);
		printf("mesh->tris->obj_s[1][1]: %i\n", (*mesh)->tris->obj_s[1][1]);
		printf("mesh->tris->obj_s[2][1]: %i\n", (*mesh)->tris->obj_s[2][1]);
 		printf("mesh->tris->bounds[1].minimum.x: %f\n", (*mesh)->tris->bounds[1].minimum.x());
		printf("mesh->tris->info[2].ids[1]: %i\n", (*mesh)->tris->info[2].ids[0]);
		printf("mesh->tris->objs[1]->vertex0.x: %f\n", (*mesh)->tris->objs[1]->vertex0.x());
		printf("mesh->tris->objs[1]->mp->albedo->data[150]: %i\n", (*mesh)->tris->objs[1]->mp->albedo->data[150]);*/


		/*printf("mesh->n: %i\n", mesh->n); 
		printf("mesh->tris->n: %i\n", mesh->tris->n); 
		printf("mesh->tris->obj_s[0][1]: %i\n", mesh->tris->obj_s[0][1]);
		printf("mesh->tris->obj_s[1][1]: %i\n", mesh->tris->obj_s[1][1]);
		printf("mesh->tris->obj_s[2][1]: %i\n", mesh->tris->obj_s[2][1]);
 		printf("mesh->tris->bounds[1].minimum.x: %f\n", mesh->tris->bounds[1].minimum.x());
		printf("mesh->tris->info[2].ids[1]: %i\n", mesh->tris->info[2].ids[0]);
		printf("mesh->tris->objs[1]->vertex0.x: %f\n", mesh->tris->objs[1]->vertex0.x());
		printf("mesh->tris->objs[1]->mp->albedo->data[150]: %i\n", mesh->tris->objs[1]->mp->albedo->data[150]);*/

		/*
		hit_record temp;
		ray r;
		printf("xxxxxx\n");
		auto d = d_list[1]->hit(r, 0.0f, 1.0f, temp, s);
		printf("qqasfaf\n");
		auto c = d_list[0]->foo();
		//printf("asfaf\n");
		//auto b = mesh->foo();
		printf("xxxqqvasva\n");
		auto q = d_list[0]->hit(r, 0.0f, 1.0f, temp, s);
		printf("qqvasva\n");
		//auto a = mesh->hit(r, 0.0f, 1.0f, temp, s);
		//printf("vasva\n");
		*/

		



		*d_world    = new hittable_list(d_list, 2);
		*d_camera   = new camera(vec3(-3,4,-5), vec3(0,1,0), vec3(0,1,0), 20, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );


	}
	
}


struct door_scene : public scene {
	door_scene() : scene(16.0f/9.0f, background_color::sky) {
		//std::vector<std::string> objs;
		//hittable** obj_list;
		//int size_of_meshes;
		//objs.push_back("../assets/door/door.obj");
		
		//create_meshes(objs, obj_list, num, size_of_meshes);
		std::cout << "Generating model\n";
		auto door_mesh = generate_model("../assets/door/door.obj");


		triangle_mesh<lambertian<image_texture>>* door_mesh_d;
		//triangle_mesh<lambertian<image_texture>>* test;
		cudaMalloc((void**)&door_mesh_d, sizeof(triangle_mesh<lambertian<image_texture>>));
		checkCudaErrors(cudaMemcpy(door_mesh_d, door_mesh, sizeof(triangle_mesh<lambertian<image_texture>>), cudaMemcpyDefault));
		door_mesh->cpy_constit_d(door_mesh_d);

		/*cudaMalloc((void**)&door_mesh_d, sizeof(triangle_mesh<lambertian<image_texture>>*));
		cudaMalloc((void**)&test, sizeof(triangle_mesh<lambertian<image_texture>>));
		checkCudaErrors(cudaMemcpy(test, door_mesh, sizeof(triangle_mesh<lambertian<image_texture>>), cudaMemcpyDefault));

		checkCudaErrors(cudaMemcpy(door_mesh_d, &test, sizeof(triangle_mesh<lambertian<image_texture>>*), cudaMemcpyDefault)); 
		door_mesh->cpy_constit_d(test);*/


		std::cout << "door_mesh->n: " << door_mesh->n << "\n";
		std::cout << "door_mesh->tris->n: " << door_mesh->tris->n << "\n";
		std::cout << "door_mesh->tris->obj_s[0][1]: " << door_mesh->tris->obj_s[0][1] << "\n";
		std::cout << "door_mesh->tris->obj_s[1][1]: " << door_mesh->tris->obj_s[1][1] << "\n";
		std::cout << "door_mesh->tris->obj_s[2][1]: " << door_mesh->tris->obj_s[2][1] << "\n";
		std::cout << "door_mesh->tris->bounds[1].minimum.x: " << door_mesh->tris->bounds[1].minimum.x() << "\n";
		std::cout << "door_mesh->tris->info[2].ids[1]: " <<  door_mesh->tris->info[2].ids[0] << "\n";
		std::cout << "door_mesh->tris->objs[1]->vertex0.x: " << door_mesh->tris->objs[1]->vertex0.x() <<  "\n";
		std::cout << "door_mesh->tris->objs[1]->mp->albedo->data[150]: " <<  (int)door_mesh->tris->objs[1]->mp->albedo->data[150] << "\n";





		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		//std::cout << "Testing done\n";

		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	
		checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*) ));	
		//std::cout<< "tst\n";
		no_hittables = 2;
		//create_door_world<<<1,1>>>(d_list, d_world, d_camera, rand_state, door_mesh_d);
		create_door_world<<<1,1>>>(d_list, d_world, d_camera, rand_state, door_mesh_d);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
		//std::cout << "magica\n";
	}
};




/*
__global__ void create_backpack_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState* s, hittable** obj_list, unsigned* obj_sizes) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		
		d_list[0] = new triangle_mesh(obj_list, obj_sizes, 0, 1, s, 0);
		d_list[0] = new sphere(vec3(0,-100,-1), 100, new lambertian(vec3(0, 1, 0)));

		*d_world    = new hittable_list(d_list, 1);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 20, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );

	}
	
}





struct backpack_scene : public scene {
	backpack_scene() : scene(16.0f/9.0f, background_color::sky) {
		thrust::device_ptr<unsigned> num;
		std::vector<std::string> objs;
		hittable** obj_list;
		int size_of_meshes;
		objs.push_back("../assets/backpack/backpack.obj");
		
		create_meshes(objs, obj_list, num, size_of_meshes);

		

		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	
		//checkCudaErrors(cudaMalloc((void**)&d_list, size_of_meshes + sizeof(hittable*) ));	//+hittable* because of ground
		checkCudaErrors(cudaMalloc((void**)&d_list, sizeof(hittable*) ));	//+hittable* because of ground

		no_hittables = 1;//2;
		create_backpack_world<<<1,1>>>(d_list, d_world, d_camera, rand_state, obj_list, thrust::raw_pointer_cast(num));

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created

	}
};



__global__ void create_cup_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState* s, hittable** obj_list, unsigned* obj_sizes) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		*d_camera   = new camera(vec3(0,0,-1), vec3(0,0,0), vec3(0,1,0), 20, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );

		d_list[0] = new triangle_mesh(obj_list, obj_sizes, 0, 1, s, 0);
		d_list[1] = new sphere(vec3(0,-100,-1), 100, new lambertian(vec3(0, 1, 0)));

		*d_world    = new hittable_list(d_list, 2);

	}
	
}





struct cup_scene : public scene {
	cup_scene() : scene(16.0f/9.0f, background_color::sky) {
		thrust::device_ptr<unsigned> num;
		std::vector<std::string> objs;
		hittable** obj_list;
		int size_of_meshes;
		objs.push_back("../assets/cup/cup.obj");
		
		create_meshes(objs, obj_list, num, size_of_meshes);

		

		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	
		checkCudaErrors(cudaMalloc((void**)&d_list, size_of_meshes + sizeof(hittable*) ));	//+hittable* because of ground

		no_hittables = 2;
		create_cup_world<<<1,1>>>(d_list, d_world, d_camera, rand_state, obj_list, thrust::raw_pointer_cast(num));

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created

	}
}; */




__global__ void create_crate_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState* s, triangle_mesh<lambertian<image_texture>>* mesh) {
//__global__ void create_crate_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState* s, hittable* mesh) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		
		//assumes there is only 1 texture and 1 material
		const auto texture = new image_texture(mesh->tris->objs[0]->mp->albedo->data, mesh->tris->objs[0]->mp->albedo->width, mesh->tris->objs[0]->mp->albedo->height, mesh->tris->objs[0]->mp->albedo->bytes_per_pixel);
		const auto material = new lambertian(texture);
		
		const auto n = mesh->n;
		auto objects = new triangle<lambertian<image_texture>>*[n];
		for (size_t i = 0; i < n; i++) {
			objects[i] = new triangle(material, mesh->tris->objs[i]->vertex0, mesh->tris->objs[i]->vertex1, mesh->tris->objs[i]->vertex2, mesh->tris->objs[i]->u_0, mesh->tris->objs[i]->v_0, mesh->tris->objs[i]->u_1, mesh->tris->objs[i]->v_1, mesh->tris->objs[i]->u_2, mesh->tris->objs[i]->v_2, mesh->tris->objs[i]->S, mesh->tris->objs[i]->T, mesh->tris->objs[i]->v0, mesh->tris->objs[i]->v1, mesh->tris->objs[i]->d00, mesh->tris->objs[i]->d01, mesh->tris->objs[i]->d11, mesh->tris->objs[i]->invDenom, mesh->tris->objs[i]->vertex_normals, mesh->tris->objs[i]->normal0, mesh->tris->objs[i]->normal1, mesh->tris->objs[i]->normal2);
		}
		const auto info = new node_info[ceil(log2f(n))*n];
		for (size_t i = 0; i < ceil(log2f(n))*n; i++) {
			info[i] = node_info(mesh->tris->info[i].end, mesh->tris->info[i].num, mesh->tris->info[i].left, mesh->tris->info[i].right, mesh->tris->info[i].parent, mesh->tris->info[i].ids);
		}

		const auto bounds = new aabb[num_bvh_nodes_d(n) - n];
		for (size_t i = 0; i < num_bvh_nodes_d(n) - n; i++) {
			bounds[i] = aabb(mesh->tris->bounds[i].minimum, mesh->tris->bounds[i].maximum);
		}

		
		//const auto tris_ = new bvh_node(mesh->tris->objs, mesh->tris->info, mesh->tris->n, mesh->tris->bounds, mesh->tris->obj_s[0], mesh->tris->obj_s[1], mesh->tris->obj_s[2]);
		//const auto tris_ = new bvh_node(objects, mesh->tris->info, mesh->tris->n, mesh->tris->bounds, mesh->tris->obj_s[0], mesh->tris->obj_s[1], mesh->tris->obj_s[2]);
		const auto tris_ = new bvh_node(objects, info, mesh->tris->n, bounds, mesh->tris->obj_s[0], mesh->tris->obj_s[1], mesh->tris->obj_s[2]);

		//d_list[0] = new triangle_mesh(mesh->tris, mesh->n);
		d_list[0] = new triangle_mesh(tris_, mesh->n);//mesh; //new triangle_mesh(obj_list, obj_sizes, 0, 1, s, 0);
		d_list[1] = new sphere(vec3(0,-100,-1), 100, new lambertian<solid_color>(vec3(0, 1, 0)));
		
		/*d_list[0] = new sphere(vec3(0,-100,-1), 100, new lambertian<solid_color>(vec3(0, 1, 0)));
		d_list[1] = *mesh; //new triangle_mesh(obj_list, obj_sizes, 0, 1, s, 0);
	*/

		
		/*printf("mesh->n: %i\n", (*mesh)->n); 
		printf("mesh->tris->n: %i\n", (*mesh)->tris->n); 
		printf("mesh->tris->obj_s[0][1]: %i\n", (*mesh)->tris->obj_s[0][1]);
		printf("mesh->tris->obj_s[1][1]: %i\n", (*mesh)->tris->obj_s[1][1]);
		printf("mesh->tris->obj_s[2][1]: %i\n", (*mesh)->tris->obj_s[2][1]);
 		printf("mesh->tris->bounds[1].minimum.x: %f\n", (*mesh)->tris->bounds[1].minimum.x());
		printf("mesh->tris->info[2].ids[1]: %i\n", (*mesh)->tris->info[2].ids[0]);
		printf("mesh->tris->objs[1]->vertex0.x: %f\n", (*mesh)->tris->objs[1]->vertex0.x());
		printf("mesh->tris->objs[1]->mp->albedo->data[150]: %i\n", (*mesh)->tris->objs[1]->mp->albedo->data[150]);*/


		/*printf("mesh->n: %i\n", mesh->n); 
		printf("mesh->tris->n: %i\n", mesh->tris->n); 
		printf("mesh->tris->obj_s[0][1]: %i\n", mesh->tris->obj_s[0][1]);
		printf("mesh->tris->obj_s[1][1]: %i\n", mesh->tris->obj_s[1][1]);
		printf("mesh->tris->obj_s[2][1]: %i\n", mesh->tris->obj_s[2][1]);
 		printf("mesh->tris->bounds[1].minimum.x: %f\n", mesh->tris->bounds[1].minimum.x());
		printf("mesh->tris->info[2].ids[1]: %i\n", mesh->tris->info[2].ids[0]);
		printf("mesh->tris->objs[1]->vertex0.x: %f\n", mesh->tris->objs[1]->vertex0.x());
		printf("mesh->tris->objs[1]->mp->albedo->data[150]: %i\n", mesh->tris->objs[1]->mp->albedo->data[150]);*/

		/*
		hit_record temp;
		ray r;
		printf("xxxxxx\n");
		auto d = d_list[1]->hit(r, 0.0f, 1.0f, temp, s);
		printf("qqasfaf\n");
		auto c = d_list[0]->foo();
		//printf("asfaf\n");
		//auto b = mesh->foo();
		printf("xxxqqvasva\n");
		auto q = d_list[0]->hit(r, 0.0f, 1.0f, temp, s);
		printf("qqvasva\n");
		//auto a = mesh->hit(r, 0.0f, 1.0f, temp, s);
		//printf("vasva\n");
		*/

		



		*d_world    = new hittable_list(d_list, 2);
		*d_camera   = new camera(vec3(-3,4,-5), vec3(0,1,0), vec3(0,1,0), 20, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );


	}
	
}





struct crate_scene : public scene {
	crate_scene() : scene(16.0f/9.0f, background_color::sky) {
		//std::vector<std::string> objs;
		//hittable** obj_list;
		//int size_of_meshes;
		//objs.push_back("../assets/door/door.obj");
		
		//create_meshes(objs, obj_list, num, size_of_meshes);
		std::cout << "Generating model\n";
		auto door_mesh = generate_model("../assets/crate/Crate1.obj");


		triangle_mesh<lambertian<image_texture>>* door_mesh_d;
		//triangle_mesh<lambertian<image_texture>>* test;
		cudaMalloc((void**)&door_mesh_d, sizeof(triangle_mesh<lambertian<image_texture>>));
		checkCudaErrors(cudaMemcpy(door_mesh_d, door_mesh, sizeof(triangle_mesh<lambertian<image_texture>>), cudaMemcpyDefault));
		door_mesh->cpy_constit_d(door_mesh_d);

		/*cudaMalloc((void**)&door_mesh_d, sizeof(triangle_mesh<lambertian<image_texture>>*));
		cudaMalloc((void**)&test, sizeof(triangle_mesh<lambertian<image_texture>>));
		checkCudaErrors(cudaMemcpy(test, door_mesh, sizeof(triangle_mesh<lambertian<image_texture>>), cudaMemcpyDefault));

		checkCudaErrors(cudaMemcpy(door_mesh_d, &test, sizeof(triangle_mesh<lambertian<image_texture>>*), cudaMemcpyDefault)); 
		door_mesh->cpy_constit_d(test);*/


		std::cout << "door_mesh->n: " << door_mesh->n << "\n";
		std::cout << "door_mesh->tris->n: " << door_mesh->tris->n << "\n";
		std::cout << "door_mesh->tris->obj_s[0][1]: " << door_mesh->tris->obj_s[0][1] << "\n";
		std::cout << "door_mesh->tris->obj_s[1][1]: " << door_mesh->tris->obj_s[1][1] << "\n";
		std::cout << "door_mesh->tris->obj_s[2][1]: " << door_mesh->tris->obj_s[2][1] << "\n";
		std::cout << "door_mesh->tris->bounds[1].minimum.x: " << door_mesh->tris->bounds[1].minimum.x() << "\n";
		std::cout << "door_mesh->tris->info[2].ids[1]: " <<  door_mesh->tris->info[2].ids[0] << "\n";
		std::cout << "door_mesh->tris->objs[1]->vertex0.x: " << door_mesh->tris->objs[1]->vertex0.x() <<  "\n";
		std::cout << "door_mesh->tris->objs[1]->mp->albedo->data[150]: " <<  (int)door_mesh->tris->objs[1]->mp->albedo->data[150] << "\n";





		curandState* rand_state;
		checkCudaErrors(cudaMalloc((void**)&rand_state, sizeof(curandState) ));

		//std::cout << "Testing done\n";

		world_init<<<1,1>>>(rand_state);	//intialising rand_state
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	
		checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*) ));	
		//std::cout<< "tst\n";
		no_hittables = 2;
		//create_door_world<<<1,1>>>(d_list, d_world, d_camera, rand_state, door_mesh_d);
		create_crate_world<<<1,1>>>(d_list, d_world, d_camera, rand_state, door_mesh_d);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
		//std::cout << "magica\n";
	}
};


