#include <iostream>
#include <chrono>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "common.h"

#include "hittable_list.h"

//for compiling purposes
#include "color.h"
#include "camera.h"
#include "moving_sphere.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
#include "bvh.h"

#include "scenes.h"


__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0, 1, 0)));
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0, 0, 1)));
		*d_world    = new hittable_list(d_list,2);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 40, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );
	}
}





__device__ vec3 color_f(ray& r, hittable **world, curandState *local_rand_state, int depth) {
	const vec3 background(0.7f, 0.8f, 1.0f);

	hit_record rec;

	if (depth <= 0)
		return color(0,0,0);
	
	if (!(*world)->hit(r, 0.001f, infinity, rec, local_rand_state)) 
		return background;

	ray scattered;
	color attenuation;
	const color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state))
		return emitted;
	
	return emitted + attenuation*color_f(scattered, world, local_rand_state, depth-1);	
}


__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	//Initialising for random numbers
	//Each thread gets the same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera **cam, curandState *rand_state,  hittable **world, int max_depth) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;	//if trying the work with more values than wanted
	int pixel_index = j*max_x + i;

	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0,0,0);
	
	for(int s=0; s < ns; s++) {
		float u = float(i+random_float(&local_rand_state)) / max_x;
		float v = float(j+random_float(&local_rand_state)) / max_y;
		
		ray r = (*cam)->get_ray(rand_state, u,v);
		col += color_f(r, world, &local_rand_state, max_depth);
	}

	fb[pixel_index] = col/float(ns);
}



int main() {
	const unsigned nx = 1200;	//image width in frame buffer (also the output image size)
	const double aspect_ratio = 16.0 / 9.0;
	const unsigned ny = static_cast<unsigned>(nx / aspect_ratio);
	const unsigned num_pixels = nx*ny;
	const unsigned ns = 100;	//rays per pixel

	const unsigned tx = 8;	//dividing the work on the GPU into
	const unsigned ty = 8; 	//threads of tx*ty threads

	std::cerr << "Generating a " << nx << "x" << ny << " image with " << ns << " rays per pixel\n";
	std::cerr << "using " << tx << "x" << ty << " blocks.\n";



	std::cerr << "Allocating Frame Buffer" << std::flush;
	//Frame buffer (holds the image in the GPU)
	vec3 *fb;
	const size_t fb_size = num_pixels*sizeof(vec3);	
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));	//allocating the frame buffer on the GPU
	
	std::cerr << "\rCreating World" << std::flush;
	//scene curr_scene = basic_scene(aspect_ratio);
	//make our world of hittables and the camera
	hittable **d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*) ));	//2 because 2 hittables
	hittable **d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable *) ));

	camera **d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*) ));

	create_world<<<1,1>>>(d_list, d_world, d_camera);		//create_world is defined above
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	
	
	//Render to the frame buffer
	dim3 blocks(nx/tx+1, ny/ty+1);
	dim3 threads(tx, ty);
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState) ));

	std::cerr << "\rIntialising the render" << std::flush;
	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);		//initialising the render -- currently just setting up the random numbers
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	std::cerr << "\rRendering to frame buffer" << std::flush;
	render<<<blocks, threads>>>(fb, nx, ny, ns,	//render is a function defined above
					d_camera,
					d_rand_state,
					d_world, 10);		
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tells the CPU that the GPU is done rendering
	
	std::cerr << "\rOutputting image" << std::flush;
	write_frame_buffer(std::cout, fb, nx, ny);

	std::cerr << "\rCleaning Up" << std::flush;	
	//clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1,1>>>(d_list,d_world,d_camera,2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(fb));

	std::cerr << std::endl;

	return 0;
}
