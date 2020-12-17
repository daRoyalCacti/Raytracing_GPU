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




/*
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
}*/

__device__ vec3 color_f(ray& r, hittable **world, curandState *local_rand_state, int depth) {
	ray cur_ray = r;
	const vec3 background(0.7f, 0.8f, 1.0f);
	color cur_attenuation(1,1,1);
	color cur_col(1,1,1);

	for (int i = 0; i < depth; i++) {
		hit_record rec;

		if (!(*world)->hit(cur_ray, 0.001f, infinity, rec, local_rand_state)) 
			return cur_attenuation*background;

		ray scattered;
		color attenuation;
		const color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

		if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
			cur_col = emitted + cur_attenuation*cur_col;
			cur_attenuation *= attenuation;
			cur_ray = scattered;
		} else {
			return cur_attenuation*emitted;
		}

	}
	return color(0,0,0);	//exceeded recursion

	//return emitted + attenuation*color_f(scattered, world, local_rand_state, depth-1);
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera **cam, curandState *rand_state,  hittable **world, int max_depth, int id) {
	//max_x for size of total image
	//max_x2 for size of 1 frame buffer

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;	//if trying the work with more values than wanted
	int pixel_index = j*max_x + i;

	curandState local_rand_state = rand_state[(id*pixel_index + id)%(max_y*max_x)];
	vec3 col(0,0,0);
	
	for(int s=0; s < ns; s++) {
		float u = float(i +random_float(&local_rand_state)) / max_x;
		float v = float(j+random_float(&local_rand_state)) / max_y;
		
		ray r = (*cam)->get_ray(rand_state, u,v);
		col += color_f(r, world, &local_rand_state, max_depth);
	}

	fb[pixel_index] = col/float(ns);
}



int main() {
	bvh_nodez node(5);
	//std::cerr << num_bvh_nodes(15) << std::endl;

	for (int i = 0; i < num_bvh_nodes(node.n); i++) {
		std::cerr << "For node " << i << "\n";
		std::cerr << "\tEnd node:\t\t" << node.info[i].end << "\n";
		std::cerr << "\tNumber of objects:\t" << node.info[i].num << "\n";
		std::cerr << "\tLeft connection:\t" << node.info[i].left << "\n";
		std::cerr << "\tRight connection:\t" << node.info[i].right << "\n";
		std::cerr << "\tParent node:\t\t" << node.info[i].parent << "\n";
	}
/*
	//start timing
	const auto start = std::chrono::system_clock::now();
	const std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	std::cerr << "Computation started at " << std::ctime(&start_time);


	const double aspect_ratio = 16.0 / 9.0;
	const unsigned tx = 8;	//dividing the work on the GPU into
	const unsigned ty = 8; 	//threads of tx*ty threads
	const unsigned rpfb = 100;	//number of rays per pixel to use in a given frame buffer
	const unsigned no_fb = 10;	//number of frame buffers

	const unsigned nx = 1200;	//image width in frame buffer (also the output image size)
	const unsigned ns = rpfb*no_fb;	//rays per pixel

	const unsigned ny = static_cast<unsigned>(nx / aspect_ratio);
	const unsigned num_pixels = nx*ny;
	//const unsigned no_fb = static_cast<unsigned>(float(ns)/rpfb+1);



	std::cerr << "Generating a " << nx << "x" << ny << " image with " << ns << " rays per pixel\n";
	std::cerr << "using " << tx << "x" << ty << " blocks and " << no_fb << " frame buffers.\n";



	std::cerr << "Allocating Frame Buffer" << std::flush;
	//Frame buffer (holds the image in the GPU)
	vec3 *fb;
	const size_t fb_size = num_pixels*sizeof(vec3)*no_fb;	
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));	//allocating the frame buffer on the GPU
	
	std::cerr << "\rCreating World                " << std::flush;
	big_scene1 curr_scene;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created
	
	
	//Render to the frame buffer
	dim3 blocks(nx*no_fb/tx+1, ny/ty+1);	//making the frame buffer exceptionally long to combine the multiple frame buffers
	dim3 threads(tx, ty);
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*no_fb*sizeof(curandState) ));



	std::cerr << "\rIntialising the render        " << std::flush;
	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);		//initialising the render -- currently just setting up the random numbers
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	


	for (int i = 0; i < no_fb; i++) {
		std::cerr << "\rRendering to frame buffer " << i+1 << "/" << no_fb << "         "  << std::flush;
		render<<<blocks, threads>>>(fb, nx, ny, rpfb,	//render is a function defined above
						curr_scene.d_camera,
						d_rand_state,
						curr_scene.d_world, 50, i);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tells the CPU that the GPU is done rendering
		write_frame_buffer("output/" + std::to_string(i) + ".ppm", fb, nx, ny);
	}

	
	std::cerr << "\rAveraging Frame Buffers         " << std::flush;
	average_images("./output", "image.ppm");



	std::cerr << "\rCleaning Up                   " << std::flush;	
	//clean up
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(fb));


	//end timing
	const auto end = std::chrono::system_clock::now();
	const std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cerr << "\rComputation ended at " << std::ctime(&end_time) << "                ";

	const std::chrono::duration<double> elapsed_seconds = end - start;
	std::cerr << "Elapsed time: " << elapsed_seconds.count() << "s  or  " << elapsed_seconds.count() / 60.0f << "m  or  " << elapsed_seconds.count() / (60.0f * 60.0f) << "h\n";
*/
	return 0;
}
