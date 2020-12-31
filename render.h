#include "vec3.h"
#include "ray.h"
#include "common.h"

#include <string>
#include <random>

#include "hittable_list.h"

#include "color.h"
#include "camera.h"
#include "scenes.h"


//for creating a directory
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 


struct render_settings {
	double aspect_ratio;
	unsigned image_width = 1200, image_height;
	unsigned samples_per_pixel_per_fb = 100;
	unsigned no_fb = 10;
	unsigned threads_x = 8, threads_y = 8;
	unsigned max_depth = 50;

	unsigned rays_per_pixel;
	unsigned num_pixels;

	inline void calc_height() {
		image_height = static_cast<int>(image_width / aspect_ratio);
	}

	inline void calc_rays_per_pixel() {
		rays_per_pixel = samples_per_pixel_per_fb * no_fb;
	}

	inline void calc_num_pixels() {
		num_pixels = image_width * image_height;
	}

	inline void calc_all() {
		calc_height();
		calc_rays_per_pixel();
		calc_num_pixels();
	}

};




__device__ vec3 color_f(ray& r, hittable **world, curandState *local_rand_state, int depth, vec3 background) {
	ray cur_ray = r;
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera **cam, curandState *rand_state,  hittable **world, int max_depth, int id, color back) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;	//if trying the work with more values than wanted
	int pixel_index = j*max_x + i;

	curandState local_rand_state = rand_state[((id+1)*pixel_index + id+1)%(max_y*max_x)];
	vec3 col(0,0,0);
	
	for(int s=0; s < ns; s++) {
		float u = float(i+random_float(&local_rand_state)) / max_x;
		float v = float(j+random_float(&local_rand_state)) / max_y;
		
		ray r = (*cam)->get_ray(rand_state, u,v);
		col += color_f(r, world, &local_rand_state, max_depth, back);
	}

	fb[pixel_index] = col/float(ns);
}




void draw(scene& curr_scene, render_settings settings) {
	std::cerr << "Allocating Frame Buffer" << std::flush;
	//Frame buffer (holds the image in the GPU)
	vec3 *fb;
	const size_t fb_size = settings.num_pixels*sizeof(vec3);	
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));	//allocating the frame buffer on the GPU
	
	
	
	
	//Render to the frame buffer
	dim3 blocks(settings.image_width/settings.threads_x+1, settings.image_height/settings.threads_y+1);	//making the frame buffer exceptionally long to combine the multiple frame buffers
	dim3 threads(settings.threads_x, settings.threads_y);
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, settings.num_pixels*sizeof(curandState) ));



	std::cerr << "\rIntialising the render        " << std::flush;
	render_init<<<blocks, threads>>>(settings.image_width, settings.image_height, d_rand_state);		//initialising the render -- currently just setting up the random numbers
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	std::cerr << "\rCreating temp dir             " << std::flush;
	//Create a random name for the temp dir
	std::string rand_str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
	std::random_device rd;	
	std::mt19937 generator(rd());

	std::shuffle(rand_str.begin(), rand_str.end(), generator);

	const std::string temp_file_dir = "./" +  rand_str.substr(0, 32);

	mkdir(temp_file_dir.c_str(), 0777);
	for (int i = 0; i < settings.no_fb; i++) {
		std::cerr << "\rRendering to frame buffer " << i+1 << "/" << settings.no_fb << "         "  << std::flush;
		render<<<blocks, threads>>>(fb, settings.image_width, settings.image_height, settings.samples_per_pixel_per_fb,	//render is a function defined above
						curr_scene.d_camera,
						d_rand_state,
						curr_scene.d_world, settings.max_depth, i,
						curr_scene.background);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tells the CPU that the GPU is done rendering
		write_frame_buffer(temp_file_dir + "/" + std::to_string(i) + ".ppm", fb, settings.image_width, settings.image_height);
	}

	checkCudaErrors(cudaFree(fb));


	std::cerr << "\rAveraging Frame Buffers         " << std::flush;
	average_images(temp_file_dir, "image.ppm");

	std::cerr << "\rConverting image to png" << std::flush;
	system("./to_png.sh");

	std::cerr << "\rDeleting temp files" << std::flush;
	remove("image.ppm");
	std::filesystem::path pathToDelete(temp_file_dir);
	remove_all(pathToDelete);

}

