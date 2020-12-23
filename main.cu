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

#include <filesystem>

#include "render.h"

//for creating a directory
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 



int main() {
	//start timing
	const auto start = std::chrono::system_clock::now();
	const std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	std::cerr << "Computation started at " << std::ctime(&start_time);

	
	std::cerr << "Creating World                " << std::flush;
	first_scene curr_scene;

	render_settings settings;

	settings.image_width = 1200;
	settings.no_fb = 10;
	settings.samples_per_pixel_per_fb = 100;
	settings.aspect_ratio = curr_scene.aspect;

	settings.calc_all();




	std::cerr << "\rGenerating a " << settings.image_width << "x" << settings.image_height << " image with " << settings.rays_per_pixel << " rays per pixel" << std::flush;
	std::cerr << " using " << settings.threads_x << "x" << settings.threads_y << " blocks and " << settings.no_fb << " frame buffers.\n";



	std::cerr << "Allocating Frame Buffer" << std::flush;
	//Frame buffer (holds the image in the GPU)
	vec3 *fb;
	const size_t fb_size = settings.num_pixels*sizeof(vec3)/**settings.no_fb*/;	
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));	//allocating the frame buffer on the GPU
	
	
	
	
	//Render to the frame buffer
	dim3 blocks(settings.image_width/**settings.no_fb*//settings.threads_x+1, settings.image_height/settings.threads_y+1);	//making the frame buffer exceptionally long to combine the multiple frame buffers
	dim3 threads(settings.threads_x, settings.threads_y);
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, settings.num_pixels/**no_fb*/*sizeof(curandState) ));



	std::cerr << "\rIntialising the render        " << std::flush;
	render_init<<<blocks, threads>>>(settings.image_width, settings.image_height, d_rand_state);		//initialising the render -- currently just setting up the random numbers
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	

	mkdir("./output", 0777);
	for (int i = 0; i < settings.no_fb; i++) {
		std::cerr << "\rRendering to frame buffer " << i+1 << "/" << settings.no_fb << "         "  << std::flush;
		render<<<blocks, threads>>>(fb, settings.image_width, settings.image_height, settings.samples_per_pixel_per_fb,	//render is a function defined above
						curr_scene.d_camera,
						d_rand_state,
						curr_scene.d_world, settings.max_depth, i,
						curr_scene.background);		
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());	//tells the CPU that the GPU is done rendering
		write_frame_buffer("output/" + std::to_string(i) + ".ppm", fb, settings.image_width, settings.image_height);
	}

	
	std::cerr << "\rAveraging Frame Buffers         " << std::flush;
	average_images("./output", "image.ppm");

	std::cerr << "\rConverting image to png" << std::endl;
	system("./to_png.sh");

	std::cerr << "Deleting temp files" << std::endl;
	remove("image.ppm");
	std::filesystem::path pathToDelete("./output");
	remove_all(pathToDelete);

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

	return 0;
}
