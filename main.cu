#include <iostream>
#include <chrono>

#include "scenes.h"
#include "render.h"


int main() {
	//start timing
	const auto start = std::chrono::system_clock::now();
	const std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	std::cerr << "Computation started at " << std::ctime(&start_time);

	
	std::cerr << "Creating World                " << std::flush;
	cornell_smoke_box_scene curr_scene;


	render_settings settings;

	settings.image_width = 1200;
	settings.no_fb = 50;
	settings.samples_per_pixel_per_fb = 1000;
	settings.aspect_ratio = curr_scene.aspect;

	settings.calc_all();



	std::cerr << "\rGenerating a " << settings.image_width << "x" << settings.image_height << " image with " << settings.rays_per_pixel << " rays per pixel" << std::flush;
	std::cerr << " using " << settings.threads_x << "x" << settings.threads_y << " blocks and " << settings.no_fb << " frame buffers.\n";


	draw(curr_scene, settings);


	//end timing
	const auto end = std::chrono::system_clock::now();
	const std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cerr << "\rComputation ended at " << std::ctime(&end_time) << "                ";

	const std::chrono::duration<double> elapsed_seconds = end - start;
	std::cerr << "Elapsed time: " << elapsed_seconds.count() << "s  or  " << elapsed_seconds.count() / 60.0f << "m  or  " << elapsed_seconds.count() / (60.0f * 60.0f) << "h\n";

	return 0;
}
