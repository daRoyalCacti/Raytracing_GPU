#pragma once

#include "vec3.h"

#include <iostream>

void write_color(std::ostream &out, const color pixel_color, const int samples_per_pixel) {
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	//Divide the color by the number of samples
	//the component of each color is added to at each iteratio
	//so this acts as an averge
	const auto scale = 1.0 / samples_per_pixel;
	//gamma correcting using "gamma 2"
	//i.e. raising the color the power of 1/gamma = 1/2
	r = sqrt(scale * r);
	b = sqrt(scale * b);
	g = sqrt(scale * g);

	//Write the color
	//color is scaled to be in [0, 255]
	out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';
}

void write_frame_buffer(std::ostream &out, const color* f, const int width, const int height) {
	out << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height-1; j>=0; j--) 
		for (int i = 0; i < width; i++) {
			const size_t pixel_index = j*width + i;

			auto r = f[pixel_index].x();
			auto g = f[pixel_index].y();
			auto b = f[pixel_index].z();

			//gamma correcting using "gamma 2"
			//i.e. raising the color to the power of 1/gamma = 1/2
			r = sqrt(r);
			g = sqrt(g);
			b = sqrt(b);

			//write the color scaled to be in [0, 255]
			out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
				<< static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
				<< static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';

						
			/*const int ir = int(255.99*fb[pixel_index].x() );
			const int ig = int(255.99*fb[pixel_index].y() );
			const int ib = int(255.99*fb[pixel_index].z() );

			std::cout << ir << " " << ig << " " << ib << "\n";*/
		}

}
