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
