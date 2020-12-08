#pragma once

#include "common.h"

#include "perlin.h"
#include "stb_image_ne.h"

struct texturez {
	//int n;
	//__device__ texturez() {}
	//__device__ texturez(int n_) : n(n_) {}
	__device__ virtual color value(const float u, const float v, const point3& p) const {return color(0,0,0);}
};

struct solid_color : public texturez {
	__device__ solid_color() {}
	__device__ solid_color(const color c) : color_value(c) {}
	__device__ solid_color(const float red, const float green, const float blue) : solid_color(color(red, green, blue)) {}
	
	__device__ virtual color value(const float u, const float v, const vec3& p) const override {
		return color_value;
	}

	private:
	color color_value;
};


struct checker_texture : public texturez {
	texturez *odd;
	texturez *even;

	__device__ checker_texture() {}
	__device__ checker_texture(texturez *_even, texturez *_odd) : even(_even), odd(_odd) {}
	__device__ checker_texture(color c1, color c2)  {//: even(solid_color(c1)), odd(solid_color(c2)) {}
		even = new solid_color(c1);
		odd  = new solid_color(c2);
	}

	__device__ virtual color value(const float u, const float v, const point3& p) const override {
		const auto sines = sin(10*p.x()) * sin(10*p.y()) * sin(10*p.z());	//essentially a 4D sine wave

		if(sines < 0)	//sine is periodic. Having a different texture for if sine is positive or negative will given distinct regions of different colors
			return odd->value(u, v, p);
		else 
			return even->value(u, v, p);

	}
};


struct noise_texture : public texturez {
	perlin noise;
	float scale;	//how detailed the noise is, bigger number := more noise

	__device__ noise_texture(curandState *s, const float sc = 1.0) : scale(sc) {
		noise = perlin(s);
	}

	__device__ virtual color value(const float u, const float v, const point3& p) const override {
		return color(1,1,1) *0.5 *(1.0 + noise.noise(scale*p));	//creates a gray color
									//needs to be scaled to go between 0 and 1 else the gamma correcting function will return NaN's
									// (sqrt of a negative number)
	}
};


struct turbulent_texture : public texturez {
	perlin noise;
	float scale;	//how detailed the noise is, bigger number := more noise
	int depth;	//number of layers of noise

	__device__ turbulent_texture(curandState *s, const float sc = 1.0, const int dpt = 7) : scale(sc), depth(dpt) {
		noise = perlin(s);
	}
	
	__device__ virtual color value(const float u, const float v, const point3& p) const override {
		return color(1,1,1) * noise.turb(scale*p, depth);
	}
};


struct marble_texture : public texturez {
	perlin noise;
	float scale;	//how detailed the noise is, bigger number := more noise

	__device__ marble_texture(curandState *s, const float sc = 1.0) : scale(sc) {
		noise = perlin(s);
	}
	
	__device__ virtual color value(const float u, const float v, const point3& p) const override {
		return color(1,1,1) * 0.5 * (1 + sin(scale*p.z() + 10* noise.turb(scale*p, 7) ) );
	}
};





class image_texture : public texturez {
	unsigned char* data;	//the data read from file
	int width, height;	//the width and height of the image
	int bytes_per_scanline;

	public:
	const static int bytes_per_pixel = 3;

	__host__ __device__ image_texture() : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

	__host__ image_texture(const char* filename) {
		auto components_per_pixel = bytes_per_pixel;

		data = stbi_load(filename, &width, &height, &components_per_pixel, components_per_pixel);	//reading the data from disk

		if (!data) {	//file not read
			std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
			width = height =0;
		}
		bytes_per_scanline = bytes_per_pixel * width;
	}

	__host__ __device__ ~image_texture() {
		delete data;
	}

	__device__ virtual color value(const float u, const float v, const vec3& p) const override {
		if (data == nullptr)	//if not texture data, return cyan color
			return color(0, 1, 1);

		//Clamp input texture coordinates to [0,1]^2
		const auto uu = clamp(u, 0.0, 1.0);
		const auto vv = 1.0 - clamp(v, 0.0, 1.0);	//Flip v to image coordinates

		auto i = static_cast<int>(uu*width);
		auto j = static_cast<int>(vv*height);

		//Clamp integer mapping sicne actual coordinates should be less than 1.0
		if (i >= width)  i = width - 1;
		if (j >= height) j = height - 1;

		const auto color_scale = 1.0 / 255.0;	//to scale the input from [0,255] to [0,1]
		const auto pixel = data + j*bytes_per_scanline + i*bytes_per_pixel;	//the pixel at coordinates (u,v)

		return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
	}
};
