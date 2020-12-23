#include "vec3.h"
#include "ray.h"
#include "common.h"

#include "hittable_list.h"

//for compiling purposes
#include "color.h"
#include "camera.h"

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


__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0, 1, 0)));
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0, 0, 1)));
		*d_world    = new hittable_list(d_list,2);
		*d_camera   = new camera(vec3(0,0,-3), vec3(0,0,0), vec3(0,1,0), 40, 16.0f/9.0f, 0.0f, 10.0f, 0, 1 );
	}
}




__device__ vec3 color_f(ray& r, hittable **world, curandState *local_rand_state, int depth, vec3 background) {
	ray cur_ray = r;
	//const vec3 background(0.7f, 0.8f, 1.0f);
	color cur_attenuation(1,1,1);
	color cur_col(1,1,1);

	for (int i = 0; i < depth; i++) {
		//printf("%i\n", i);
		hit_record rec;

		//printf("aaa\n");
		if (!(*world)->hit(cur_ray, 0.001f, infinity, rec, local_rand_state)) 
			return cur_attenuation*background;
		//printf("bbb\n");

		ray scattered;
		color attenuation;
		const color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
		
		//printf("ccc\n");
		if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
			//printf("ddd\n");
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera **cam, curandState *rand_state,  hittable **world, int max_depth, int id, color back) {
	//max_x for size of total image
	//max_x2 for size of 1 frame buffer

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;	//if trying the work with more values than wanted
	int pixel_index = j*max_x + i;

	curandState local_rand_state = rand_state[(id*pixel_index + id)%(max_y*max_x)];
	vec3 col(0,0,0);
	
	for(int s=0; s < ns; s++) {
		//printf("%i\n", s);
		float u = float(i +random_float(&local_rand_state)) / max_x;
		float v = float(j+random_float(&local_rand_state)) / max_y;
		
		ray r = (*cam)->get_ray(rand_state, u,v);
		col += color_f(r, world, &local_rand_state, max_depth, back);
	}

	fb[pixel_index] = col/float(ns);
}

