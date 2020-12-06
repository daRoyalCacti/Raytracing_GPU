#include <iostream>
#include <chrono>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "common.h"

#include "hittable_list.h"



__global__ void create_world(hittable **d_list, hittable **d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//not need for parallism
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5);
		*(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
		*d_world    = new hittable_list(d_list,2);
	}
}

__global__ void free_world(hittable ** d_list, hittable **d_world) {
	delete *(d_list);
	delete *(d_list+1);
	delete *d_world;
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - 4.0f*a*c;
	return (discriminant > 0.0f);
}

__device__ vec3 color_f(const ray& r, hittable **world) {
	hit_record rec;
	if ((*world)->hit(r, 0.001, infinity, rec)) {
		return 0.5f*vec3(rec.normal.x()+1, rec.normal.y() + 1.0f, rec.normal.z()+1.0f);
	} else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
	}
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, curandState *rand_state,  hittable **world) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;	//if trying the work with more values than wanted
	int pixel_index = j*max_x + i;

	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0,0,0);
	
	for(int s=0; s < ns; s++) {
		float u = float(i+random_float(&local_rand_state)) / max_x;
		float v = float(j+random_float(&local_rand_state)) / max_y;
		ray r(origin,lower_left_corner + u*horizontal + v*vertical);
		col += color_f(r, world);
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

	//Frame buffer (holds the image in the GPU)
	vec3 *fb;
	const size_t fb_size = num_pixels*sizeof(vec3);	
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));	//allocating the frame buffer on the GPU

	//make our world of hittables
	hittable **d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*) ));	//2 because 2 hittables
	hittable **d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable *) ));
	create_world<<<1,1>>>(d_list, d_world);		//create_world is defined above
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the world is created

	//Render to the frame buffer
	dim3 blocks(nx/tx+1, ny/ty+1);
	dim3 threads(tx, ty);
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState) ));

	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);		//initialising the render -- currently just setting up the random numbers
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>>(fb, nx, ny, ns,	//render is a function defined above
					vec3(-2.0, -1.0, -1.0),
					vec3(4.0, 0.0, 0.0),
					vec3(0.0, 2.0, 0.0),
					vec3(0.0, 0.0, 0.0),
					d_rand_state,
					d_world);		
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//lets the CUP that the GPU is done rendering

	//Ouput FB as Image
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny-1; j>=0; j--) 
		for (int i = 0; i < nx; i++) {
			const size_t pixel_index = j*nx + i;
			
			const int ir = int(255.99*fb[pixel_index].x() );
			const int ig = int(255.99*fb[pixel_index].y() );
			const int ib = int(255.99*fb[pixel_index].z() );

			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	
	//clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1,1>>>(d_list,d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(fb));

	return 0;
}
