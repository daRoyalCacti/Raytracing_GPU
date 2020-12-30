#pragma once

#include "common.h"

class camera {
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	float lens_radius;
	float time0, time1;	//shutter open/close times
	
	vec3 u, v, w;

	public:
	__device__ camera() {}

	__device__ camera(const point3 lookfrom, const vec3 lookat, const vec3 vup, const float vfov, const float aspect_ratio, const float aperture, const float focus_dist,
			const float _time0, const float _time1) : time0(_time0), time1(_time1) {
		//vfov := vertical field of view in degrees
		//lookfrom := position of the camera
		//lookat := point for the camera to look at
		//vup := defines the remaining tilt of the camera - the up vector for the camera
		//focus_dist := distance from the camera that is in focus

		const float theta = degrees_to_radians(vfov);
		const float h = tan(theta/2.0f);

		const auto viewport_height = 2.0f * h;
		const auto viewport_width = aspect_ratio*viewport_height;

		w = unit_vector(lookfrom - lookat);	//vector pointing from camera to point to look at
		u = unit_vector(cross(vup, w));		//vector pointing to the right -- orthogonal to the up vector and w
		v = cross(w, u);			//the projection of the up vector on to the plane described by w (i.e. perpendicular to w)
		
		//uses a lense
		origin = lookfrom;	//camera is positioned at 'lookfrom'
		horizontal = focus_dist * viewport_width * u;	//defines the direction of horizontal through u, and how far horizontal to draw through viewport_width
		vertical = focus_dist * viewport_height * v;		//defines the direction of vertical through v, and how far vertical to draw through viewport_height
		lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - focus_dist*w;	//w pushes the corner back to make the camera still have the origin at 'origin'
											//focus_dist*w because this makes the light come from the lense of the camera
											// not just the camera itself
		lens_radius = aperture/2;
		//printf("%f\n", focus_dist);
		//printf("%f\n", lens_radius);
		//printf("%f\n", theta);
	}

	__device__ inline ray get_ray(curandState *state, const float s, const float t) const {
		//uses the thin lens approximation to generate depth of field
		//printf("z\n");
		const vec3 rd = lens_radius * random_in_unit_disk(state);	//randomness is required to get the blur
		//printf("y\n");
		const vec3 offset = u * rd.x() + v*rd.y();		//offset for where the light is coming from
		//printf("x\n");

		return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset, random_float(state, time0, time1));	//random time to simulate motion blur
	}
};
