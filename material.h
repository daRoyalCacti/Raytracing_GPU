#pragma once

#include "common.h"

#include "texture.h"

struct hit_record;

struct material {
	__device__ virtual bool scatter(const ray& ray_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *s) const = 0;	
	__device__ virtual color emitted(const float u, const float v, const point3& p) const {
		return color(0, 0, 0);
	}
};

struct lambertian : public material {
	texturez *albedo;
	
	__device__ lambertian(const color& a) {
		albedo = new solid_color(a);
	}
	__device__ lambertian(texturez* a) : albedo(a) {}


	__device__ virtual bool scatter(const ray& ray_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *s) const override {
		vec3 scatter_direction = rec.normal + random_unit_vector(s);
		
		//Catch degenerate scattering direction
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

		scattered = ray(rec.p, scatter_direction, ray_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}
};


struct metal : public material {
	texturez *albedo;
	float fuzz;	//how much light spreads out on collison
			//fuzz = 0 for perfect reflections
			//fuzz = 1 for very fuzzy reflections

	__device__ metal(const color& a, const float f = 0) : fuzz(f) {
		albedo = new solid_color(a);
	}
	__device__ metal(texturez *a) : albedo(a) {}

	__device__ virtual bool scatter(const ray& ray_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *s) const override {
		vec3 reflected = reflect(unit_vector(ray_in.direction()), rec.normal);	//the incomming ray reflected about the normal
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(s), ray_in.time());	//the scattered ray
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return (dot(scattered.direction(), rec.normal) > 0);	//making sure scattering not oppsoing the normal
	}
};


struct dielectric : public material {
	float ir;	//index of refraction of the material

	__device__ dielectric(const float index_of_refraction) : ir(index_of_refraction) {}

	__device__ virtual bool scatter(const ray& ray_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *s) const override {
		attenuation = color(1.0, 1.0, 1.0);	//material should be clear so white is a good choice for the color (absorbs nothing)
		const float refraction_ratio = rec.front_face ? (1.0f/ir) : ir;	//refraction ratio = (refractive index of incident material) / (refrative index of transmitted material)
										//assuiming that the incident material is air, and the refractive index of air is 1
										//this means 'refractive index of incident material' = 1
										// - that is assuming the ray was initially in air and then collides with the object
										// - if the ray was in the object and collided with air 'refractive index of transmitted material' = 1
										//Therefore, if the ray is initially in air, refraction ratio = 1/ir
										//else refraction ratio = ir
										//if front_face is true, then the ray opposes the normal vector
										// - i.e. the light is moving towards the material
		const vec3 unit_direction = unit_vector(ray_in.direction());
		
		//implementing total internal reflectiion
		const float cos_theta = min(dot(-unit_direction, rec.normal), 1.0f);
		const float sin_theta = sqrt(1.0f - cos_theta*cos_theta);
		
		const bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		vec3 direction;
		
		//uses the Schlick approximation to give reflectivity that varies with angle
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(s)) {
			//Must reflect
			direction = reflect(unit_direction, rec.normal);
		} else {
			//Can Refract (taking it as always refracting)
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}

		scattered = ray(rec.p, direction, ray_in.time());
		return true;
	}

	private:
	__device__ static float reflectance(const float cosine, const float ref_idx) {
		//use Schlick's approximation for reflectance
		const auto sqrt_r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
		const auto r0 = sqrt_r0 * sqrt_r0;
		return r0 + (1.0f-r0) * pow(1.0f-cosine, 5);
	}
};


struct diffuse_light : public material {
	texturez *emit;

	__device__ diffuse_light(texturez *a) : emit(a) {}
	__device__ diffuse_light(const color c) {
		emit = new solid_color(c);
	}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *s) const override {
		return false;
	}

	__device__ virtual color emitted(const float u, const float v, const point3& p) const {
		return emit->value(u, v, p);
	}
};

//for scattering
struct isotropic : public material {
	texturez *albedo;

	__device__ isotropic(const color c){
		albedo = new solid_color(c);
	}
	__device__ isotropic(texturez *a) : albedo(a) {}

	__device__ virtual bool scatter(const ray& ray_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *s) const override {
		scattered = ray(rec.p, random_in_unit_sphere(s), ray_in.time());	//pick a random direction for the ray to scatter
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}
};
