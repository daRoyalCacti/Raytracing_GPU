#pragma once

#include "hittable.h"
#include "vec3.h"
#include "material.h"

struct sphere : public hittable {
	point3 center;
	float radius;
	material* mat_ptr;

	__device__ sphere() {}
	__device__ sphere(const point3 cen, const float r, material* m): center(cen), radius(r), mat_ptr(m) {}

	__device__ virtual bool hit(const ray&r, const float t_min, const float t_max, hit_record& rec, curandState* s) const override;
	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override;

	private:
	__device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
		//p: a given point on a unit sphere
		//u: normalised angle around Y axis (starting from x=-1)
		//v: normalised angle around Z axis (from Y=-1 to Y=1)
		// - normalised means in [0,1] as is standard for texture coordinates

		const float theta = acosf(-p.y());			//theta and phi in standard spherical coordinates
		const float phi = atan2f(-p.z(), p.x()) + pi;	//techically phi = atan2(p.z, -p.x) but this is discontinuous
								// - this uses atan2(a, b) = atan2(-a,-b) + pi which is continuous
								// - atan2(a,b) = atan(a/b)

		u = phi / (2 * pi);	//simple normalisation
		v = theta / pi;
	}
};

__device__ bool sphere::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState* s) const {
	//using the quadratic equation to find if (and when) 'ray' collides with sphere centred at 'center' with radius 'radius'
	
	const vec3 oc = r.origin() - center;	
	const float a = r.direction().length_squared();
	const float half_b =  dot(oc, r.direction());	//uses half b to simplify the quadratic equation
	const float c = oc.length_squared() - radius * radius;
	const float discriminant = half_b*half_b - a*c;

	if (discriminant < 0) return false;	//if there is no collision
	const float sqrtd = sqrtf(discriminant);

	//Find the nearest root that lies in the acceptable range.
	float root = (-half_b - sqrtd) / a;	//first root
	if (root < t_min || t_max < root) {	//if the first root is ouside of the accepctable range
		//if true, check the second root
		root = (-half_b - sqrtd) / a;
		if (root < t_min || t_max < root)
			//if the second root is ouside of the range, there is no hit
			return false;
	}

	//if the first root was accpetable, root is the first root
	//if the first root was not acceptable, root is the second root
	
	rec.t = root;	//root is finding time
	rec.p = r.at(rec.t);
	const vec3 outward_normal = (rec.p - center) / radius;	//a normal vector is just a point on the sphere less the center
								//dividing by radius to make it normalised
	rec.set_face_normal(r, outward_normal);
	//rec.normal = (r.at(root) - center)/radius * 1.5;
	rec.mat_ptr = mat_ptr;

	get_sphere_uv(outward_normal, rec.u, rec.v);	//setting the texture coordinates
							//outward_normal is technical a vec3 not a point3 but they are the same thing
							// - it points to the correct position on a unit sphere

	return true;	//the ray collides with the sphere
}

__device__ bool sphere::bounding_box(const float time0, const float time1, aabb& output_box) const {
	output_box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
	return true;
}
