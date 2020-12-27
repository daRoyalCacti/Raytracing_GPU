#pragma once

#include "hittable.h"
#include "common.h"

struct triangle : public hittable {
	material *mp;
	vec3 vertex0, vertex1, vertex2;	//position of vertex
	float u0, v0, u1, v1, u2, v2;	//texture coords for each vertex
	
	__device__ triangle() {}
	__device__ triangle(const vec3 v0, const vec3 v1, const vec3 v2, const float u0_, const float v0_, const float u1_, const float v1_, const float u2_, const float v2_,  material *mat)
		: vertex0(v0), vertex1(v1), vertex2(v2), u0(u0_), v0(v0_), u1(u1_), v1(v1_), u2(u2_), v2(v2_),  mp(mat) {};
	
	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *state) const override;

	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		//finding the min and max of each coordinate
		float min_x = vertex0.x(), min_y = vertex0.y(), min_z = vertex0.z();
		float max_x = vertex0.x(), max_y = vertex0.y(), max_z = vertex0.z();

		if (vertex1.x() < min_x)
			min_x = vertex1.x();
		if (vertex1.y() < min_y)
			min_y = vertex1.y();
		if (vertex1.z() < min_z)
			min_z = vertex1.z();

		if (vertex2.x() < min_x)
			min_x = vertex2.x();
		if (vertex2.y() < min_y)
			min_y = vertex2.y();
		if (vertex2.z() < min_z)
			min_z = vertex2.z();

		//creating the bounding box
		output_box = aabb(vec3(min_x, min_y, min_z), vec3(max_x, max_y, max_z) );
		return true;
	}
};

__device__ bool triangle::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *state) const {
	//using the Moller-Trumbore intersection algorithm
	//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	
	const float epsilon = 0.0000001;
	vec3 edge1, edge2, h, s, q;
	float a, f, u, v;
	
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;

	h = cross(r.dir, edge2);
        a = dot(edge1, h);

	if (a > -epsilon && a < epsilon)	//ray is parallel to triangle
		return false;

	f = 1.0 / a;
	s = r.orig - vertex0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f)
		return false;

	q = cross(s, edge1);
	v = f * dot(r.dir, q);

	if (v < 0.0f | u + v > 1.0f)
		return false;

	//computing the time of intersection
	const float t = f * dot(edge2, q);

	if (t < t_min || t > t_max)	//time of intersection falls outside of time range considered
		return false;

	rec.t = t;

	rec.set_face_normal(r, cross(edge1, edge2) );
	rec.mat_ptr = mp;
	rec.p = r.at(t);

	//finding the uv coordinates using barycentric coordinates
	//https://computergraphics.stackexchange.com/questions/1866/how-to-map-square-texture-to-triangle

	//first rotating the coordinates to get a "2D" triangle
	//https://math.stackexchange.com/questions/174598/3d-to-2d-rotation-matrix
	// - think its creating an orthonormal basis on the plane that the triangle lines in
	const vec3 S = edge1 / edge1.length();								//regular Gram-Schmidt
	const vec3 T = cross(S,  cross(S,  edge2) )  / cross(S,  cross(S,  edge2) ).length() ;		//likely a special case of Gram-Schmidt

	//creating A,B,C vectors from link about UV coords
	// - using the decomposition theorm for orthonormal bases
	// - a vector V, can be represented as <V, e1>e1 + <V, e2>e2
	// - here taking e1 = (1,0) and e2 = (0,1)
	// - This includes a horizontal shift to put vertex0 at the origin 
	const float A_x = dot(S, (vertex0 - vertex0));
	const float A_y = dot(T, (vertex0 - vertex0));

	const float B_x = dot(S, (vertex1 - vertex0));
	const float B_y = dot(T, (vertex1 - vertex0));

	const float C_x = dot(S, (vertex2 - vertex0));
	const float C_y = dot(T, (vertex2 - vertex0));

	const float P_x = dot(S, (rec.p - vertex0));
	const float P_y = dot(T, (rec.p - vertex0));

	const float Bary0 = ( (B_y-C_y) * (P_x-C_x)  +  (C_x-B_x) * (P_y-C_y))    /   ( (B_y-C_y) * (A_x-C_x)  +  (C_x-B_x) * (A_y-C_y) );
	const float Bary1 = ( (C_y-A_y) * (P_x-C_x)  +  (A_x-C_x) * (P_y-C_y))    /   ( (B_y-C_y) * (A_x-C_x)  +  (C_x-B_x) * (A_y-C_y) );
	const float Bary2 = 1 - Bary0 - Bary1;

	rec.u = Bary0*u0 + Bary1*u1 + Bary2*u2; 
	rec.v = Bary0*v0 + Bary1*v1 + Bary2*v2; 

	return true;	
}
