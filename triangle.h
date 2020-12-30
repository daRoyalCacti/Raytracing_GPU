#pragma once

#include "hittable.h"
#include "common.h"

struct triangle : public hittable {
	material *mp;
	vec3 vertex0, vertex1, vertex2;	//position of vertex
	float u0, v0, u1, v1, u2, v2;	//texture coords for each vertex
	vec3 S, T;	//orthonormal vectors on the plane of the triangle
	vec3 edge1, edge2;	//edges of the triangle
	float A_x, A_y, B_x, B_y, C_x, C_y, bar_denom;	//helpful quantities for finding texture coords
	
	__device__ triangle() {}
	__device__ triangle(const vec3 v0, const vec3 v1, const vec3 v2, const float u0_, const float v0_, const float u1_, const float v1_, const float u2_, const float v2_,  material *mat)
		: vertex0(v0), vertex1(v1), vertex2(v2), u0(u0_), v0(v0_), u1(u1_), v1(v1_), u2(u2_), v2(v2_),  mp(mat) {
		
		edge1 = vertex1 - vertex0;
		edge2 = vertex2 - vertex0;

		//rotating the coordinates to get a "2D" triangle
		//https://math.stackexchange.com/questions/174598/3d-to-2d-rotation-matrix
		// - using gram-Schmidt
		S = edge1 / edge1.length();
				
		T = edge2 - dot(S, edge2)*S;
		T /= T.length();
		//cross(S,  cross(S,  edge2) )  / cross(S,  cross(S,  edge2) ).length() ;	

		//for finding the uv coordinates using barycentric coordinates
		//https://computergraphics.stackexchange.com/questions/1866/how-to-map-square-texture-to-triangle
		
		//creating A,B,C vectors from link about UV coords
		// - using the decomposition theorm for orthonormal bases
		project_to_plane(vertex0, A_x, A_y);
		project_to_plane(vertex1, B_x, B_y);
		project_to_plane(vertex2, C_x, C_y);

		bar_denom = (B_y-C_y) * (A_x-C_x)  +  (C_x-B_x) * (A_y-C_y);

		/*A_x = dot(S, (vertex0 - vertex0));
		A_y = dot(T, (vertex0 - vertex0));

		B_x = dot(S, (vertex1 - vertex0));
		B_y = dot(T, (vertex1 - vertex0));

		C_x = dot(S, (vertex2 - vertex0));
		C_y = dot(T, (vertex2 - vertex0));*/

		};

	__device__ inline void project_to_plane(const vec3 v, float &output_x, float &output_y) const {
		//for an orthonormal basis, the projection of v onto the basis S={e1,e2,e3,...} is given by
		// v = <v,e1>e1 + <v,e2>e2 + <v,e3>e3 + ...
		//then using the standard euclidian inner product
		//v = dot(v, S)S + dot(v,T)T
		// defining S to be the x direction and T to be the y direction gives rise to the formula below (dot products are invariant)
		output_x = dot(S, v);
		output_y = dot(T, v);

	}
	
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


		if (vertex1.x() > max_x)
			max_x = vertex1.x();
		if (vertex1.y() > max_y)
			max_y = vertex1.y();
		if (vertex1.z() > max_z)
			max_z = vertex1.z();

		if (vertex2.x() > max_x)
			max_x = vertex2.x();
		if (vertex2.y() > max_y)
			max_y = vertex2.y();
		if (vertex2.z() > max_z)
			max_z = vertex2.z();

		//creating the bounding box
		output_box = aabb(vec3(min_x, min_y, min_z), vec3(max_x, max_y, max_z) );
		return true;
	}

};

__device__ bool triangle::hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *state) const {
	//using the Moller-Trumbore intersection algorithm
	//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	
	const float epsilon = 0.0000001;
	vec3 h, s, q;
	float a, f, u, v;
	
	
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

	
	//const float P_x = dot(S, (rec.p - vertex0));
	//const float P_y = dot(T, (rec.p - vertex0));
	float P_x, P_y;
	project_to_plane(rec.p, P_x, P_y);

	const float Bary0 = ( (B_y-C_y) * (P_x-C_x)  +  (C_x-B_x) * (P_y-C_y))    /   bar_denom;
	const float Bary1 = ( (C_y-A_y) * (P_x-C_x)  +  (A_x-C_x) * (P_y-C_y))    /   bar_denom;
	const float Bary2 = 1 - Bary0 - Bary1;

	rec.u = Bary0*u0 + Bary1*u1 + Bary2*u2; 
	rec.v = Bary0*v0 + Bary1*v1 + Bary2*v2; 

	return true;	
}
