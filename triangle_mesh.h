#pragma once

#include "triangle.h"

#include <vector>
#include <string>

//for loading a model

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"


struct triangle_mesh : public hittable {
	material *mp;
	//triangle* tris;	//all triangles
	bvh_node* tris;
	int n;		//number of triangles
	
	__device__ triangle_mesh() {}
	__device__ triangle_mesh(hittable** triangles, int num, const float time0, const float time1, curandState *s) {
		n = num;
		tris = new bvh_node(triangles, num, time0, time1, s);
	}
	
	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const override {
		return tris->hit(r, t_min, t_max, rec, s);
	}

	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		return tris->bounding_box(time0, time1, output_box);	
	}
};



void load_model(const std::string file_name, std::vector<float> vertices, std::vector<float> indices, std::vector<float> uvs) {
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(file_name, aiProcess_Triangulate | aiProcess_GenNormals /*| aiProcess_FlipUVS*/);	
			//aiProcess_Triangulate tells assimp to make the model entirely out of triangles
			//aiProcess_GenNormals creates normal vectors for each vertex
			//aiProcess_FlipUVS flips the texture coordinates on the y-axis
	
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "Assimp Error:\n\t" << importer.GetErrorString() << std::endl;
		return;
	}

}
