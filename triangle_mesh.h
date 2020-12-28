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

	__device__ triangle_mesh(hittable** triangles, int* num, const float time0, const float time1, curandState *s, int* offset, int id) {
		n = num[id];
		tris = new bvh_node(triangles+ offset[id], num[id], time0, time1, s);
	}

	/*__device__ triangle_mesh(float* verts, int num_v, unsigned* inds, int num_i,  float* uvs, int num_uvs,  unsigned tex_id, hittable** temp_hittables) {
		for (int i = 0; i < 	
	}*/

	
	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const override {
		return tris->hit(r, t_min, t_max, rec, s);
	}

	__device__ virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		return tris->bounding_box(time0, time1, output_box);	
	}
};




void processMesh(aiMesh *mesh, const aiScene *scene, std::vector<float> vertices, std::vector<unsigned> indices, std::vector<float> uvs, std::vector<std::string> tex_paths) {
	
	for (unsigned i = 0; i < mesh->mNumVertices; i++) {
		//process vertex positions and texture coordinates
		//will also do normals here when they get added

		vertices.push_back(mesh->mVertices[i].x);
		vertices.push_back(mesh->mVertices[i].y);
		vertices.push_back(mesh->mVertices[i].z);
		
		if (mesh->mTextureCoords[0]) {	//does the mesh contiain texture coords
			uvs.push_back(mesh->mTextureCoords[0][i].x);
			uvs.push_back(mesh->mTextureCoords[0][i].y);
		} else {
			uvs.push_back(0.0f);
			uvs.push_back(0.0f);
		}
	}

	for (unsigned i = 0; i < mesh->mNumFaces; i++) {
		//assimp defines eah mesh as having an array of faces
		// - aiProcess_Triangulate means these faces are always triangles
		//Iterate over all the faces and store the face's indices

		aiFace face = mesh->mFaces[i];
		for (unsigned j = 0; j < face.mNumIndices; j++) {
			indices.push_back(face.mIndices[j]);
		}
	}

	if (mesh->mMaterialIndex >= 0) {
		//currently only processes diffuse textures

		//retribve the aiMaterial object from the scene
		aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
		//saying that want all diffuse textures
		const aiTextureType type = aiTextureType_DIFFUSE;
	
		//actually loading the textures
		for (unsigned i = 0; i < mat->GetTextureCount(type); i++) {
			aiString str;
			mat->GetTexture(type, i , &str);
			std::string path = str.C_Str();
			tex_paths.push_back(path);	
		}
	}

}


void processNode(aiNode *node, const aiScene *scene, std::vector<float> vertices, std::vector<unsigned> indices, std::vector<float> uvs, std::vector<std::string> tex_paths) {
	//process all the node's meshes
	for (unsigned i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		processMesh(mesh, scene, vertices, indices, uvs, tex_paths);
	}

	//process the meshes of all the nodes children
	for (unsigned i = 0; i < node->mNumChildren; i++) {
		processNode(node->mChildren[i], scene, vertices, indices, uvs, tex_paths);
	}
}


void load_model(const std::string file_name, std::vector<float> vertices, std::vector<unsigned> indices, std::vector<float> uvs, std::vector<std::string> tex_paths) {
	//https://learnopengl.com/Model-Loading/Model	
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(file_name, aiProcess_Triangulate | aiProcess_GenNormals /*| aiProcess_FlipUVS*/);	
			//aiProcess_Triangulate tells assimp to make the model entirely out of triangles
			//aiProcess_GenNormals creates normal vectors for each vertex
			//aiProcess_FlipUVS flips the texture coordinates on the y-axis
	
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "Assimp Error:\n\t" << importer.GetErrorString() << std::endl;
		return;
	}

	processNode(scene->mRootNode, scene, vertices, indices, uvs, tex_paths);
}



__global__ void create_meshes_d(unsigned char* imdata, int *widths, int *heights, int* bytes_per_pixels) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism

	}
}

// need to also take some input that will actually be passed on the main scene creation
// - maybe input a structure to hold all of the triangles
// - and an array of offsets to know where each obj starts
// - also need size of obj to know where it ends
void create_meshes(std::vector<std::string> objs) {
	std::vector<std::vector<float>> vertices;
	std::vector<std::vector<unsigned>> indices;
	std::vector<std::vector<float>> uvs;
	std::vector<std::vector<std::string>> tex_paths;

	const int num = objs.size();
	vertices.resize(num);
	indices.resize(num);
	uvs.resize(num);
	tex_paths.resize(num);

	unsigned no_textures = 0;
	for (unsigned i = 0; i < objs.size(); i++) {
		load_model(objs[i], vertices[i], indices[i], uvs[i], tex_paths[i]);
		no_textures += tex_paths[i].size();
	}

	thrust::device_ptr<unsigned char> imdata;
	thrust::device_ptr<int> imwidths;
	thrust::device_ptr<int> imhs;
	thrust::device_ptr<int> imch;
	
	std::vector<const char*> image_locs;
	image_locs.resize(no_textures);
	unsigned counter = 0;
	for (unsigned i = 0; i < objs.size(); i++) {
		for (unsigned j = 0; j < tex_paths[i].size(); j++) {
			image_locs[counter++] = tex_paths[i][j].c_str();
		}
	}
	
	make_image(image_locs, imdata, imwidths, imhs, imch);


	/*have to copy host data over to device*/


	create_meshes_d<<<1,1>>>(thrust::raw_pointer_cast(imdata),
				thrust::raw_pointer_cast(imwidths),
				thrust::raw_pointer_cast(imhs),
				thrust::raw_pointer_cast(imch) );
}
