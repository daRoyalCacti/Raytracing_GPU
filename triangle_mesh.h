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

	__device__ triangle_mesh(hittable** triangles, unsigned* num, const float time0, const float time1, curandState *s, int id) {
		//printf("va\n");
		n = num[id];
		
		int offset = 0;
		for (int i = 0; i < id; i++) {
			printf("%i\n", i);
			offset += num[i];
		}
		//printf("a\n");
		printf("Creating bvh_node");
		tris = new bvh_node(triangles+offset, num[id], time0, time1, s);
		//printf("b\n");
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




void processMesh(aiMesh *mesh, const aiScene *scene, std::vector<float> &vertices, std::vector<unsigned> &indices, std::vector<float> &uvs, std::vector<std::string> &tex_paths) {
	
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

			bool already_loaded = false;
			for (int i = 0; i < tex_paths.size(); i++) {
				if (path == tex_paths[i]) {
					already_loaded = true;
					break;
				}
			}
			if (!already_loaded) {
				tex_paths.push_back(path);	
			}

		}
	}

}


void processNode(aiNode *node, const aiScene *scene, std::vector<float> &vertices, std::vector<unsigned> &indices, std::vector<float> &uvs, std::vector<std::string> &tex_paths) {
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


void load_model(const std::string file_name, std::vector<float> &vertices, std::vector<unsigned> &indices, std::vector<float> &uvs, std::vector<std::string> &tex_paths) {
	//https://learnopengl.com/Model-Loading/Model	
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(file_name, aiProcess_Triangulate | aiProcess_GenNormals /*| aiProcess_FlipUVs*/);	
			//aiProcess_Triangulate tells assimp to make the model entirely out of triangles
			//aiProcess_GenNormals creates normal vectors for each vertex
			//aiProcess_FlipUVS flips the texture coordinates on the y-axis
	
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "Assimp Error:\n\t" << importer.GetErrorString() << std::endl;
		return;
	}

	processNode(scene->mRootNode, scene, vertices, indices, uvs, tex_paths);
}



__global__ void create_meshes_d(hittable** hits, unsigned* num_tris, unsigned num_objs, unsigned char* imdata, int *widths, int *heights, int* bytes_per_pixels, float* vertices, unsigned* indices, float* uvs) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
	
		unsigned hit_counter = 0, ind_counter = 0;
		unsigned v_offset = 0, u_offset = 0, u_offset_t;	//holds the total number of indices before the current obj
		image_texture* current_texture;	//the texture for the current obj
		lambertian* current_material;
		for (int i = 0; i < num_objs; i++) {
			current_texture = new image_texture(imdata, widths, heights, bytes_per_pixels, i);
			current_material = new lambertian(current_texture);
			for (int j = 0; j < num_tris[i]; j++) {
				//printf("j: %i/%i\n", j+1, num_tris[i]);
				hits[hit_counter++] = new triangle( 	vec3(vertices[ 3*indices[ind_counter] + v_offset],		//each element of indices refers to 1 vertex
									     vertices[ 3*indices[ind_counter] + v_offset + 1],	//each vertex has 3 elements - x,y,z
									     vertices[ 3*indices[ind_counter] + v_offset + 2]),	

									vec3(vertices[ 3*indices[ind_counter+1] + v_offset],
									     vertices[ 3*indices[ind_counter+1] + v_offset + 1],
									     vertices[ 3*indices[ind_counter+1] + v_offset + 2]),

									vec3(vertices[ 3*indices[ind_counter+2] + v_offset],
									     vertices[ 3*indices[ind_counter+2] + v_offset + 1],
									     vertices[ 3*indices[ind_counter+2] + v_offset + 2]),

									uvs[2*indices[ind_counter] + u_offset],
									uvs[2*indices[ind_counter] + u_offset+1],

									uvs[2*indices[ind_counter+1] + u_offset],
									uvs[2*indices[ind_counter+1] + u_offset+1],

									uvs[2*indices[ind_counter+2] + u_offset],
									uvs[2*indices[ind_counter+2] + u_offset+1],

									current_material);

				ind_counter += 3;
				u_offset_t += 2;
			}
			v_offset += ind_counter;//3*num_tris[i];
			u_offset += u_offset_t;
		}	
	}


}


__global__ void meshes_free(unsigned char* imdata, int *widths, int *heights, int* bytes_per_pixels, float* vertices, unsigned* indices, float* uvs) {
	delete [] imdata;
	delete [] widths;
	delete [] heights;
	delete [] bytes_per_pixels;
	delete [] vertices;
	delete [] indices;
	delete [] uvs;
}

// need to also take some input that will actually be passed on the main scene creation
// - maybe input a structure to hold all of the triangles
// - and an array of offsets to know where each obj starts
// - also need size of obj to know where it ends
void create_meshes(std::vector<std::string> objs, hittable** &hits, thrust::device_ptr<unsigned> &num_data, int& size) {
	//size is the size in bytes of all the meshes
	//num_data is the number of triangles for a mesh
	
	
	
	std::vector<std::string> file_dirs;
	file_dirs.resize(objs.size());
	//figuring out the file directory for a particular obj
	
	//const char *to_find = "/";
	

	std::cout << "finding directories" << std::endl;
	for (int i = 0; i < objs.size(); i++) {
		file_dirs[i] = "";

		std::string temp_string = objs[i];
		reverse(temp_string.begin(), temp_string.end());

		int char_until_back = temp_string.find("/");	//characters until forwardslash

		file_dirs[i].append(temp_string, char_until_back, temp_string.size() - char_until_back );
		reverse(file_dirs[i].begin(), file_dirs[i].end() );
	}

	//the structures the meshes should be read into
	std::vector<std::vector<float>> vertices;
	std::vector<std::vector<unsigned>> indices;
	std::vector<std::vector<float>> uvs;
	std::vector<std::vector<std::string>> tex_paths;

	const int num = objs.size();
	vertices.resize(num);
	indices.resize(num);
	uvs.resize(num);
	tex_paths.resize(num);

	std::vector<unsigned> num_t;	//to hold the number of triangles for a mesh on the host
	num_t.resize(num);

	unsigned no_vert = 0, no_ind = 0, no_uv = 0;

	//keeping track of the size of each constituent of the mesh
	std::vector<unsigned> num_vert, num_ind, num_uv;
	num_vert.resize(num);
	num_ind.resize(num);
	num_uv.resize(num);

	//to convert the  vector<vector> of verticles to a single vector for passing to the gpu
	std::vector<float> all_vert, all_uv;
	std::vector<unsigned> all_ind;

	
	int num_triangles = 0;

	std::cout << "making the meshes" << std::endl;
	for (unsigned i = 0; i < objs.size(); i++) {
		load_model(objs[i], vertices[i], indices[i], uvs[i], tex_paths[i]);
		num_t[i] = indices[i].size()/3;	//3 indices make a triangle
		num_triangles += num_t[i];
		no_vert += vertices[i].size();
		no_ind += indices[i].size();
		no_uv += uvs[i].size();

		num_vert[i] = vertices[i].size();
		num_ind[i] = indices[i].size();
		num_uv[i] = uvs[i].size();

		for (const auto& v : vertices[i]) {
			all_vert.push_back(v);
		}

		for (const auto& n : indices[i]) {
			all_ind.push_back(n);
		}

		for (const auto& u : uvs[i]) {
			all_uv.push_back(u);
		}


		if (tex_paths[i].size() == 0) {
			std::cerr << "Meshes without textures are not supported" << std::endl;
			return;
		} else if (tex_paths[i].size() > 1) {
			std::cerr << "Meshes with more than 1 texture are not supported" << std::endl;
			return;
		}
	}
	
	//copying the data to the gpu
	thrust::device_ptr<float> vert_data;
	thrust::device_ptr<float> uv_data;
	thrust::device_ptr<unsigned> ind_data;

	upload_to_device(vert_data, all_vert);	//upload_to_device defined in common.h
	upload_to_device(ind_data, all_ind);
	upload_to_device(uv_data, all_uv);
	upload_to_device(num_data, num_t);



	std::cout << "making images" << std::endl;
	//reading in the images
	thrust::device_ptr<unsigned char> imdata;
	thrust::device_ptr<int> imwidths;
	thrust::device_ptr<int> imhs;
	thrust::device_ptr<int> imch;
	
	std::vector<const char*> image_locs;
	image_locs.resize(objs.size());
	for (unsigned i = 0; i < objs.size(); i++) {
		image_locs[i] = (file_dirs[i].append(tex_paths[i][0])).c_str();
	}

	//creating the images
	make_image(image_locs, imdata, imwidths, imhs, imch);

	checkCudaErrors(cudaMalloc((void**)&hits, num_triangles*sizeof(triangle*) ));

	
	std::cout << "main" << std::endl;

	create_meshes_d<<<1, 1>>>(hits, thrust::raw_pointer_cast(num_data), 
				//thrust::raw_pointer_cast(num_obj),
				objs.size(),
				thrust::raw_pointer_cast(imdata),
				thrust::raw_pointer_cast(imwidths),
				thrust::raw_pointer_cast(imhs),
				thrust::raw_pointer_cast(imch),
		       		
				thrust::raw_pointer_cast(vert_data),
				thrust::raw_pointer_cast(ind_data),
				thrust::raw_pointer_cast(uv_data));
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the meshes are created

	std::cout << "Cleanup" << std::endl;
	/*meshes_free<<<1, 1>>>(thrust::raw_pointer_cast(imdata),
				thrust::raw_pointer_cast(imwidths),
				thrust::raw_pointer_cast(imhs),
				thrust::raw_pointer_cast(imch),
		       		
				thrust::raw_pointer_cast(vert_data),
				thrust::raw_pointer_cast(ind_data),
				thrust::raw_pointer_cast(uv_data));


	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(imdata) ));
	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(imwidths) ));
	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(imhs) ));
	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(imch) ));
	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(vert_data) ));
	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(ind_data) ));
	checkCudaErrors(cudaFree(thrust::raw_pointer_cast(uv_data) ));*/

	size = 0;
	for (int i = 0; i < objs.size(); i++) {
		size += size_of_bvh(num_t[i]);
	}
	/*size += num_triangles*sizeof(hittable*);
	size += num_t.size() * sizeof(unsigned);*/
}
