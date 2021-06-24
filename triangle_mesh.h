#pragma once

#include "triangle.h"

#include <vector>
#include <string>

//for loading a model

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"


//U should be a material
template <typename U>
struct triangle_mesh : public hittable {
	//material *mp;
	bvh_node<triangle<U>>* tris;
	int n;		//number of triangles
	
	triangle_mesh() {}
	triangle_mesh(triangle<U>** &triangles, int num, const float time0, const float time1) {
		n = num;
		tris = new bvh_node(triangles, num, time0, time1);
	}

	__host__ __device__ triangle_mesh(bvh_node<triangle<U>>* tris_, const int n_) {
		tris = tris_;
		n = n_;
	}



	
	__device__ virtual bool hit(const ray& r, const float t_min, const float t_max, hit_record& rec, curandState *s) const override {
		//printf("aa\n");
		return tris->hit(r, t_min, t_max, rec, s);
	}

	virtual bool bounding_box(const float time0, const float time1, aabb& output_box) const override {
		return tris->bounding_box(time0, time1, output_box);	
	}


	template <typename T>
	void cpy_constit_d(T* d_ptr) const override {
		bvh_node<T>* tris_d;
		cudaMalloc((void**)&tris_d, sizeof(bvh_node<T> ) );
		checkCudaErrors(cudaMemcpy(tris_d, tris, sizeof(bvh_node<T>), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&(d_ptr->tris), &tris_d, sizeof(bvh_node<T>*), cudaMemcpyDefault)); 
		tris->cpy_constit_d(tris_d);

	}
};




void processMesh(aiMesh *mesh, const aiScene *scene, std::vector<float> &vertices, std::vector<unsigned> &indices, std::vector<float> &uvs, std::vector<std::string> &tex_paths, std::vector<float> &norms) {
	
	for (unsigned i = 0; i < mesh->mNumVertices; i++) {
		//process vertex positions, normals and texture coordinates

		vertices.push_back(mesh->mVertices[i].x);
		vertices.push_back(mesh->mVertices[i].y);
		vertices.push_back(mesh->mVertices[i].z);

		norms.push_back(mesh->mNormals[i].x);
		norms.push_back(mesh->mNormals[i].y);
		norms.push_back(mesh->mNormals[i].z);
		
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

		const aiFace face = mesh->mFaces[i];
		for (unsigned j = 0; j < face.mNumIndices; j++) {
			indices.push_back(face.mIndices[j]);
		}
	}


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


void processNode(aiNode *node, const aiScene *scene, std::vector<float> &vertices, std::vector<unsigned> &indices, std::vector<float> &uvs, std::vector<std::string> &tex_paths, std::vector<float> &norms) {
	//process all the node's meshes
	for (unsigned i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		processMesh(mesh, scene, vertices, indices, uvs, tex_paths, norms);
	}

	//process the meshes of all the nodes children
	for (unsigned i = 0; i < node->mNumChildren; i++) {
		processNode(node->mChildren[i], scene, vertices, indices, uvs, tex_paths, norms);
	}
}


triangle_mesh<lambertian<image_texture>>* generate_model(const std::string& file_name, const bool flip_uvs = false)  {
	//https://learnopengl.com/Model-Loading/Model	

	std::vector<float> vertices;
	std::vector<unsigned> indices;
	std::vector<float> uvs;
	std::vector<float> norms;
	std::vector<std::string> tex_paths;

	unsigned assimp_settings = aiProcess_Triangulate | aiProcess_GenNormals;


	if (flip_uvs)
		assimp_settings |= aiProcess_FlipUVs;


	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(file_name, assimp_settings);	
	//aiProcess_Triangulate tells assimp to make the model entirely out of triangles
	//aiProcess_GenNormals creates normal vectors for each vertex
	//aiProcess_FlipUVS flips the texture coordinates on the y-axis


	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "Assimp Error:\n\t" << importer.GetErrorString() << std::endl;
	}

	processNode(scene->mRootNode, scene, vertices, indices, uvs, tex_paths, norms);


	//turing the read in data into a triangle mesh
	triangle<lambertian<image_texture>>** triangles;	
	triangles = new triangle<lambertian<image_texture>>*[indices.size()/3];

	std::string file_dir = file_name.substr(0, file_name.find_last_of('/') );
	file_dir.append("/");


	auto current_material = new lambertian<image_texture>(new image_texture(file_dir.append(tex_paths[0]).c_str()) );
	//const auto current_texture = new image_texture(imdata, widths, heights, bytes_per_pixels, i);
	//const auto current_material = new lambertian(current_texture);
	for (int i = 0; i < indices.size(); i += 3) {
		triangles[i/3] = new triangle(	vec3(vertices[ 3*indices[i] ],		//each element of indices refers to 1 vertex
							vertices[ 3*indices[i]  + 1],	//each vertex has 3 elements - x,y,z
							vertices[ 3*indices[i]  + 2]),	

							vec3(vertices[ 3*indices[i+1] ],
							vertices[ 3*indices[i+1]  + 1],
							vertices[ 3*indices[i+1]  + 2]),

							vec3(vertices[ 3*indices[i+2] ],
							vertices[ 3*indices[i+2]  + 1],
							vertices[ 3*indices[i+2]  + 2]),

							vec3(norms[ 3*indices[i] ],		//each element of indices refers to 1 normal
							norms[ 3*indices[i]  + 1],		//each normal has 3 elements - x,y,z
							norms[ 3*indices[i]  + 2]),	

							vec3(norms[ 3*indices[i+1] ],
							norms[ 3*indices[i+1]  + 1],
							norms[ 3*indices[i+1]  + 2]),

							vec3(norms[ 3*indices[i+2] ],
							norms[ 3*indices[i+2]  + 1],
							norms[ 3*indices[i+2]  + 2]),


							uvs[2*indices[i] ],
							uvs[2*indices[i] +1],

							uvs[2*indices[i+1] ],
							uvs[2*indices[i+1] +1],

							uvs[2*indices[i+2] ],
							uvs[2*indices[i+2] +1],

							current_material);


	}

	return new triangle_mesh(triangles, indices.size()/3, 0, 1);

}

void load_model(const std::string file_name, std::vector<float> &vertices, std::vector<unsigned> &indices, std::vector<float> &uvs, std::vector<std::string> &tex_paths, std::vector<float> &norms) {
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

	processNode(scene->mRootNode, scene, vertices, indices, uvs, tex_paths, norms);
}



/*
__global__ void create_meshes_d(hittable** hits, unsigned* num_tris, unsigned num_objs, unsigned char* imdata, int *widths, int *heights, int* bytes_per_pixels, float* vertices, unsigned* indices, float* uvs, float* norms) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {	//no need for parallism
	
		unsigned hit_counter = 0, ind_counter = 0;
		unsigned v_offset = 0, u_offset = 0, u_offset_t;	//holds the total number of indices before the current obj
		image_texture* current_texture;	//the texture for the current obj
		lambertian* current_material;
		for (int i = 0; i < num_objs; i++) {
			current_texture = new image_texture(imdata, widths, heights, bytes_per_pixels, i);
			current_material = new lambertian(current_texture);
			for (int j = 0; j < num_tris[i]; j++) {
				hits[hit_counter++] = new triangle( 	vec3(vertices[ 3*indices[ind_counter] + v_offset],		//each element of indices refers to 1 vertex
									     vertices[ 3*indices[ind_counter] + v_offset + 1],	//each vertex has 3 elements - x,y,z
									     vertices[ 3*indices[ind_counter] + v_offset + 2]),	

									vec3(vertices[ 3*indices[ind_counter+1] + v_offset],
									     vertices[ 3*indices[ind_counter+1] + v_offset + 1],
									     vertices[ 3*indices[ind_counter+1] + v_offset + 2]),

									vec3(vertices[ 3*indices[ind_counter+2] + v_offset],
									     vertices[ 3*indices[ind_counter+2] + v_offset + 1],
									     vertices[ 3*indices[ind_counter+2] + v_offset + 2]),

									vec3(norms[ 3*indices[ind_counter] + v_offset],		//each element of indices refers to 1 normal
									     norms[ 3*indices[ind_counter] + v_offset + 1],		//each normal has 3 elements - x,y,z
									     norms[ 3*indices[ind_counter] + v_offset + 2]),	

									vec3(norms[ 3*indices[ind_counter+1] + v_offset],
									     norms[ 3*indices[ind_counter+1] + v_offset + 1],
									     norms[ 3*indices[ind_counter+1] + v_offset + 2]),

									vec3(norms[ 3*indices[ind_counter+2] + v_offset],
									     norms[ 3*indices[ind_counter+2] + v_offset + 1],
									     norms[ 3*indices[ind_counter+2] + v_offset + 2]),


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
			v_offset += ind_counter;
			u_offset += u_offset_t;
		}	
	}


}

*/

/*
void create_meshes(std::vector<std::string> objs, hittable** &hits, thrust::device_ptr<unsigned> &num_data, int& size) {
	//size is the size in bytes of all the meshes
	//num_data is the number of triangles for a mesh
	
	
	
	std::vector<std::string> file_dirs;
	file_dirs.resize(objs.size());
	//figuring out the file directory for a particular obj
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
	std::vector<std::vector<float>> norms;

	const int num = objs.size();
	vertices.resize(num);
	indices.resize(num);
	uvs.resize(num);
	tex_paths.resize(num);
	norms.resize(num);

	std::vector<unsigned> num_t;	//to hold the number of triangles for a mesh on the host
	num_t.resize(num);

	//to convert the  vector<vector> of verticles to a single vector for passing to the gpu
	std::vector<float> all_vert, all_uv, all_norm;
	std::vector<unsigned> all_ind;

	
	int num_triangles = 0;

	for (unsigned i = 0; i < objs.size(); i++) {
		load_model(objs[i], vertices[i], indices[i], uvs[i], tex_paths[i], norms[i]);
		num_t[i] = indices[i].size()/3;	//3 indices make a triangle
		num_triangles += num_t[i];

		for (const auto& v : vertices[i]) {
			all_vert.push_back(v);
		}

		for (const auto& n : indices[i]) {
			all_ind.push_back(n);
		}

		for (const auto& u : uvs[i]) {
			all_uv.push_back(u);
		}

		for (const auto& n : norms[i]) {
			all_norm.push_back(n);
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
	thrust::device_ptr<float> norm_data;

	upload_to_device(vert_data, all_vert);	//upload_to_device defined in common.h
	upload_to_device(ind_data, all_ind);
	upload_to_device(uv_data, all_uv);
	upload_to_device(num_data, num_t);
	upload_to_device(norm_data, all_norm);



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

	

	create_meshes_d<<<1, 1>>>(hits, thrust::raw_pointer_cast(num_data), 
				//thrust::raw_pointer_cast(num_obj),
				objs.size(),
				thrust::raw_pointer_cast(imdata),
				thrust::raw_pointer_cast(imwidths),
				thrust::raw_pointer_cast(imhs),
				thrust::raw_pointer_cast(imch),
		       		
				thrust::raw_pointer_cast(vert_data),
				thrust::raw_pointer_cast(ind_data),
				thrust::raw_pointer_cast(uv_data),
				thrust::raw_pointer_cast(norm_data));
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the meshes are created


	thrust::device_free(imdata);
	thrust::device_free(imwidths);
	thrust::device_free(imhs);
	thrust::device_free(imch);
	thrust::device_free(vert_data);
	thrust::device_free(ind_data);
	thrust::device_free(uv_data);
	thrust::device_free(norm_data);

	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());	//tell cpu the meshes are created

	size = 0;
	for (int i = 0; i < objs.size(); i++) {
		size += size_of_bvh(num_t[i]);
	}

}
*/
