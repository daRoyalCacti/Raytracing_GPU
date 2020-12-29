#pragma once

#include "vec3.h"

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <cctype> 	//for isspace()
#include <vector>
#include "common.h"



void write_frame_buffer(std::string file_loc, const color* f, const int width, const int height, const int no_fb = 1) {
	std::ofstream out;
	out.open(file_loc);
	out << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height-1; j >= 0; j--) 
		for (int i = 0; i < width; i++) {
			const size_t pixel_index = j*width + i;
			float r = f[pixel_index].x();
			float g = f[pixel_index].y();
			float b = f[pixel_index].z();
			

			r /= no_fb;
			g /= no_fb;
			b /= no_fb;

			//gamma correcting using "gamma 2"
			//i.e. raising the color to the power of 1/gamma = 1/2
			r = sqrt(r);
			g = sqrt(g);
			b = sqrt(b);

			//write the color scaled to be in [0, 255]
			out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
				<< static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
				<< static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';

						
		}
	out.close();
}





namespace fs = std::filesystem;
//could be updated using std::string.find(std::string)
void average_images(std::string file_dir, std::string output_loc) {
	int num_files = 0;
	bool first_file = true;
	int img_w;
	int img_h;
	std::vector<color> fb;
	for (const auto & entry : fs::directory_iterator(file_dir) ) {	//itterating through all files in file_dir
		num_files++;
		int fb_counter = 0;
		
		std::string line;
		std::ifstream input(entry.path());	//reading in each file in file_dir
		if (input.is_open() ) {	
			int line_counter = 0;
			while (getline (input,line)) {	//reading in each line in the file
				line_counter++;

				if (first_file && line_counter == 2) {	//this line holds the image dimensions
					//setting img_w and img_h
					std::string width = "";
					std::string height = "";
					
					int size_of_width = 0;	//need to know how many digits in width. e.g. 1200 vs 120
					for (const auto& chara : line) {
						size_of_width++;
						if (isspace(chara)) {
							size_of_width--;
							break;
						}
					}

					width.append(line, 0, size_of_width);
				        height.append(line, size_of_width+1, size_of_width+10);	//making size of height large to account for the case of small width and large height
					img_w = std::stoi(width);
					img_h = std::stoi(height);

					fb.resize(img_w*img_h);
				}

				if (line_counter > 3) {	//the ppm files have 3 lines of data that is not the image itself
					std::string col1 = "";
					std::string col2 = "";
					std::string col3 = "";

					//finding space 1
					int space1 = 0;
					for (const auto& chara : line) {
						space1++;
						if (isspace(chara)) {
							space1--;
							break;
						}
					}

					//finding space 2
					int space2 = 0;
					for (const auto& chara : line) {
						space2++;
						if (space2 > (space1+1) && isspace(chara)) {
							space2--;
							break;
						}
					}

					col1.append(line, 0, space1);
					col2.append(line, space1+1, space2-space1);
					col3.append(line, space2+1, 3);

					color col(std::stoi(col1)*std::stoi(col1)/(255.0f*255.0f), std::stoi(col2)*std::stoi(col2)/(255.0f*255.0f), std::stoi(col3)*std::stoi(col3)/(255.0f*255.0f) );

					fb[fb_counter++] += col;

				}

			}

		}

		first_file = false;

		input.close();
	}


	std::ofstream out;
	out.open(output_loc);
	out << "P3\n" << img_w << " " << img_h << "\n255\n";
	for (int j = 0; j<img_h; j++) 
		for (int i = 0; i <img_w; i++) {
			const size_t pixel_index = j*img_w + i;

			float r = fb[pixel_index].x();
			float g = fb[pixel_index].y();
			float b = fb[pixel_index].z();
			
			r /= num_files;
			g /= num_files;
			b /= num_files;

			//gamma correcting using "gamma 2"
			//i.e. raising the color to the power of 1/gamma = 1/2
			r = sqrt(r);
			g = sqrt(g);
			b = sqrt(b);

			//write the color scaled to be in [0, 255]
			out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
				<< static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
				<< static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';

						
		}
	out.close();

}
