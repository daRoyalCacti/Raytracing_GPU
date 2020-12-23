
#include "color.h"

int main() {
	const std::string file_dir = "./output";
	const std::string output_loc = "image.ppm";

	average_images(file_dir, output_loc);

	return 0;
}
