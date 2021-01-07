//header for including stb_image without the complier displaying warnings while doing so
#pragma once

// Disable warnings
#if defined(_MSC_VER) || defined(__NVCC__)
	//Visual C++
	#pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

//Restoring warnings
#if defined(_MSC_VER) || defined(__NVCC__)
	#pragma warning (pop)
#endif
