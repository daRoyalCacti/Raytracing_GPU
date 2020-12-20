//header for including stb_image without the complier displaying warnings while doing so
#pragma once

// Disable warnings
#ifdef _MSC_VER
	//Visual C++
	#pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

//Restoring warnings
#ifdef _MSC_VER
	#pragma warning (pop)
#endif
