#ifndef _kernel_GPU_julia
#define _kernel_GPU_julia

#include <cstdint>
#include <stdio.h>

extern __global__ void kernel_updateImage_GPU_julia(const double zoom, const double offsetX, const double offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters);


#endif
