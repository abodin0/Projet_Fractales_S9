#ifndef _Calcul_GPU_float
#define _Calcul_GPU_float

#include <cstdint>
#include <stdio.h>

extern __global__ void kernel_updateImage_GPU_float(const float zoom, const float offsetX, const float offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters);


#endif
