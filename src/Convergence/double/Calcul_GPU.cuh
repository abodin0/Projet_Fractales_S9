#ifndef _Calcul_GPU
#define _Calcul_GPU

#include <cstdint>
#include <stdio.h>

extern __global__ void kernel_updateImage_GPU(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters);


#endif
