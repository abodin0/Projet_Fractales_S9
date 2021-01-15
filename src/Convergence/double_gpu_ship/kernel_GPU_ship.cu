#include "kernel_GPU_ship.cuh"
#include "cuda.h"

__global__ void kernel_updateImage_GPU_ship(const double zoom, const double offsetX, const double offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
{
    int blockID = blockIdx.x + (blockIdx.y * gridDim.x);
    int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int y = threadID/IMAGE_WIDTH;
    int x = threadID%IMAGE_WIDTH;

    double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
    double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom + (x * zoom);

    int value    = max_iters - 1;
    double zReal = startReal;
    double zImag = startImag;

    for (unsigned int counter = 0; counter < max_iters; counter++) 
    {
        zImag = abs(zImag);
        zReal = abs(zReal);

        double r2 = zReal * zReal;
        double i2 = zImag * zImag;
        
        zImag = 2.0f * zReal * zImag + startImag;
        zReal = r2 - i2 + startReal;

        if ( (r2 + i2) > 4.0f) {
            value = counter;
            break;
        }
    }
    deviceTab[y*IMAGE_WIDTH+x] = value;
}