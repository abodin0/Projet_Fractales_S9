#include "kernel_GPU_mme.cuh"
#include "cuda.h"

__global__ void kernel_updateImage_GPU_mme(const double zoom, const double offsetX, const double offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
{
    int blockID = blockIdx.x + (blockIdx.y * gridDim.x);
    int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int y = threadID/IMAGE_WIDTH;
    int x = threadID%IMAGE_WIDTH;

    double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom + (x * zoom);
    double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
    
    int value    = max_iters - 1;

    double zReal = startReal;
    double zImag = startImag;

    for (unsigned int counter = 0; counter < max_iters; counter++) 
    {
        zReal = abs(zReal);
        zImag = -zImag;

        double r2 = zReal * zReal;
        double r3 = zReal * zReal * zReal;
        double i2 = zImag * zImag;
        double i3 = zImag * zImag * zImag;
        
        zImag = 3.0f * r2 * zImag - i3 + startImag;
        zReal = r3 - 3.0f * zReal * i2 + startReal;

        if ( (r2 + i2) > 4.0f) {
            value = counter;
            break;
        }
    }
    deviceTab[y*IMAGE_WIDTH+x] = value;
}