#include "kernel_GPU_mr.cuh"
#include "cuda.h"

__global__ void kernel_updateImage_GPU_mr(const double zoom, const double offsetX, const double offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
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

        double r5 = zReal * zReal * zReal * zReal * zReal;
        double r4 = zReal * zReal * zReal * zReal;
        double r3 = zReal * zReal * zReal;
        double r2 = zReal * zReal;
        double i2 = zImag * zImag;
        double i3 = zImag * zImag * zImag;
        double i4 = zImag * zImag * zImag * zImag;
        double i5 = zImag * zImag * zImag * zImag * zImag;
        
        zImag = 5.0f * r4 * zImag - 10.0f * r2 * i3 + i5 + startImag;
        zReal = r5 -10.0f * r3 * i2 + 5.0f * zReal * i4 + startReal;

        if ( (r2 + i2) > 4.0f) {
            value = counter;
            break;
        }
    }
    deviceTab[y*IMAGE_WIDTH+x] = value;
}