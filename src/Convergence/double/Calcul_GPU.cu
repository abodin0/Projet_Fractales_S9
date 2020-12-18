#include "Calcul_GPU.cuh"
#include "cuda.h"

__global__ void kernel_updateImage_GPU(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
{
    int blockID = blockIdx.x * blockIdx.y + blockIdx.x;
    int threadID = blockID * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    //printf("blockID : %d, threadID : %d\n", blockID, threadID);

    //const int nb_point = IMAGE_WIDTH * IMAGE_HEIGHT;

    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    int x = threadID%IMAGE_WIDTH;
    int y = threadID/IMAGE_WIDTH;

    //printf("x = %d, y = %d\n", x,y);

    double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
    double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom;
/*
   //for (int x = 0; x < IMAGE_WIDTH; x++)
    //{
        int value    = max_iters - 1;
        double zReal = startReal;
        double zImag = startImag;

        for (unsigned int counter = 0; counter < max_iters; counter++) 
        {
            double r2 = zReal * zReal;
            double i2 = zImag * zImag;
            zImag = 2.0f * zReal * zImag + startImag;
            zReal = r2 - i2 + startReal;
            if ( (r2 + i2) > 4.0f) {
                value = counter;
                break;
            }
        }
        __syncthreads();
       // deviceTab[y*IMAGE_WIDTH+x] = value;
        //startReal += zoom;
    //}
    */
    deviceTab[y*IMAGE_WIDTH+x] = threadID%256;
}


__global__ void kernel_1D_updateImage_GPU(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;;

    //const int nb_point = IMAGE_WIDTH * IMAGE_HEIGHT;

    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    //const int x = threadID%IMAGE_WIDTH;
    int y = threadID/IMAGE_WIDTH;

    double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
    double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom;

    for (int x = 0; x < IMAGE_WIDTH; x++)
    {
        int value    = max_iters - 1;
        double zReal = startReal;
        double zImag = startImag;

        for (unsigned int counter = 0; counter < max_iters; counter++) 
        {
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
        startReal += zoom;
    }
}