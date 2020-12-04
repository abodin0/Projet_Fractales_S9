#include "Calcul_GPU.cuh"
#include "cuda.h"

Calcul_GPU::Calcul_GPU() : Convergence("GPU")
{

}


Calcul_GPU::Calcul_GPU(ColorMap* _colors, int _max_iters) : Convergence("GPU")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Calcul_GPU::~Calcul_GPU( ){

}


__global__ void kernel_updateImage_GPU(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_point = IMAGE_WIDTH * IMAGE_HEIGHT;
    int x, y = 0;

    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    if(i<nb_point)
    {
      x = i%IMAGE_WIDTH;
      y = (i-x)/IMAGE_WIDTH;

      double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
      double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom;
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

      deviceTab[x+y*IMAGE_WIDTH] = value;
      startReal += zoom;
    }
}

void Calcul_GPU::updateImage_GPU(int nblocks, int nthreads, const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters)
{
  kernel_updateImage_GPU<<<nblocks, nthreads>>>(_zoom, _offsetX, _offsetY, IMAGE_WIDTH, IMAGE_HEIGHT, deviceTab, max_iters);
}
