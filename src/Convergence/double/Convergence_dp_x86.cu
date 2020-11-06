#include "Convergence_dp_x86.hpp"

Convergence_dp_x86::Convergence_dp_x86() : Convergence("DP")
{

}


Convergence_dp_x86::Convergence_dp_x86(ColorMap* _colors, int _max_iters) : Convergence("DP")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_dp_x86::~Convergence_dp_x86( ){

}


__global__ void Convergence_dp_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    cudaError_t a = cudaMallocManaged(&offsetX, N*sizeof(double));
    if (a != cudaSuccess) {
      printf("Error %s %d\n",__FILE__, __LINE__);
      exit(0);
    }

    cudaError_t b = cudaMallocManaged(&offsetY, N*sizeof(double));
    if (b != cudaSuccess) {
      printf("Error %s %d\n",__FILE__, __LINE__);
      exit(0);
    }

    cudaError_t c = cudaMallocManaged(&zoom, N*sizeof(double));
    if (c != cudaSuccess) {
      printf("Error %s %d\n",__FILE__, __LINE__);
      exit(0);
    }

    for (int y = 0; y < IMAGE_HEIGHT; y++) {

        double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
        double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom;

        for (int x = 0; x < IMAGE_WIDTH;  x++) {
            int value    = max_iters - 1;
            double zReal = startReal;
            double zImag = startImag;

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                double r2 = zReal * zReal;
                double i2 = zImag * zImag;
                zImag = 2.0f * zReal * zImag + startImag;
                zReal = r2 - i2 + startReal;
                if ( (r2 + i2) > 4.0f) {
                    value = counter;
                    break;
                }
            }
            image.setPixel(x, y, colors->getColor(value));
            startReal += zoom;
        }
    }
}
