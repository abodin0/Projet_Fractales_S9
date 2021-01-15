#include "Convergence_GPU_ship.hpp"
#include "kernel_GPU_ship.cuh"

#include "cuda_runtime.h"

inline bool CUDA_MALLOC( void ** devPtr, size_t size ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc( devPtr, size );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to allocate buffer\n");
		return false;
	}
	return true;
}

inline bool CUDA_MEMCPY( void * dst, const void * src, size_t count, enum cudaMemcpyKind kind ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy( dst, src, count, kind );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to copy buffer\n");
		return false;
	}
	return true;
}

Convergence_GPU_ship::Convergence_GPU_ship() : Convergence("GPU_double_ship")
{

}


Convergence_GPU_ship::Convergence_GPU_ship(ColorMap* _colors, int _max_iters) : Convergence("GPU_double_ship")
{
    colors    = _colors;
    max_iters = _max_iters;

    hostTab = nullptr;
    deviceTab = nullptr;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to setup cuda device\n");
        exit(0);
	}
}


Convergence_GPU_ship::~Convergence_GPU_ship( ){
     cudaError_t cudaStatus = cudaDeviceReset();
     free(hostTab);
     free(deviceTab);
}

void Convergence_GPU_ship::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    int nb_point = IMAGE_WIDTH*IMAGE_HEIGHT;
    dim3 grid(80,50,1); //nbr bloc
    dim3 block(16,16,1); //nbr threads

    if(hostTab == nullptr)
        hostTab = new uint32_t[nb_point];

    if(deviceTab == nullptr)
        CUDA_MALLOC((void**)&deviceTab, nb_point * sizeof(uint32_t));

    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    kernel_updateImage_GPU_ship<<<grid, block>>>(zoom, offsetX, offsetY, IMAGE_WIDTH, IMAGE_HEIGHT, deviceTab, max_iters);

    CUDA_MEMCPY(hostTab, deviceTab, nb_point*sizeof(uint32_t), cudaMemcpyDeviceToHost);


    for(int y = 0; y < IMAGE_HEIGHT; y++)
    {
        for(int x = 0; x < IMAGE_WIDTH; x++)
        {
            image.setPixel(x, y, colors->getColor(hostTab[x+y*IMAGE_WIDTH]));
        }
    }
}