#include "Convergence_GPU.hpp"
#include "Calcul_GPU.cuh"

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

Convergence_GPU::Convergence_GPU() : Convergence("GPU")
{

}


Convergence_GPU::Convergence_GPU(ColorMap* _colors, int _max_iters) : Convergence("GPU")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_GPU::~Convergence_GPU( ){

}

void Convergence_GPU::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    uint32_t * hostTab = NULL;
    uint32_t * deviceTab = NULL;

    Calcul_GPU* obj = new Calcul_GPU(colors, max_iters);

    int nb_point = IMAGE_WIDTH*IMAGE_HEIGHT;
    int nthreads = 1024;
    int nblocks = ( nb_point + ( nthreads - 1 ) ) / nthreads;
    
    hostTab = (uint32_t*)malloc(sizeof(uint32_t)*nb_point);

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to setup cuda device\n");
	}

    CUDA_MALLOC((void**)&deviceTab, nb_point * sizeof(uint32_t));

    obj->updateImage_GPU(nblocks, nthreads, _zoom, _offsetX, _offsetY, IMAGE_WIDTH, IMAGE_HEIGHT, deviceTab, max_iters);

    CUDA_MEMCPY(hostTab, deviceTab, nb_point*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for(int y = 0; y < IMAGE_WIDTH; y++)
    {
        for(int x = 0; x < IMAGE_HEIGHT; x++)
        {
            image.setPixel(x, y, colors->getColor(hostTab[x+y*IMAGE_WIDTH]));
        }
    }
    cudaStatus = cudaDeviceReset();
}