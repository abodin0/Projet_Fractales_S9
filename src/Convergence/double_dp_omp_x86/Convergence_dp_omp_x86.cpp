#include "Convergence_dp_omp_x86.hpp"

Convergence_dp_omp_x86::Convergence_dp_omp_x86() : Convergence("DP OMP")
{

}


Convergence_dp_omp_x86::Convergence_dp_omp_x86(ColorMap* _colors, int _max_iters) : Convergence("DP OMP")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_dp_omp_x86::~Convergence_dp_omp_x86( ){

}


void Convergence_dp_omp_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

    for (int y = 0; y < IMAGE_HEIGHT; y++) {

        double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);

        #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

        for (int x = 0; x < IMAGE_WIDTH;  x++) {

            double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom + (x * zoom);

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
        }
    }
}