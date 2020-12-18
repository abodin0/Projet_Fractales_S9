#include "Convergence_dpj_x86.hpp"

Convergence_dpj_x86::Convergence_dpj_x86() : Convergence("DPJ")
{

}


Convergence_dpj_x86::Convergence_dpj_x86(ColorMap* _colors, int _max_iters) : Convergence("DPJ")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_dpj_x86::~Convergence_dpj_x86( ){

}


void Convergence_dpj_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    #pragma omp parallel for

    for (int y = 0; y < IMAGE_HEIGHT; y++) {            
        for (int x = 0; x < IMAGE_WIDTH;  x++) {
            int value    = max_iters - 1;
        
            double zReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom + (x * zoom);   //xn
            double zImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);   //yn
            double startReal = 0.285;
            double startImag = 0.01;

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                double r2 = zReal * zReal;
                double i2 = zImag * zImag;
                zImag = 2.0f * zReal * zImag + startImag;   //yn
                zReal = (r2 - i2) + startReal;                //xn
                if ( (r2 + i2) > 4.0f) {
                    value = counter;
                    break;
                }
            }
            image.setPixel(x, y, colors->getColor(value));
        }
    }
}
