#include "Convergence_dpm_x86.hpp"

Convergence_dpm_x86::Convergence_dpm_x86() : Convergence("DPM")
{

}


Convergence_dpm_x86::Convergence_dpm_x86(ColorMap* _colors, int _max_iters) : Convergence("DPM")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_dpm_x86::~Convergence_dpm_x86( ){

}


void Convergence_dpm_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    #pragma omp parallel for

    for (int y = 0; y < IMAGE_HEIGHT; y++) {

        double startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
        double startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom;

        #pragma omp parallel for

        for (int x = 0; x < IMAGE_WIDTH;  x++) {
            int value    = max_iters - 1;
            double zReal = startReal;
            double zImag = startImag;

            for (unsigned int counter = 0; counter < max_iters; counter++) {
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
            //cout << value << endl;
            image.setPixel(x, y, colors->getColor(value));
            startReal += zoom;
        }
    }
}
