#include "Convergence_m256d_x86.hpp"

Convergence_m256d_x86::Convergence_m256d_x86() : Convergence("M256D")
{

}


Convergence_m256d_x86::Convergence_m256d_x86(ColorMap* _colors, int _max_iters) : Convergence("M256D")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_m256d_x86::~Convergence_m256d_x86( ){

}


void Convergence_m256d_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    #pragma omp parallel for

    for (int y = 0; y < IMAGE_HEIGHT; y++) {

        double _startImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);
        __m256d startImag = _mm256_set_pd(_startImag, _startImag, _startImag, _startImag);
        double _startReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom;
        __m256d startReal = _mm256_set_pd(_startReal, _startReal, _startReal, _startReal);

        const int simd = sizeof(__m256d) / sizeof(double);

        #pragma omp parallel for
        for (int x = 0; x < IMAGE_WIDTH;  x += simd) {
            __m256i value = _mm256_set1_epi64x(max_iters - 1);
            __m256d zReal = _mm256_set_pd(_startReal, _startReal + zoom, _startReal + 2 * zoom, _startReal + 3 * zoom);
            __m256d zImag = _mm256_set_pd(_startImag, _startImag, _startImag, _startImag);

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                __m256d r2 = _mm256_mul_pd (zReal, zReal);
                __m256d i2 = _mm256_mul_pd (zImag, zImag);
                zImag = _mm256_add_pd (_mm256_mul_pd (_mm256_set1_pd(2), _mm256_mul_pd (zReal, zImag)), startImag);
                zReal = _mm256_add_pd (_mm256_sub_pd (r2, i2), startReal);
                __m256d add = _mm256_add_pd(r2, i2);
                __m256i conv = _mm256_castpd_si256 (_mm256_cmp_pd(add, _mm256_set1_pd(4), _CMP_GT_OS));
                value = _mm256_or_si256(_mm256_and_si256(value, conv), _mm256_andnot_si256(_mm256_sub_epi64(value, _mm256_set1_epi64x(1)), conv));

                if(_mm256_movemask_epi8(value) == 0)
                    break;
            }
            cout << _mm256_extract_epi64(value, 0) << endl;
            cout << _mm256_extract_epi64(value, 1) << endl;
            cout << _mm256_extract_epi64(value, 2) << endl;
            cout << _mm256_extract_epi64(value, 3) << endl;
            image.setPixel(x, y, colors->getColor(_mm256_extract_epi64(value, 0)));
            image.setPixel(x+1, y, colors->getColor(_mm256_extract_epi64(value, 1)));
            image.setPixel(x+2, y, colors->getColor(_mm256_extract_epi64(value, 2)));
            image.setPixel(x+3, y, colors->getColor(_mm256_extract_epi64(value, 3)));
            startReal += simd * zoom;
        }
    }
}
