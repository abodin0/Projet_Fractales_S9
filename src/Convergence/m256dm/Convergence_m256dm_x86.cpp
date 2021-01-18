#include "Convergence_m256dm_x86.hpp"

Convergence_m256dm_x86::Convergence_m256dm_x86() : Convergence("M256DM")
{

}


Convergence_m256dm_x86::Convergence_m256dm_x86(ColorMap* _colors, int _max_iters) : Convergence("M256DM")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_m256dm_x86::~Convergence_m256dm_x86( ){

}


void Convergence_m256dm_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    double offsetX = _offsetX;
    double offsetY = _offsetX;
    double zoom    = _zoom;

    __m256i one = _mm256_set1_epi64x(1);
    __m256d two = _mm256_set1_pd(2);
    __m256d three = _mm256_set1_pd(3);
    __m256d four = _mm256_set1_pd(4);

    const int simd = sizeof(__m256d) / sizeof(double);

    __m256d XStep = _mm256_set1_pd(simd * zoom);

    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

    for (int y = 0; y < IMAGE_HEIGHT; y++) {

        double _startImag = offsetY - IMAGE_HEIGHT / 2.0 * zoom + (y * zoom);
        double _startReal = offsetX - IMAGE_WIDTH  / 2.0 * zoom;

        __m256d startImag = _mm256_setr_pd(_startImag, _startImag, _startImag, _startImag);
        __m256d startReal = _mm256_setr_pd(_startReal, _startReal + zoom, _startReal + 2.0 * zoom, _startReal + 3.0 * zoom);

        for (int x = 0; x < IMAGE_WIDTH;  x += simd) {
            __m256i value = _mm256_set1_epi64x(0);
            __m256d zReal = startReal;
            __m256d zImag = startImag;

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                __m256d r2 = _mm256_mul_pd (zReal, zReal);
                __m256d i2 = _mm256_mul_pd (zImag, zImag);

                __m256d r3 = _mm256_mul_pd (r2, zReal);
                __m256d i3 = _mm256_mul_pd (i2, zImag);

                __m256d mul1 = _mm256_mul_pd (r2, zImag);
                __m256d mul2 = _mm256_mul_pd (three, mul1);
                __m256d sub1 = _mm256_sub_pd (mul2, i3);
                zImag = _mm256_add_pd (sub1, startImag);

                __m256d mul3 = _mm256_mul_pd (i2, zReal);
                __m256d mul4 = _mm256_mul_pd (three, mul3);
                __m256d sub2 = _mm256_sub_pd (r3, mul4);
                zReal = _mm256_add_pd (sub2, startReal);

                __m256d add = _mm256_add_pd(r2, i2);
                __m256i v = _mm256_add_epi64(value, one);
                __m256d mask = _mm256_cmp_pd(add, four, _CMP_LT_OS);

                value = _mm256_blendv_epi8(value, v, _mm256_castpd_si256(mask));

                if(_mm256_movemask_pd(mask) == 0)
                    break;
            }
            #if 0
            cout << _mm256_extract_epi64(value, 0) << endl;
            cout << _mm256_extract_epi64(value, 1) << endl;
            cout << _mm256_extract_epi64(value, 2) << endl;
            cout << _mm256_extract_epi64(value, 3) << endl;
            #endif

            image.setPixel(x  , y, colors->getColor(_mm256_extract_epi64(value, 0)));
            image.setPixel(x+1, y, colors->getColor(_mm256_extract_epi64(value, 1)));
            image.setPixel(x+2, y, colors->getColor(_mm256_extract_epi64(value, 2)));
            image.setPixel(x+3, y, colors->getColor(_mm256_extract_epi64(value, 3)));
            startReal = _mm256_add_pd(startReal, XStep);


        }
    }
}
