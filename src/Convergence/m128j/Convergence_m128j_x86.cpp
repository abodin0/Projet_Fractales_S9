#include "Convergence_m128j_x86.hpp"

Convergence_m128j_x86::Convergence_m128j_x86() : Convergence("M128J")
{

}


Convergence_m128j_x86::Convergence_m128j_x86(ColorMap* _colors, int _max_iters) : Convergence("M128J")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_m128j_x86::~Convergence_m128j_x86( ){

}


void Convergence_m128j_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    float offsetX = _offsetX;
    float offsetY = _offsetX;
    float zoom    = _zoom;

    __m128i one = _mm_set1_epi32(1);
    __m128 two = _mm_set1_ps(2);
    __m128 four = _mm_set1_ps(4);

    const int simd = sizeof(__m128) / sizeof(float);

    __m128 XStep = _mm_set1_ps(simd * zoom);

    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH;  x += simd) {

            float _startReal = 0.285;
            float _startImag = 0.01;

            float _zReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom + (x * zoom);
            float _zImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);

            __m128 zImag = _mm_setr_ps(_zImag, _zImag, _zImag, _zImag);
            __m128 zReal = _mm_setr_ps(_zReal, _zReal + zoom, _zReal + 2.0 * zoom, _zReal + 3.0 * zoom);

            __m128 startImag = _mm_setr_ps(_startImag, _startImag, _startImag, _startImag);
            __m128 startReal = _mm_setr_ps(_startReal, _startReal, _startReal, _startReal);

            __m128i value = _mm_set1_epi32(0);

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                __m128 r2 = _mm_mul_ps(zReal, zReal);
                __m128 i2 = _mm_mul_ps(zImag, zImag);

                __m128 mul1 = _mm_mul_ps(zReal, zImag);
                __m128 mul2 = _mm_mul_ps(two, mul1);
                zImag = _mm_add_ps(mul2, startImag);

                __m128 sub = _mm_sub_ps(r2, i2);
                zReal = _mm_add_ps(sub, startReal);

                __m128 add = _mm_add_ps(r2, i2);
                __m128i v = _mm_add_epi32(value, one);
                __m128 mask = _mm_cmp_ps(add, four, _CMP_LT_OS);

                value = _mm_blendv_epi8(value, v, _mm_castps_si128(mask));

                if(_mm_movemask_ps(mask) == 0)
                    break;
            }
            #if 0
            cout << _mm_extract_epi32(value, 0) << endl;
            cout << _mm_extract_epi32(value, 1) << endl;
            cout << _mm_extract_epi32(value, 2) << endl;
            cout << _mm_extract_epi32(value, 3) << endl;
            #endif

            image.setPixel(x  , y, colors->getColor(_mm_extract_epi32(value, 0)));
            image.setPixel(x+1, y, colors->getColor(_mm_extract_epi32(value, 1)));
            image.setPixel(x+2, y, colors->getColor(_mm_extract_epi32(value, 2)));
            image.setPixel(x+3, y, colors->getColor(_mm_extract_epi32(value, 3)));


        }
    }
}
