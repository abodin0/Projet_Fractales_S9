#include "Convergence_m128_x86.hpp"

Convergence_m128_x86::Convergence_m128_x86() : Convergence("M128")
{

}


Convergence_m128_x86::Convergence_m128_x86(ColorMap* _colors, int _max_iters) : Convergence("M128")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_m128_x86::~Convergence_m128_x86( ){

}


void Convergence_m128_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
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

        float _startImag = offsetY - IMAGE_HEIGHT / 2.0 * zoom + (y * zoom);
        float _startReal = offsetX - IMAGE_WIDTH  / 2.0 * zoom;

        __m128 startImag = _mm_setr_ps(_startImag, _startImag, _startImag, _startImag);
        __m128 startReal = _mm_setr_ps(_startReal, _startReal + zoom, _startReal + 2.0 * zoom, _startReal + 3.0 * zoom);


        #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

        for (int x = 0; x < IMAGE_WIDTH;  x += simd) {
            __m128i value = _mm_set1_epi32(0);
            __m128 zReal = startReal;
            __m128 zImag = startImag;

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

                //__m256i mask = _mm256_castpd_si256 (_mm256_cmp_pd(add, _mm256_set1_pd(4), _CMP_GT_OS));
                //value = _mm256_or_si256(_mm256_and_si256(value, mask), _mm256_andnot_si256(_mm256_sub_epi64(value, _mm256_set1_epi64x(1)), mask));

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
            startReal = _mm_add_ps(startReal, XStep);


        }
    }
}
