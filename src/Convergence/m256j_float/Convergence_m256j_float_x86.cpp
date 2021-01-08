#include "Convergence_m256j_float_x86.hpp"

Convergence_m256j_float_x86::Convergence_m256j_float_x86() : Convergence("M256DJ_FLOAT")
{

}


Convergence_m256j_float_x86::Convergence_m256j_float_x86(ColorMap* _colors, int _max_iters) : Convergence("M256J_FLOAT")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_m256j_float_x86::~Convergence_m256j_float_x86( ){

}


void Convergence_m256j_float_x86::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    float offsetX = _offsetX;
    float offsetY = _offsetX;
    float zoom    = _zoom;

    __m256i one = _mm256_set1_epi32(1);
    __m256 two = _mm256_set1_ps(2);
    __m256 four = _mm256_set1_ps(4);

    const int simd = sizeof(__m256) / sizeof(double);

    __m256 XStep = _mm256_set1_ps(simd * zoom);

    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH;  x += simd) {

            float _startReal = 0.285;
            float _startImag = 0.01;

            float _zReal = offsetX - IMAGE_WIDTH  / 2.0f * zoom + (x * zoom);
            float _zImag = offsetY - IMAGE_HEIGHT / 2.0f * zoom + (y * zoom);

            __m256 zImag = _mm256_setr_ps(_zImag, _zImag, _zImag, _zImag, _zImag, _zImag, _zImag, _zImag);
            __m256 zReal = _mm256_setr_ps(_zReal, _zReal + zoom, _zReal + 2.0 * zoom, _zReal + 3.0 * zoom, _zReal + 4.0 * zoom, _zReal + 5.0 * zoom, _zReal + 6.0 * zoom, _zReal + 7.0 * zoom);

            __m256 startImag = _mm256_setr_ps(_startImag, _startImag, _startImag, _startImag, _startImag, _startImag, _startImag, _startImag);
            __m256 startReal = _mm256_setr_ps(_startReal, _startReal, _startReal, _startReal, _startReal, _startReal, _startReal, _startReal);

            __m256i value = _mm256_set1_epi32(0);

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                __m256 r2 = _mm256_mul_ps(zReal, zReal);
                __m256 i2 = _mm256_mul_ps(zImag, zImag);

                __m256 mul1 = _mm256_mul_ps(zReal, zImag);
                __m256 mul2 = _mm256_mul_ps(two, mul1);
                zImag = _mm256_add_ps(mul2, startImag);

                __m256 sub = _mm256_sub_ps(r2, i2);
                zReal = _mm256_add_ps(sub, startReal);

                __m256 add = _mm256_add_ps(r2, i2);
                __m256i v = _mm256_add_epi32(value, one);
                __m256 mask = _mm256_cmp_ps(add, four, _CMP_LT_OS);

                value = _mm256_blendv_epi8(value, v, _mm256_castps_si256(mask));

                if(_mm256_movemask_ps(mask) == 0)
                    break;
            }
            #if 0
            cout << _mm256_extract_epi32(value, 0) << endl;
            cout << _mm256_extract_epi32(value, 1) << endl;
            cout << _mm256_extract_epi32(value, 2) << endl;
            cout << _mm256_extract_epi32(value, 3) << endl;
            cout << _mm256_extract_epi32(value, 4) << endl;
            cout << _mm256_extract_epi32(value, 5) << endl;
            cout << _mm256_extract_epi32(value, 6) << endl;
            cout << _mm256_extract_epi32(value, 7) << endl;
            #endif

            image.setPixel(x  , y, colors->getColor(_mm256_extract_epi32(value, 0)));
            image.setPixel(x+1, y, colors->getColor(_mm256_extract_epi32(value, 1)));
            image.setPixel(x+2, y, colors->getColor(_mm256_extract_epi32(value, 2)));
            image.setPixel(x+3, y, colors->getColor(_mm256_extract_epi32(value, 3)));
            image.setPixel(x+4, y, colors->getColor(_mm256_extract_epi32(value, 4)));
            image.setPixel(x+5, y, colors->getColor(_mm256_extract_epi32(value, 5)));
            image.setPixel(x+6, y, colors->getColor(_mm256_extract_epi32(value, 6)));
            image.setPixel(x+7, y, colors->getColor(_mm256_extract_epi32(value, 7)));
        }
    }
}
