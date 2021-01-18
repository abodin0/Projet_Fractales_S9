#include "Convergence_m256f_vect.hpp"

Convergence_m256f_vect::Convergence_m256f_vect() : Convergence("M256F_VECT")
{

}


Convergence_m256f_vect::Convergence_m256f_vect(ColorMap* _colors, int _max_iters) : Convergence("M256F_VECT")
{
    colors    = _colors;
    max_iters = _max_iters;
}


Convergence_m256f_vect::~Convergence_m256f_vect( ){

}


void Convergence_m256f_vect::updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image)
{
    float offsetX = _offsetX;
    float offsetY = _offsetX;
    float zoom    = _zoom;

    Vec8f one = Vec8f(1);
    Vec8f two = Vec8f(2);
    Vec8f four = Vec8f(4);

    const int simd = sizeof(__m256) / sizeof(double);

    Vec8f XStep = Vec8f(simd * zoom);

    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

    for (int y = 0; y < IMAGE_HEIGHT; y++) {

        float _startImag = offsetY - IMAGE_HEIGHT / 2.0 * zoom + (y * zoom);
        float _startReal = offsetX - IMAGE_WIDTH  / 2.0 * zoom;

        Vec8f startImag = Vec8f(_startImag, _startImag, _startImag, _startImag, _startImag, _startImag, _startImag, _startImag);
        Vec8f startReal = Vec8f(_startReal, _startReal + zoom, _startReal + 2.0 * zoom, _startReal + 3.0 * zoom, _startReal + 4.0 * zoom, _startReal + 5.0 * zoom, _startReal + 6.0 * zoom, _startReal + 7.0 * zoom);


        #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(dynamic)

        for (int x = 0; x < IMAGE_WIDTH;  x += simd) {

            Vec8f value = Vec8f(0);
            Vec8f zReal = startReal;
            Vec8f zImag = startImag;

            for (unsigned int counter = 0; counter < max_iters; counter++) {
                Vec8f r2 = zReal * zReal;
                Vec8f i2 = zImag * zImag;

                Vec8f mul1 = zReal * zImag;
                Vec8f mul2 = two * mul1;

                zImag = mul2 + startImag;

                Vec8f sub = r2 - i2;

                zReal = sub + startReal;

                Vec8f add = r2 + i2;
                Vec8f v = value + one;
                Vec8fb mask = add > four;

                value = select(mask, v, value);


                //__m256i mask = _mm256_castpd_si256 (_mm256_cmp_pd(add, _mm256_set1_pd(4), _CMP_GT_OS));
                //value = _mm256_or_si256(_mm256_and_si256(value, mask), _mm256_andnot_si256(_mm256_sub_epi64(value, _mm256_set1_epi64x(1)), mask));

                if(horizontal_or(mask))
                    break;
            }

            float tmp[8];

            value.store(tmp);

            for (int i=0; i<=8; i++)
              image.setPixel(x + i , y, colors->getColor(tmp[i]));

            startReal = startReal + XStep;


        }
    }
}
