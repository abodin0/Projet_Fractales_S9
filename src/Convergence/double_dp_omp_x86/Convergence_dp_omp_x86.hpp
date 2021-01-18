#ifndef _Convergence_dp_omp_x86_
#define _Convergence_dp_omp_x86_

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include <thread>
#include "immintrin.h"
#include <omp.h>

#include "../Convergence.hpp"

class Convergence_dp_omp_x86 : public Convergence {

public:

  Convergence_dp_omp_x86();

  Convergence_dp_omp_x86(ColorMap* _colors, int _max_iters);

  ~Convergence_dp_omp_x86( );

//  virtual unsigned int process(const double startReal, const double startImag, unsigned int max_iters);

  virtual void updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image);

};

#endif
