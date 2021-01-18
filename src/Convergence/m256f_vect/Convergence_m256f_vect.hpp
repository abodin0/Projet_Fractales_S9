#ifndef _Convergence_m256f_vect_
#define _Convergence_m256f_vect_

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include <thread>
#include "immintrin.h"
#include <omp.h>
#include <thread>
#include "Convergence/vectorclass/vectorclass.h"

#include "../Convergence.hpp"

class Convergence_m256f_vect : public Convergence {

public:

  Convergence_m256f_vect();

  Convergence_m256f_vect(ColorMap* _colors, int _max_iters);

  ~Convergence_m256f_vect( );

//  virtual unsigned int process(const double startReal, const double startImag, unsigned int max_iters);

  virtual void updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image);

};

#endif
