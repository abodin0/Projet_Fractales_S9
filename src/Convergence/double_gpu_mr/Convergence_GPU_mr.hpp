#ifndef _Convergence_GPU_mr
#define _Convergence_GPU_mr

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include <thread>
#include "immintrin.h"

#include "../Convergence.hpp"

class Convergence_GPU_mr : public Convergence {

public:

  Convergence_GPU_mr();

  Convergence_GPU_mr(ColorMap* _colors, int _max_iters);

  ~Convergence_GPU_mr( );

//  virtual unsigned int process(const double startReal, const double startImag, unsigned int max_iters);

  virtual void updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image);

private:

    uint32_t * hostTab;
    uint32_t * deviceTab;

};

#endif