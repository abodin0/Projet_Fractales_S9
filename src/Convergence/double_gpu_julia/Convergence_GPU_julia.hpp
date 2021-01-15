#ifndef _Convergence_GPU_julia
#define _Convergence_GPU_julia

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include <thread>
#include "immintrin.h"

#include "../Convergence.hpp"

class Convergence_GPU_julia : public Convergence {

public:

  Convergence_GPU_julia();

  Convergence_GPU_julia(ColorMap* _colors, int _max_iters);

  ~Convergence_GPU_julia( );

//  virtual unsigned int process(const double startReal, const double startImag, unsigned int max_iters);

  virtual void updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, sf::Image& image);

private:

    uint32_t * hostTab;
    uint32_t * deviceTab;

};

#endif