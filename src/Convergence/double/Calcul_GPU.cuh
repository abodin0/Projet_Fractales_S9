#ifndef _Calcul_GPU
#define _Calcul_GPU

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include <thread>
#include "immintrin.h"

#include "../Convergence.hpp"

class Calcul_GPU : public Convergence {

public:

  Calcul_GPU();

  Calcul_GPU(ColorMap* _colors, int _max_iters);

  ~Calcul_GPU( );

//  virtual unsigned int process(const double startReal, const double startImag, unsigned int max_iters);

  //__global__ void kernel_updateImage(const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab);
  void updateImage_GPU(int nblocks, int nthreads, const long double _zoom, const long double _offsetX, const long double _offsetY, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, uint32_t * deviceTab, int max_iters);

};

#endif
