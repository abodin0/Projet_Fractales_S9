#include "Colorgrey.hpp"

    Colorgrey::Colorgrey(int max_iters) : ColorMap("grey"){
        setIters(max_iters);
    }

    void Colorgrey::setIters(int max_iters) {

      MAX = max_iters;

      for (int i=0; i < MAX; ++i) {

        const float t = (float)i/(float)max_iters;

        const int r   = (int)std::round(t * 255.0f);
        const int g   = (int)std::round(t * 255.0f);
        const int b   = (int)std::round(t * 255.0f);

        colors[i] = sf::Color(r, g, b);
    }
}

    Colorgrey::~Colorgrey(){

    }
