#include "ColorSmooth.hpp"

    ColorSmooth::ColorSmooth(int max_iters) : ColorMap("SMOOTH"){
        setIters(max_iters);
    }

    void ColorSmooth::setIters(int max_iters) {

        MAX = max_iters; // memorisation du nombre d'iterations max.

        for (int i=0; i <= MAX; ++i) {
            const float t = (float)i/(float)max_iters;
            const int r   = (int)std::round( 9.f * (1.0f-t) * t * t * t               * 255.0f);
            const int g   = (int)std::round(15.f * (1.0f-t) * (1.0f-t) * t * t        * 255.0f);
            const int b   = (int)std::round(8.5f * (1.0f-t) * (1.0f-t) * (1.0f-t) * t * 255.0f);
            colors[i] = sf::Color(r, g, b);
        }
    }

    ColorSmooth::~ColorSmooth(){

    }
