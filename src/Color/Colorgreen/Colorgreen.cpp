#include "Colorgreen.hpp"

    Colorgreen::Colorgreen(int max_iters) : ColorMap("green"){
        setIters(max_iters);
    }

    void Colorgreen::setIters(int max_iters) {

      MAX = max_iters;
      for (int i=0; i < MAX; ++i) {

        int r = 0;
        int g = 0;
        int b = 0;
        if (i >= 128)
        {
            r = i - 128;
            g = 63 - r;
        }
        else if (i >= 64)
        {
            g= i - 64;
            b = 63 - g;
        }
        else
        {
            b = i;
        }
        colors[i] = sf::Color(r, g, b);

        //const int r   = (int)std::round(t * 255.0f);
        //const int g   = (int)std::round(t * 255.0f);
        //const int b   = (int)std::round(t * 255.0f);
        //colors[i] = sf::Color(r, g, b);
    }
}

    Colorgreen::~Colorgreen(){

    }
