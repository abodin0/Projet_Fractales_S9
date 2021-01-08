#include "Colorblue.hpp"

    Colorblue::Colorblue(int max_iters) : ColorMap("BLUE"){
        setIters(max_iters);
    }

    void Colorblue::setIters(int max_iters) {

      MAX = max_iters;

      for (int i=0; i <= MAX; ++i) {
        int N = 256;
        int N3 = N * N * N;
        double t = (double)i/(double)MAX;
        int n = (int)(t * (double) N3);

        int b = n/(N * N);
        int nn = n - b * N * N;
        int r = nn/N;
        int g = nn - r * N;
        colors[i] = sf::Color(r, g, b);

        //const int r   = (int)std::round(t * 255.0f);
        //const int g   = (int)std::round(t * 255.0f);
        //const int b   = (int)std::round(t * 255.0f);
        //colors[i] = sf::Color(r, g, b);
    }
}

    Colorblue::~Colorblue(){

    }
