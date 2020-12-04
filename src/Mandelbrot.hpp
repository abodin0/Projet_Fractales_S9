#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <unistd.h>


#include "Utils/Settings.hpp"

#include "Convergence/Convergence.hpp"

#include "Color/ColorLibrary.hpp"
#include "Convergence/ConvergenceLibrary.hpp"

#include "Color/ColorMap.hpp"

enum t_format    { f_double, f_float, f_int };
enum t_processor { x86, sse, avx, cuda };

class Mandelbrot {
private:
    ColorMap* colors;
    Convergence* c;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
//    unsigned int max_iters;

public:

    Mandelbrot(Settings* p);

    ~Mandelbrot();

    void freeRessources();

    void allocRessources();

    void Update();

    void updateImage(const long double zoom, const long double offsetX, const long double offsetY, sf::Image& image);

    void nextColorMap();
    void previousColorMap();
    void nextConvergence();
    void previousConvergence();

private:
    int MAX;
    void updateImageSlice(const long double zoom, const long double offsetX, const long double offsetY, sf::Image& image, int minY, int maxY);
    //void updateImage(double zoom, double offsetX, double offsetY, sf::Image& image);
    ColorLibrary       library;
    ConvergenceLibrary converge;
    Settings*          params;
};


#endif
