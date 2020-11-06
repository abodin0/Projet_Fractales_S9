#include "Mandelbrot.hpp"

Mandelbrot::Mandelbrot(Settings* p) {
    params    = p;
    c         = nullptr;
    colors    = nullptr;
    allocRessources( );
}

Mandelbrot::~Mandelbrot(){
    freeRessources( );
}

void Mandelbrot::allocRessources( ){

    if( colors == nullptr )
        colors = library.get( params->ColorMapMode() );
    else
        colors = library.get();
    colors->setIters( params->Iterations() );

    if( c == nullptr )
        c = converge.get( params->ConvergenceType() );
    else
        c = converge.get();

    c->setIters( params->Iterations() );
    c->setColor( colors );
}


void Mandelbrot::freeRessources( ){
//    delete c;
//    delete colors;
}


//
//
//
void Mandelbrot::Update(){
    freeRessources( );
    allocRessources( );
}


//
//
//
void Mandelbrot::nextColorMap(){
    freeRessources ();
    library.next   ();
    allocRessources();
}


//
//
//
void Mandelbrot::previousColorMap(){
    freeRessources  ();
    library.previous();
    allocRessources ();
}


//
//
//
void Mandelbrot::nextConvergence(){
    freeRessources  ();
    converge.next();
    allocRessources ();
}


//
//
//
void Mandelbrot::previousConvergence(){
    freeRessources  ();
    converge.previous();
    allocRessources ();
}


//
//
//
void Mandelbrot::updateImage(const long double zoom, const long double offsetX, const long double offsetY, sf::Image& image) {

    c->updateImage<<1,1>>(zoom, offsetX, offsetY, params->Width(), params->Height(), image);

    if (params->isCentralDotEnabled) {
        sf::Color white(255, 255, 255);
        image.setPixel(params->Width()/2-1, params->Height()/2,   white);
        image.setPixel(params->Width()/2+1, params->Height()/2,   white);
        image.setPixel(params->Width()/2,   params->Height()/2,   white);
        image.setPixel(params->Width()/2,   params->Height()/2-1, white);
        image.setPixel(params->Width()/2,   params->Height()/2+1, white);
    }
}
