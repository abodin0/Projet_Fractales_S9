#include "Convergence.hpp"

Convergence::Convergence(std::string value)
{
    _name = value;
}

//Convergence(std::string value; ColorMap* _colors, int _max_iters) {}

Convergence::~Convergence()
{

}

std::string Convergence::name()
{
    return _name;
}

void Convergence::setColor(ColorMap* colorizer)
{
    colors = colorizer;
}

void Convergence::setIters(const unsigned int value)
{
    max_iters = value;
}