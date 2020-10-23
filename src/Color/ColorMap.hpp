#ifndef _ColorMap_
#define _ColorMap_

#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <cassert>
#include "ColorUtils.hpp"

#define VARIATION 0// (MAX/8)
#define ITER ((i+VARIATION)%MAX)

class ColorMap {
public:

    ColorMap(std::string str){
    	_name = str;
    }

    virtual ~ColorMap(){

    }

    virtual void setIters(const int max_iters){
        std::cout << "Oups on est dans ColorMap::setIters()" << std::endl;
    }

//    virtual sf::Color getColor(int iterations){
//    	return sf::Color(0, 0, 0);;
//    }

    virtual sf::Color getColor(const int iterations){
        assert( iterations <= MAX );
        return colors[iterations];
    }

    virtual std::string name(){
        return _name;
    }

protected:
    std::array<sf::Color, 65536> colors;

    int MAX;

    std::string _name;
};

#endif // _ColorMap_
