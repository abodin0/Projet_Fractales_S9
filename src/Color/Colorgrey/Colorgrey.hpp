#ifndef _Colorgrey_
#define _Colorgrey_

#include "../ColorMap.hpp"

class Colorgrey : public ColorMap {
public:

    Colorgrey(int max_iters = 255);

    void setIters(int max_iters);

    virtual ~Colorgrey();

};

#endif // _Colorgrey_
