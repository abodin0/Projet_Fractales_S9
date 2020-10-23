#ifndef _ColorSmooth_
#define _ColorSmooth_

#include "../ColorMap.hpp"

class ColorSmooth : public ColorMap {
public:

    ColorSmooth(int max_iters = 255);

    void setIters(int max_iters);

    virtual ~ColorSmooth();

};

#endif // _ColorSmooth_
