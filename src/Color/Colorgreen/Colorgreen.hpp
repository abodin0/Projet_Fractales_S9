#ifndef _Colorgreen_
#define _Colorgreen_

#include "../ColorMap.hpp"

class Colorgreen : public ColorMap {
public:

    Colorgreen(int max_iters = 255);

    void setIters(int max_iters);

    virtual ~Colorgreen();

};

#endif // _Colorgreen_
