#ifndef _Colorblue_
#define _Colorblue_

#include "../ColorMap.hpp"

class Colorblue : public ColorMap {
public:

    Colorblue(int max_iters = 255);

    void setIters(int max_iters);

    virtual ~Colorblue();

};

#endif // _Colorblue_
