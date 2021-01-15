#ifndef _ColorLibrary_
#define _ColorLibrary_

#include <vector>
#include <map>
#include <string>

#include "ColorMap.hpp"
#include "ColorSmooth/ColorSmooth.hpp"
#include "Colorgreen/Colorgreen.hpp"
#include "Colorblue/Colorblue.hpp"
#include "Colorgrey/Colorgrey.hpp"

class ColorLibrary {
private:
    std::vector<            ColorMap*> list;
    std::map   <std::string,ColorMap*> dico;
    std::map   <std::string,int      > indx;

    int counter;

public:

    ColorLibrary(){

        //
        list.push_back( new ColorSmooth    () );
        list.push_back( new Colorblue      () );
        list.push_back( new Colorgreen     () );
        list.push_back( new Colorgrey      () );

        //
        //
        for(int i=0; i<list.size(); i++){
            dico[ list[i]->name() ] = list[i];
            indx[ list[i]->name() ] =      i ;
        }

        counter = 0;

    }

    virtual ~ColorLibrary(){

        for(int i=0; i<list.size(); i++)
            delete list[i];

    }

    ColorMap* get(std::string name){
        counter = indx[name];
        return get();
    }

    ColorMap* get(int num){
        counter = num;
        return get();
    }

    ColorMap* get(){
        printf("ColorMap :: get (%d) : (%p) name = %s \n", counter, list[counter], list[counter]->name().c_str());
        return list[counter];
    }

    void next(){
        counter = (counter + 1) % list.size();
        //printf("COUNTER %d / %lu\n", counter, list.size());
        //return get();
    }

    void previous(){
        counter = (counter + list.size() - 1) % list.size();
        //printf("COUNTER %d / %lu\n", counter, list.size());
        //return get();
    }

};

#endif // _ColorMap_
