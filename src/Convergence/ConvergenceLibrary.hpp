#ifndef _ConvergenceLibrary_
#define _ConvergenceLibrary_

#include <vector>
#include <map>
#include <string>

#include "double/Convergence_dp_x86.hpp"
#include "m256d/Convergence_m256d_x86.hpp"
#include "doublej/Convergence_dpj_x86.hpp"
#include "m128/Convergence_m128_x86.hpp"
#include "m256dj/Convergence_m256dj_x86.hpp"

class ConvergenceLibrary {
private:
    std::vector<            Convergence*> list;
    std::map   <std::string,Convergence*> dico;
    std::map   <std::string,int         > indx;

    int counter;

public:

    ConvergenceLibrary(){

        //
        //
        //
        list.push_back( new Convergence_dp_x86(nullptr, 255));
        list.push_back( new Convergence_m256d_x86(nullptr, 255));
        list.push_back( new Convergence_dpj_x86(nullptr, 255));
        list.push_back( new Convergence_m128_x86(nullptr, 255) );
        list.push_back( new Convergence_m256dj_x86(nullptr, 255));


        //
        //
        //
        for(int i=0; i<list.size(); i++){
            dico[ list[i]->name() ] = list[i];
            indx[ list[i]->name() ] =      i ;
        }

        counter = 0;

    }

    virtual ~ConvergenceLibrary(){

        for(int i=0; i<list.size(); i++)
            delete list[i];

    }

    Convergence* get(std::string name){
        counter = indx[name];
        return get();
    }

    Convergence* get(int num){
        counter = num;
        return get();
    }

    Convergence* get(){
//        printf("GET %d : (%p) name = %s \n", counter, list[counter], list[counter]->name().c_str());
        printf("Convergence :: get (%d) : (%p) name = %s \n", counter, list[counter], list[counter]->name().c_str());
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
