#ifndef _ConvergenceLibrary_
#define _ConvergenceLibrary_

#include <vector>
#include <map>
#include <string>

#include "double/Convergence_dp_x86.hpp"
#include "double_gpu/Convergence_GPU.hpp"
#include "float_gpu/Convergence_GPU_float.hpp"
#include "double_gpu_julia/Convergence_GPU_julia.hpp"
#include "double_gpu_multibrot/Convergence_GPU_multibrot.hpp"
#include "double_gpu_mme/Convergence_GPU_mme.hpp"
#include "double_gpu_ship/Convergence_GPU_ship.hpp"
#include "double_gpu_mr/Convergence_GPU_mr.hpp"

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
        list.push_back( new Convergence_dp_x86        (nullptr, 255));
        list.push_back( new Convergence_GPU           (nullptr, 255));
        list.push_back( new Convergence_GPU_float     (nullptr, 255));
        list.push_back( new Convergence_GPU_julia     (nullptr, 255));
        list.push_back( new Convergence_GPU_multibrot (nullptr, 255));
        list.push_back( new Convergence_GPU_mme       (nullptr, 255));
        list.push_back( new Convergence_GPU_mr        (nullptr, 255));
        list.push_back( new Convergence_GPU_ship      (nullptr, 255));
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
