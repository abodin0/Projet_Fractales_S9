#ifndef _ConvergenceLibrary_
#define _ConvergenceLibrary_

#include <vector>
#include <map>
#include <string>

#include "double/Convergence_dp_x86.hpp"
#include "doublem/Convergence_dpm_x86.hpp"
#include "m256d/Convergence_m256d_x86.hpp"
#include "m256dm/Convergence_m256dm_x86.hpp"
#include "doublej/Convergence_dpj_x86.hpp"
#include "m128/Convergence_m128_x86.hpp"
#include "m128j/Convergence_m128j_x86.hpp"
#include "m256dj/Convergence_m256dj_x86.hpp"
#include "m256_float/Convergence_m256_float_x86.hpp"
#include "m256j_float/Convergence_m256j_float_x86.hpp"
<<<<<<< HEAD
#include "m256f_vect/Convergence_m256f_vect.hpp"
=======
#include "doublen/Convergence_dpn_x86.hpp"
#include "double_n/Convergence_dp_n_x86.hpp"
#include "doublebs/Convergence_dpbs_x86.hpp"
#include "doublemr/Convergence_dpmr_x86.hpp"
#include "doublemme/Convergence_dpmme_x86.hpp"
>>>>>>> 287149597b5ee14a70624fba3ab0332f107108ea

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
        list.push_back( new Convergence_m128_x86(nullptr, 255));
        list.push_back( new Convergence_m256_float_x86(nullptr, 255));
        list.push_back( new Convergence_dpj_x86(nullptr, 255));
        list.push_back( new Convergence_m256dj_x86(nullptr, 255));
        list.push_back( new Convergence_m128j_x86(nullptr, 255));
        list.push_back( new Convergence_m256j_float_x86(nullptr, 255));
        list.push_back( new Convergence_dpm_x86(nullptr, 255));
        list.push_back( new Convergence_m256dm_x86(nullptr, 255));
<<<<<<< HEAD
        list.push_back( new Convergence_m256f_vect(nullptr, 255));
=======
        list.push_back( new Convergence_dpn_x86(nullptr, 255));
        list.push_back( new Convergence_dp_n_x86(nullptr, 255));
        list.push_back( new Convergence_dpbs_x86(nullptr, 255));
        list.push_back( new Convergence_dpmr_x86(nullptr, 255));
        list.push_back( new Convergence_dpmme_x86(nullptr, 255));
>>>>>>> 287149597b5ee14a70624fba3ab0332f107108ea


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
