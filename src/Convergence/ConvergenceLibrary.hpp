#ifndef _ConvergenceLibrary_
#define _ConvergenceLibrary_

#include <vector>
#include <map>
#include <string>

#include "double/Convergence_dp_x86.hpp"
#include "double_dp_omp_x86/Convergence_dp_omp_x86.hpp"
#include "double_gpu/Convergence_GPU.hpp"
#include "float_gpu/Convergence_GPU_float.hpp"
#include "double_gpu_julia/Convergence_GPU_julia.hpp"
#include "double_gpu_multibrot/Convergence_GPU_multibrot.hpp"
#include "double_gpu_mme/Convergence_GPU_mme.hpp"
#include "double_gpu_ship/Convergence_GPU_ship.hpp"
#include "double_gpu_mr/Convergence_GPU_mr.hpp"
#include "doublem/Convergence_dpm_x86.hpp"
#include "m256d/Convergence_m256d_x86.hpp"
#include "m256dm/Convergence_m256dm_x86.hpp"
#include "doublej/Convergence_dpj_x86.hpp"
#include "m128/Convergence_m128_x86.hpp"
#include "m128j/Convergence_m128j_x86.hpp"
#include "m256dj/Convergence_m256dj_x86.hpp"
#include "m256_float/Convergence_m256_float_x86.hpp"
#include "m256j_float/Convergence_m256j_float_x86.hpp"
#include "doublen/Convergence_dpn_x86.hpp"
#include "double_n/Convergence_dp_n_x86.hpp"
#include "doublebs/Convergence_dpbs_x86.hpp"
#include "doublemr/Convergence_dpmr_x86.hpp"
#include "doublemme/Convergence_dpmme_x86.hpp"

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
        // Mandelbrot
        list.push_back( new Convergence_dp_x86          (nullptr, 255)); // pas d'optimisation
        list.push_back( new Convergence_dp_omp_x86      (nullptr, 255)); // OpenMP
        list.push_back( new Convergence_m256d_x86       (nullptr, 255)); // SIMD(256) + OpenMP
        list.push_back( new Convergence_m256_float_x86  (nullptr, 255)); // SIMD(256) + OpenMP en float
        list.push_back( new Convergence_m128_x86        (nullptr, 255)); // SIMD(128) + OpenMP en float
        list.push_back( new Convergence_GPU             (nullptr, 255)); // GPU
        list.push_back( new Convergence_GPU_float       (nullptr, 255)); // GPU float

        // Julia
        list.push_back( new Convergence_dpj_x86         (nullptr, 255)); // OpenMP
        list.push_back( new Convergence_m256dj_x86      (nullptr, 255)); // SIMD(256) + OpenMP
        list.push_back( new Convergence_m256j_float_x86 (nullptr, 255)); // SIMD(256) + OpenMP en float
        list.push_back( new Convergence_m128j_x86       (nullptr, 255)); // SIMD(128) + OpenMP
        list.push_back( new Convergence_GPU_julia       (nullptr, 255)); // GPU

        // Multibrot (puissance 3)
        list.push_back( new Convergence_dpm_x86         (nullptr, 255)); // OpenMP
        list.push_back( new Convergence_m256dm_x86      (nullptr, 255)); // SIMD(256) + OpenMP
        list.push_back( new Convergence_GPU_multibrot   (nullptr, 255)); // GPU
        
        // Burning ship
        list.push_back( new Convergence_dpbs_x86        (nullptr, 255)); // OpenMP
        list.push_back( new Convergence_GPU_ship        (nullptr, 255)); // GPU

        // Madame
        list.push_back( new Convergence_dpmme_x86       (nullptr, 255)); // OpenMP
        list.push_back( new Convergence_GPU_mme         (nullptr, 255)); // GPU

        // Monsieur
        list.push_back( new Convergence_dpmr_x86        (nullptr, 255)); // OpenMP
        list.push_back( new Convergence_GPU_mr          (nullptr, 255)); // GPU
        
        // Multibrot puissance n
        list.push_back( new Convergence_dpn_x86         (nullptr, 255)); // pas d'omptimisation

        // Mandelbar
        list.push_back( new Convergence_dp_n_x86        (nullptr, 255)); // pas d'omptimisation
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
