#ifndef GLOBAL_H
#define GLOBAL_H

#include <bits/stdc++.h>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <random>
static float min_weight = 0.000001;

const float qnan = std::numeric_limits<float>::quiet_NaN();

std::pair<std::vector<float>, std::vector<float> > parse_csv(std::string &csvfile, std::string &xname, std::string &yname);

// implemented at ccm.hpp
class CCMParallel{
    private:
        bool replacement;
        bool enable_cpus;
        bool enable_gpus;
        bool enable_global_sort;
        int num_cpus;

        // obtain running time
        timeval t1, t2;
        unsigned long et_cpu;

        // ccm related private member
        std::vector<float> _observations;
        std::vector<float> _targets;
        size_t _E, _tau, _num_vectors;

        std::vector<std::vector<float> > _distance_matrix;
        std::vector<std::vector<size_t> > _rank_matrix;

        std::vector<size_t> which_lib;
        std::vector<size_t> which_pred;

        float dist_func(const std::vector<float>& A, const std::vector<float>& B);
        std::vector<std::vector<float> > distance_matrix_cpu(std::vector<std::vector<float> >& lag_vector);

        std::vector<std::vector<size_t> > rank_matrix_cpu();
    
    public:
        CCMParallel();
        void setGPU(bool GPUStatus);

        bool init(std::vector<float>& observations, std::vector<float>& targets, size_t E, size_t tau); // make global sort only once
        // check if you should init the embeded space
        std::vector<float> ccm(std::vector<float>& observations, std::vector<float>& targets, size_t E, size_t tau, size_t lib_size, int num_samples);
};

// implemented at a different file
std::vector<std::vector<size_t> > rank_matrix_gpu(std::vector<std::vector<float> >& distance_matrix, std::vector<size_t> which_pred);

#endif