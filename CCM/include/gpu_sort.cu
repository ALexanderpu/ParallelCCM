#ifndef GPU_SORT_CU
#define GPU_SORT_CU
#include "global.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

std::vector<std::vector<size_t> > rank_matrix_gpu(std::vector<std::vector<float> >& distance_matrix, std::vector<size_t> which_pred){

    std::vector<std::vector<size_t> > rank_matrix(distance_matrix.size(), std::vector<size_t>(distance_matrix[0].size()));
    
    // gpu sort here
    for(auto& cur_pred: which_pred){
        thrust::device_vector<float> values_gpu(distance_matrix[cur_pred]);
        thrust::device_vector<size_t> indices_gpu(distance_matrix[cur_pred].size());
        thrust::sequence(indices_gpu.begin(), indices_gpu.end());
        thrust::sort_by_key(values_gpu.begin(), values_gpu.end(), indices_gpu.begin()); // this function will change values and indices at the same time
        thrust::copy(indices_gpu.begin(), indices_gpu.end(), rank_matrix[cur_pred].begin());  // copy device to host
    }
    return rank_matrix;
}

#endif

