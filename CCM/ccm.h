/**
 * @author Bo Pu
 * @email pubo.alexander@gmail.com
 * @create date 2018-08-27 07:26:37
 * @modify date 2018-08-27 07:26:37
 * @desc [description]
*/
#ifndef CCM_H
#define CCM_H
#include<bits/stdc++.h>

typedef std::pair<size_t, size_t> time_range;

class CCM{
private:
    std::vector<double> observations;
    std::vector<double> targets;
    size_t E;
    size_t tau;
    size_t nn;
    size_t lib_size;
    size_t num_samples;
    std::vector<std::vector<double> > lag_vectors;

    std::vector<size_t> which_lib;
    std::vector<size_t> which_pred;

    std::vector<double> predicted;

    size_t num_vectors;
    std::vector<std::vector<double> > distances;
    size_t seed;

    // output results
    std::vector<double> predicted_results;

    // multithread version
    bool is_multi_threads;
    int num_threads;
    void compute_distances_multi_wrapper();
    void forecast_multi_wrapper();

    //GPU staff
    bool is_gpu;
public:
    static const double qnan;
    CCM(bool GPU, bool MultiThreads);
    void init(const std::vector<double> &_observation, const std::vector<double> &_target, size_t _e, size_t _tau, size_t _samples, size_t _lib_size);
    void make_lag_vectors();
    bool is_lag_vec_valid(size_t vec_index);
    void specify_lib();
    double dist_func(const std::vector<double>& A, const std::vector<double>& B);
    void compute_distance_matrix(size_t start, size_t end);
    // simplex forecast
    void forecast(size_t start, size_t end);
    std::vector<size_t> find_nearest_neighbors(const std::vector<double> &dist);
    // compute correlation
    double compute_rho(const std::vector<double> &observation, const std::vector<double> &predict);

    std::vector<double> get_prediction();
    void run();
};

#endif