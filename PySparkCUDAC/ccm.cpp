/**
 * @author Bo Pu
 * @email pubo.alexander@gmail.com
 * @create date 2018-08-27 07:26:45
 * @modify date 2018-08-27 07:26:45
 * @desc [description]
*/
#include "ccm.h"
#include <thread>
#include <iomanip>
#include <random>
static const double min_weight = 0.000001;
const double CCM::qnan = std::numeric_limits<double>::quiet_NaN();

CCM::CCM(bool GPU, bool MultiThreads) {
    is_gpu = GPU;
    is_multi_threads = MultiThreads;
    num_threads = std::thread::hardware_concurrency();
    seed = (size_t)(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

std::vector<double> CCM::get_prediction(){
    return predicted_results;
}

void CCM::init(const std::vector<double> &_observation, const std::vector<double> &_target, size_t _e, size_t _tau, size_t _lib_size, size_t _samples){
    // empty containers
    if (!which_lib.empty())
        which_lib.clear();
    if (!which_pred.empty())
        which_pred.clear();
    if(!lag_vectors.empty())
        lag_vectors.clear();
    if(!distances.empty())
        distances.clear();
    if(!predicted_results.empty())
        predicted_results.clear();
    if(!targets.empty())
        targets.clear();
    if(!observations.empty())
        observations.clear();

    if (_observation.size() == _target.size())
    {
        observations.assign(_observation.begin(), _observation.end());
        targets.assign(_target.begin(), _target.end());
        num_vectors = observations.size();
    }else {
        exit(0);
    }

    if (_lib_size <= observations.size() && _lib_size > 0 && _e > 0 && _tau > 0 && _samples > 0)
    {
        num_samples = _samples;
        E = _e;
        tau = _tau;
        nn = E + 1;
        lib_size = _lib_size;
    }else {
        exit(0);
    }
}

void CCM::make_lag_vectors()
{
    lag_vectors.assign(num_vectors, std::vector<double>(E, qnan));

    //beginning of lagged vectors ? can we use these points                           1
    for (size_t i = 0; i < (unsigned int)((E - 1) * tau); ++i)
        for (size_t j = 0; j < (unsigned int)(E); ++j)
            if (i >= j * tau)
                lag_vectors[i][j] = observations[i - j * tau];

    // remaining lagged vectors
    for (size_t i = (unsigned int)((E - 1) * tau); i < num_vectors; ++i)
        for (size_t j = 0; j < (unsigned int)(E); ++j)
            lag_vectors[i][j] = observations[i - j * tau];
}

bool CCM::is_lag_vec_valid(const size_t vec_index)
{
    for(int  coord_index = 0; coord_index < lag_vectors[vec_index].size(); coord_index++)
        if (std::isnan(lag_vectors[vec_index][coord_index]))
            return false;
    return true;
}

// make sure all lib index is valid
void CCM::specify_lib()
{
    // set range of indices for which_lib and which_pred
    size_t start_of_range = std::min((E - 1) * tau, num_vectors - 1);
    size_t end_of_range = num_vectors - 1;


    for (size_t j = start_of_range; j <= end_of_range; ++j)
    {
        if (is_lag_vec_valid(j))
        {
            which_lib.push_back(j);
            which_pred.push_back(j);
        }
    }
}

// define distance func
double CCM::dist_func(const std::vector<double>& A, const std::vector<double>& B)
{
    double dist = 0;
    for (auto a_iter = A.begin(), b_iter = B.begin(); a_iter != A.end(); ++a_iter, ++b_iter)
    {
        dist += (*a_iter - *b_iter) * (*a_iter - *b_iter);
    }
    return sqrt(dist);
};

void CCM::compute_distance_matrix(size_t start, size_t end) {

    // calculate distance matrix based on data_vectors
    for(size_t curr_pos = start; curr_pos < end; ++curr_pos) {
        size_t curr_pred = which_pred[curr_pos];
        for(auto& curr_lib: which_lib)
        {
            if(std::isnan(distances[curr_pred][curr_lib]))
            {
                distances[curr_pred][curr_lib] = dist_func(lag_vectors[curr_pred], lag_vectors[curr_lib]);
                distances[curr_lib][curr_pred] = distances[curr_pred][curr_lib];
            }
        }
    }
}

/*
 * 1. make lag vectors
 * 2. specify range
 * 3. compute distance matrix
 * 4. specify lib and sample to forecast (which_lib)
 * 5. save result as predicted_results
 */
void CCM::run()
{
    // generate  lag_vectors
    make_lag_vectors();
    // generate  which_lib & which_pred
    specify_lib();

    // initialize distance matrix  based on num_vectors
    distances.assign(num_vectors, std::vector<double>(num_vectors, qnan));

    // output  distances
    if (is_multi_threads)
        compute_distances_multi_wrapper();
    else
        compute_distance_matrix(0, which_pred.size());

    std::vector<size_t> full_lib;
    full_lib.assign(which_lib.begin(), which_lib.end());

    // need to update with true random seed
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> lib_sampler(0, (unsigned int)(full_lib.size() - 1));
    std::uniform_real_distribution<double> unif_01(0, 1);

    which_lib.resize(lib_size, 0);
    for (size_t sample = 0; sample < num_samples; sample++)
    {
        // randomly generate sub_lib of observations with replace
        for (auto &lib : which_lib)
        {
            lib = full_lib[lib_sampler(rng)];
        }
        // initialize predicted
        predicted.assign(num_vectors, qnan);

        // output  predicted
        if (is_multi_threads)
            forecast_multi_wrapper();
        else
            forecast(0, which_pred.size());
        predicted_results.push_back(compute_rho(targets, predicted));
    }
    which_lib = full_lib;
}

// for target (predict)
void CCM::forecast(const size_t start, const size_t end)
{
    size_t curr_pred, effective_nn, num_ties;
    double min_distance, tie_distance;
    std::vector<double> weights;
    std::vector<size_t> nearest_neighbors;
    double tie_adj_factor, total_weight;

    for (size_t pos = start; pos < end; ++pos)
    {
        curr_pred = which_pred[pos];

        // find nearest neighbors
        nearest_neighbors = find_nearest_neighbors(distances[curr_pred]);
        effective_nn = nearest_neighbors.size();

        // compute weights
        min_distance = distances[curr_pred][nearest_neighbors[0]];
        weights.assign(effective_nn, min_weight);
        if (min_distance == 0)
        {
            for (size_t k = 0; k < effective_nn; ++k)
            {
                if (distances[curr_pred][nearest_neighbors[k]] == min_distance)
                    weights[k] = 1;
                else
                    break;
            }
        }
        else
        {
            for (size_t k = 0; k < effective_nn; ++k)
            {
                weights[k] = fmax(exp(-distances[curr_pred][nearest_neighbors[k]] / min_distance), min_weight);
            }
        }
        // identify ties and adjust weights
        if (effective_nn > nn) // ties exist
        {
            tie_distance = distances[curr_pred][nearest_neighbors.back()];

            // count ties
            num_ties = 0;
            for (auto &neighbor_index : nearest_neighbors)
            {
                if (distances[curr_pred][neighbor_index] == tie_distance)
                    num_ties++;
            }

            tie_adj_factor = double(num_ties + nn - effective_nn) / double(num_ties);

            // adjust weights
            for (size_t k = 0; k < nearest_neighbors.size(); ++k)
            {
                if (distances[curr_pred][nearest_neighbors[k]] == tie_distance)
                    weights[k] *= tie_adj_factor;
            }
        }

        // make prediction
        total_weight = accumulate(weights.begin(), weights.end(), 0.0);
        predicted[curr_pred] = 0;
        for (size_t k = 0; k < effective_nn; ++k)
            predicted[curr_pred] += weights[k] * targets[nearest_neighbors[k]];
        predicted[curr_pred] = predicted[curr_pred] / total_weight;
    }
}

std::vector<size_t> CCM::find_nearest_neighbors(const std::vector<double> &dist)
{
    std::vector<size_t> neighbors;
    std::vector<size_t> nearest_neighbors;
    double curr_distance;

    sort(which_lib.begin(), which_lib.end(), [&dist](size_t i1, size_t i2) { return dist[i1] < dist[i2]; });
    neighbors = which_lib;
    std::vector<size_t>::iterator curr_lib;

    // find nearest neighbors
    for (curr_lib = neighbors.begin(); curr_lib != neighbors.end(); ++curr_lib)
    {
        nearest_neighbors.push_back(*curr_lib);
        if (nearest_neighbors.size() >= nn)
            break;
    }
    if (curr_lib == neighbors.end())
        return nearest_neighbors;

    double tie_distance = dist[nearest_neighbors.back()];

    // check for ties
    for (++curr_lib; curr_lib != neighbors.end(); ++curr_lib)
    {
        if (dist[*curr_lib] > tie_distance) // distance is bigger
            break;
        nearest_neighbors.push_back(*curr_lib); // add to nearest neighbors
    }
    return nearest_neighbors;
}

double CCM::compute_rho(const std::vector<double> &observation, const std::vector<double> &predict)
{
    size_t num_pred = 0;
    double sum_obs = 0;
    double sum_pred = 0;
    double sum_squared_obs = 0;
    double sum_squared_pred = 0;
    double sum_prod = 0;
    if (observation.size() == predict.size())
    {
        for (size_t k = 0; k < observation.size(); k++)
        {
            if (!std::isnan(observation[k]) && !std::isnan(predict[k]))
            {
                num_pred += 1;
                sum_obs += observation[k];
                sum_pred += predict[k];
                sum_squared_obs += observation[k] * observation[k];
                sum_squared_pred += predict[k] * predict[k];
                sum_prod += observation[k] * predict[k];
            }
        }
    }

    double rho = 0;
    double denominator = sqrt((sum_squared_obs * num_pred - sum_obs * sum_obs) * (sum_squared_pred * num_pred - sum_pred * sum_pred));
    double numerator = (sum_prod * num_pred - sum_obs * sum_pred);
    if(denominator != 0)
        rho =  numerator / denominator;

    return rho;
}

/****************************************** multi-thread not working right now ********************************************************/
void CCM::compute_distances_multi_wrapper()
{
    size_t rows = which_pred.size() / num_threads;
    size_t extra = which_pred.size() % num_threads;
    size_t start = 0;
    size_t end = rows;
    std::vector<std::thread> workers;
    for (int t = 1; t <= num_threads; ++t)
    {
        if (t == num_threads)
            end += extra;
        workers.emplace_back(std::thread(&CCM::compute_distance_matrix, this, start, end));
        // set up rows for next calc
        start = end;
        end = start + rows;
    }
    for (auto &tt : workers)
        tt.join();
}

void CCM::forecast_multi_wrapper()
{
    size_t rows = which_pred.size() / num_threads;
    size_t extra = which_pred.size() % num_threads;
    size_t start = 0;
    size_t end = rows;
    std::vector<std::thread> workers;
    for (int t = 1; t <= num_threads; ++t)
    {
        if (t == num_threads)
            end += extra;
        workers.emplace_back(std::thread(&CCM::forecast, this, start, end));
        // set up rows for next calc
        start = end;
        end = start + rows;
    }
    // wait for threads to finish
    for (auto &tt : workers)
        tt.join();
}
