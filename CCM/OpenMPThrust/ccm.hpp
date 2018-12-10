#include "global.h"
#include <omp.h>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <random>

bool replacement = true;
bool enable_cpus = true;
bool enable_gpus = false;
bool enable_global_sort = false;

float dist_func(const std::vector<float>& A, const std::vector<float>& B){
    float dist = 0;
    for (auto a_iter = A.begin(), b_iter = B.begin(); a_iter != A.end(); ++a_iter, ++b_iter)
        dist += (*a_iter - *b_iter) * (*a_iter - *b_iter);

    return sqrt(dist);
}


std::vector<std::vector<size_t> > rank_matrix_cpu(std::vector<std::vector<float> >& distance_matrix, std::vector<size_t>& which_pred){

    std::vector<std::vector<size_t> > rank_matrix(distance_matrix.size(), std::vector<size_t>(distance_matrix[0].size()));
    for(auto& cur_pred: which_pred){
        std::vector<float>& values_cpu = distance_matrix[cur_pred];
        std::vector<size_t> indices_cpu(values_cpu.size());
        size_t n = 0;
        std::generate(indices_cpu.begin(), indices_cpu.end(), [&n]{return n++;});
        std::sort(indices_cpu.begin(), indices_cpu.end(), [&values_cpu](size_t i1, size_t i2){return values_cpu[i1] < values_cpu[i2];});
        std::copy(indices_cpu.begin(), indices_cpu.end(), rank_matrix[cur_pred].begin());
    }
    return rank_matrix;
}

// ccm function only execute for a part of parameter sets and input data
// OpenMP only

// before call this function: make sure the length of observations and targets are the same
// E, tau, lib_size is legal and lib_size should not larger than the length of observation
// num_samples should larger than 0

std::vector<float> ccm(std::vector<float> observations, std::vector<float> targets, size_t E, size_t tau, size_t lib_size, int num_samples){
    // return rhos
    timeval t1, t2;
    
    int num_cpus = omp_get_num_procs();
    size_t num_vectors = observations.size();
    if(lib_size > num_vectors/2){   // should be adjusted
    	enable_global_sort = true;
    }
    std::vector<std::vector<float> > lag_vector(num_vectors, std::vector<float>(E, qnan));
    
    // make lag vector
    for (size_t i = 0; i < (E - 1) * tau; i++)
           for (size_t j = 0; j < E; j++)
               if (i >= j * tau)
            	   lag_vector[i][j] = observations[i - j * tau];

    for(size_t i = (E - 1) * tau; i < num_vectors; i++)
    	for(size_t j = 0; j < E; j++)
    		lag_vector[i][j] = observations[i - j * tau];
    	
    // specify index array
    // cout << "specify index array: " << endl;
    std::vector<size_t> which_lib;
    std::vector<size_t> which_pred;
    size_t start_of_range = std::min((E - 1) * tau, num_vectors);
    size_t end_of_range = num_vectors - 1;
    for (size_t j = start_of_range; j <= end_of_range; j++){
    	which_lib.push_back(j);
        which_pred.push_back(j);
    }
    lib_size = std::min(which_lib.size(), lib_size);
    /*
    for(size_t i = 0; i < which_lib.size(); i++)cout << which_lib[i] << " ";
    cout << endl;
    for(size_t i = 0; i < which_pred.size(); i++)cout << which_pred[i] << " ";
    cout << endl;
    */

    //cout << "calculate the distance matrix: " << endl;
    std::vector<std::vector<float> > distance_matrix(num_vectors, std::vector<float>(num_vectors,  std::numeric_limits<float>::max()));
    for(auto& cur_pred: which_pred){
    	for(auto& cur_lib: which_lib){
    		distance_matrix[cur_pred][cur_lib] = dist_func(lag_vector[cur_pred], lag_vector[cur_lib]);
    		distance_matrix[cur_lib][cur_pred] = distance_matrix[cur_pred][cur_lib];
    	}
    }

    //then globally sort here and search  - purpose: mitigating bottlenecks in multi-node computing
    std::vector<std::vector<size_t> > rank_matrix;
    if(enable_global_sort){
        gettimeofday(&t1, NULL);
        // can be replaced using gpu index sorting
        if(enable_gpus){
            rank_matrix = rank_matrix_gpu(distance_matrix, which_pred);
        }else{
            rank_matrix = rank_matrix_cpu(distance_matrix, which_pred);
        }
        
        gettimeofday(&t2, NULL);
        unsigned long et_cpu = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
        std::cout << "sorting rank table running time =" << (float)et_cpu/(float)(1000000) << "s" << std::endl;
    }
    if(rank_matrix.size() == 0){
        std::cout <<  "sorting distance matrix error" << std::endl;
        exit(0);
    }

    gettimeofday(&t1, NULL);
    size_t seed = (size_t)(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
	std::uniform_int_distribution<uint32_t> lib_sampler(0, (unsigned int)(which_lib.size() - 1));
	std::uniform_real_distribution<double> unif_01(0, 1);
	size_t max_lib_size = which_lib.size();


    // final result to return
    std::vector<float > rhos;


    if(enable_cpus){
		omp_set_num_threads(num_cpus);
	}else{
		omp_set_num_threads(1);
	}
    #pragma omp parallel for
    for(size_t sample = 0; sample < num_samples; sample++){
		unsigned int cpu_thread_id = omp_get_thread_num();
		// cout << "initialize data for thread id: " << cpu_thread_id << " process: " << sample<< endl;

		// sample l size of observation index
		std::vector<size_t> sample_lib;

		if(replacement){
			// sample with replacement (default)
			for(auto l = 0; l < lib_size; l++)
				sample_lib.push_back(which_lib[lib_sampler(rng)]);
		}else{
			// sample without replacement   (refer from the algorithm from Knuth)
			sample_lib.assign(lib_size, 0);
			size_t m = 0;
			size_t t= 0;
			while(m < lib_size){
				if(double(max_lib_size - t) * unif_01(rng) >= double(lib_size - m)){
					++t;
				}
				else{
					sample_lib[m] = which_lib[t];
					++t; ++m;
				}
			}
		}

		// simplex prediction to compute predicted
		size_t cur_pred_index, num_ties;
		std::vector<float > weights;
		// initialize predicted
		std::vector<float> predicted(num_vectors, qnan);
		if(enable_global_sort){
			//find nearest neighbors with global sorted rank table (rank_matrix[cur_pred_index]) when lib size is large

			std::vector<int> dict(num_vectors, 0);
			for(auto l: sample_lib){
				dict[l] += 1;
			}

			for(size_t k = 0; k < which_pred.size(); k++){
				cur_pred_index = which_pred[k];
				// using rank matrix to find neighbors
				std::vector<size_t > neighbors;

				for(auto neighbor_index: rank_matrix[cur_pred_index]){
					if(cur_pred_index == neighbor_index){
						// filter index itself
						continue;
					}
					if((neighbors.size() >= E+1) || (distance_matrix[cur_pred_index][neighbor_index] == std::numeric_limits<float>::max())){
						break;
					}
					if(dict[neighbor_index] > 0){
						// may contain repeated elements
						for(int ele = 0; ele < dict[neighbor_index]; ele++){
							neighbors.push_back(neighbor_index);
						}
					}
				}
				// not necessary to handle tie here
				float min_distance = distance_matrix[cur_pred_index][neighbors[0]];
				size_t num_neighbors = std::min(E+1, neighbors.size());
				weights.assign(num_neighbors, min_weight);
				for(size_t t = 0; t < num_neighbors; t++){
					if(distance_matrix[cur_pred_index][neighbors[t]] == 0){
						// cout << "this is special case "<< endl;
						weights[t] = 1;
					}else if(min_distance != 0){
						weights[t] = std::fmax(exp(-distance_matrix[cur_pred_index][neighbors[t]] / min_distance), min_weight);
					}
				}

				// make prediction
				float total_weight = accumulate(weights.begin(), weights.end(), 0.0);
				predicted[cur_pred_index] = 0;
				for(size_t t = 0; t < num_neighbors; t++){
					predicted[cur_pred_index] += weights[t] * targets[neighbors[t]];
				}
				// normalized
				predicted[cur_pred_index] = predicted[cur_pred_index] / total_weight;
			}
		}else{
			//find nearest neighbors with local sorted table here: distance_matrix[cur_pred_index]
			for(size_t k = 0; k < which_pred.size(); k++){
				cur_pred_index = which_pred[k];
				// find nearest neighbors
				std::vector<size_t> neighbors;
				// TODO: filter the index itself
				std::copy_if(sample_lib.begin(), sample_lib.end(), back_inserter(neighbors), [&cur_pred_index](size_t i){return i != cur_pred_index;});
				// cout << lib.size() << endl;

				const std::vector<float>& distances = distance_matrix[cur_pred_index];

				std::sort(neighbors.begin(), neighbors.end(), [&distances](size_t i1, size_t i2){return distances[i1] < distances[i2];});

				// identify tie
				size_t tie_index = std::min(neighbors.size()-1, E);
				float tie_distance = distance_matrix[cur_pred_index][neighbors[tie_index]];
				size_t cur_tie_index = tie_index;
				for(; cur_tie_index < neighbors.size(); cur_tie_index++){
					if(distance_matrix[cur_pred_index][neighbors[cur_tie_index]] > tie_distance){
						cur_tie_index -= 1; // is the previous one
						break;
					}
				}
				// 0 - cur_tie_index   in neighbors   is the k nearest neighbor index range
				float min_distance = distance_matrix[cur_pred_index][neighbors[0]];
				weights.assign(cur_tie_index+1, min_weight);
				for(size_t t = 0; t < cur_tie_index; t++){
					if(distance_matrix[cur_pred_index][neighbors[t]] == 0){
						// cout << "this is special case "<< endl;
						weights[t] = 1;
					}else if(min_distance != 0){
						weights[t] = std::fmax(exp(-distance_matrix[cur_pred_index][neighbors[t]] / min_distance), min_weight);
					}
				}

				// identify tie exist and adjust weights
				if(cur_tie_index > tie_index){
					num_ties = 0;
					int left_tie = tie_index-1;
					while(left_tie >= 0 && distance_matrix[cur_pred_index][neighbors[left_tie]] == tie_distance){
						left_tie--;
						num_ties++;
					}
					int right_tie = tie_index+1;
					while(right_tie <= cur_tie_index && distance_matrix[cur_pred_index][neighbors[right_tie]] == tie_distance){
						right_tie++;
						num_ties++;
					}
					float tie_adj_factor = float(num_ties  - cur_tie_index + tie_index) / float(num_ties);

					for(size_t t = 0; t <= cur_tie_index; t++){
						if(distance_matrix[cur_pred_index][neighbors[t]] == tie_distance)
							weights[t] *= tie_adj_factor;
					}
				}
				// make prediction
				float total_weight = accumulate(weights.begin(), weights.end(), 0.0);
				predicted[cur_pred_index] = 0;
				for(size_t t = 0; t <= cur_tie_index; t++){
					predicted[cur_pred_index] += weights[t] * targets[neighbors[t]];
				}
				// normalized
				predicted[cur_pred_index] = predicted[cur_pred_index] / total_weight;
			}
		}

		// compute rho for every sample between predicted and targets array
		// this part can use gpu acceleration: two vectors operation -> a number
		size_t num_pred = 0;
		float sum_tar = 0;
		float sum_pred = 0;
		float sum_squared_tar = 0;
		float sum_squared_pred = 0;
		float sum_prod = 0;
		if(targets.size() == predicted.size()){
			for(size_t k = 0; k < targets.size(); k++){
				if(!std::isnan(predicted[k]) && !std::isnan(targets[k])){
					num_pred += 1;
					sum_tar += targets[k];
					sum_pred += predicted[k];
					sum_squared_tar += targets[k] * targets[k];
					sum_squared_pred += predicted[k] * predicted[k];
					sum_prod += targets[k] * predicted[k];
				}
			}
		}
		float rho = 0;
		float denominator = sqrt((sum_squared_tar * num_pred - sum_tar * sum_tar) * (sum_squared_pred * num_pred - sum_pred * sum_pred));
		float numerator = (sum_prod * num_pred - sum_tar * sum_pred);
		if(denominator != 0)
			rho = numerator / denominator;

		#pragma omp critical
		rhos.push_back(rho);
	 }
    return rhos;
}
