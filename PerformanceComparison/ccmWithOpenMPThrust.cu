#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <time.h>
#include <sys/time.h>
#include <random>
#include <bits/stdc++.h>
#include <iomanip>
using namespace std;

// #define DSIZE 1000;

static const double min_weight = 0.000001;

const double qnan = std::numeric_limits<double>::quiet_NaN();

pair<vector<double>, vector<double> > parse_csv(std::string &csvfile){
    std::ifstream data(csvfile);
    std::string line;
    std::vector<std::vector<double> > csvdata;
    unsigned long length = 0;
    while(getline(data, line)){
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> parsedRow;
        while(getline(lineStream,cell,',')) //include head
        {
            parsedRow.push_back(strtof(cell.c_str(), 0));
        }
        length += 1;
        csvdata.push_back(parsedRow);
    }
    vector<double> x, y;
    cout << length << " size "<< endl;
    for(int i = 1; i < length; i++){
        x.push_back(csvdata[i][1]);
        y.push_back(csvdata[i][2]);
    }
    return make_pair(x, y);
}


double dist_func(const std::vector<double>& A, const std::vector<double>& B){
    double dist = 0;
    for (auto a_iter = A.begin(), b_iter = B.begin(); a_iter != A.end(); ++a_iter, ++b_iter)
        dist += (*a_iter - *b_iter) * (*a_iter - *b_iter);

    return sqrt(dist);
}


template<typename T>
vector<size_t> index_sort(const vector<T>& v){
	vector<size_t> result(v.size());
	iota(begin(result), end(result), 0);
	sort(begin(result), end(result), [&v](const double &lhs, const double &rhs){return v[lhs] < v[rhs];});
	return result;
}

int main(int argc, char *argv[])
{
    timeval t1, t2;
    int num_gpus = 0;   // number of CUDA GPUs
    int num_cpus = 0;

    printf("%s Starting...\n\n", argv[0]);

    // determine the number of CUDA capable GPUs
    cudaGetDeviceCount(&num_gpus);
    // determine the number of cpu
    num_cpus = omp_get_num_procs();

    if (num_gpus < 1){
        cout << "no CUDA capable devices were detected" << endl;
        return 1;
    }

    // display CPU and GPU configuration
    printf("number of host CPUs:\t%d\n", num_cpus);
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    typedef thrust::device_vector<int> dvec;
    typedef dvec *p_dvec;
    std::vector<p_dvec> dvecs;
    /*
    for(unsigned int i = 0; i < num_gpus; i++) {
      cudaSetDevice(i);
      p_dvec temp = new dvec(DSIZE);
      dvecs.push_back(temp);
    }
    */

    // access the matrix
    vector<double > observations;
    vector<double > targets;
    string csvfile = "/home/bo/Documents/CCM-Parralization/TestCSVData/test_float_10000.csv";
    std::tie(observations, targets) = parse_csv(csvfile);
    //observations = {3, 4, 5, 6, 7, 1, 2, 5, 2};
    //targets = {7, 5, 8, 1, 3, 4, 3, 2, 1};


    cout << "print observations" << endl;
    for(auto ele: observations) cout << ele << " ";
    cout << endl;
    /*
    cout << "print targets" << endl;
    for(auto ele: targets) cout << ele << " ";
    cout << endl;
	*/



    if(observations.size() != targets.size()){
    	cout << "input sequence length not match" << endl;
    	return 1;
    }
    size_t num_vectors = observations.size();
    int num_samples = 250;
    size_t E = 3;
    size_t tau = 1;
    size_t lib_size = min((size_t)300, num_vectors);
    bool enable_cpus = true;
    bool enable_global_sort = true;
    bool enable_gpu = true;
    bool replacement = true;

    vector<vector<double>> lag_vector(num_vectors, vector<double>(E, qnan));
    // make lag vector
    for (size_t i = 0; i < (E - 1) * tau; ++i)
           for (size_t j = 0; j < E; ++j)
               if (i >= j * tau)
            	   lag_vector[i][j] = observations[i - j * tau];

    for(size_t i = (E-1)*tau; i < num_vectors; i++){
    	for(size_t j = 0; j < E; j++){
    		lag_vector[i][j] = observations[i - j*tau];
    	}
    }

    // print lag  vector
    for(size_t j = 0; j < E; j++){
    	for(size_t i = 0; i < num_vectors; i++){
    		cout << lag_vector[i][j] << " ";
    	}
    	cout << endl;
    }

    // specify index array
    cout << "specify index array: " << endl;
    vector<size_t> which_lib;
    vector<size_t> which_pred;
    size_t start_of_range = std::min((E - 1) * tau, num_vectors - 1);
    size_t end_of_range = num_vectors - 1;
    for (size_t j = start_of_range; j <= end_of_range; ++j){
    	which_lib.push_back(j);
        which_pred.push_back(j);
    }
    lib_size = min(which_lib.size(), lib_size);

    for(size_t i = 0; i < which_lib.size(); i++)cout << which_lib[i] << " ";
    cout << endl;
    for(size_t i = 0; i < which_pred.size(); i++)cout << which_pred[i] << " ";
    cout << endl;



    // compute distance matrix using lag_vector  N*E  (contain nan)
    cout << "calculate the distance matrix: " << endl;
    vector<vector<double> > distance_matrix(num_vectors, vector<double>(num_vectors,  std::numeric_limits<double>::max()));
    for(auto& cur_pred: which_pred){
    	for(auto& cur_lib: which_lib){
    		distance_matrix[cur_pred][cur_lib] = dist_func(lag_vector[cur_pred], lag_vector[cur_lib]);
    		distance_matrix[cur_lib][cur_pred] = distance_matrix[cur_pred][cur_lib];
    	}
    }
    /*
    // print the distance matrix
	cout << "distance matrix: " << endl;
    for(size_t i = 0; i < distance_matrix.size(); i++){
		for(size_t j = 0; j < distance_matrix[0].size(); j++){
			cout << distance_matrix[i][j] << " ";
		}
		cout << endl;
	}
	*/


    //TODO:  then globally sort here and search  - purpose: mitigating bottlenecks in multi-node computing   cpu; network utilization
    // test rank_matrix here
    vector<vector<size_t> > rank_matrix;
    if(enable_global_sort){

    	rank_matrix(num_vectors, vector<size_t>(num_vectors, 0));
		if(enable_gpu){
			gettimeofday(&t1,NULL);
			// gpu sort here
			for(auto& cur_pred: which_pred){
				thrust::device_vector<double> values_gpu(distance_matrix[cur_pred]);
				thrust::device_vector<size_t> indices_gpu(distance_matrix[cur_pred].size());
				thrust::sequence(indices_gpu.begin(), indices_gpu.end());
				thrust::sort_by_key(values_gpu.begin(), values_gpu.end(), indices_gpu.begin()); // this function will change values and indices at the same time
				thrust::copy(indices_gpu.begin(), indices_gpu.end(), rank_matrix[cur_pred].begin());  // copy device to host
			}

			gettimeofday(&t2,NULL);
			unsigned long et_gpu = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
			printf("gpu sorting running time = %fs\n", (float)et_gpu/(float)(1000000));

		}else{
		//if(enable_gpu){
			gettimeofday(&t1,NULL);
			for(auto& cur_pred: which_pred){
				vector<double>& values_cpu = distance_matrix[cur_pred];
				vector<size_t> indices_cpu(values_cpu.size());
				size_t n = 0;
				std::generate(indices_cpu.begin(), indices_cpu.end(), [&n]{return n++;});
				std::sort(indices_cpu.begin(), indices_cpu.end(), [&values_cpu](size_t i1, size_t i2){return values_cpu[i1] < values_cpu[i2];});
				std::copy(indices_cpu.begin(), indices_cpu.end(), rank_matrix[cur_pred].begin());
			}
			gettimeofday(&t2,NULL);
			unsigned long et_cpu = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
			printf("cpu sorting running time = %fs\n", (float)et_cpu/(float)(1000000));
		}
		/*
		// print the rank matrix
		cout << "rank matrix: " << endl;
		for(size_t i = 0; i < rank_matrix.size(); i++){
			for(size_t j = 0; j < rank_matrix[0].size(); j++){
				cout << rank_matrix[i][j] << " ";
			}
			cout << endl;
		}
		*/
    }

	gettimeofday(&t1,NULL);
    size_t seed = (size_t)(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
	std::uniform_int_distribution<uint32_t> lib_sampler(0, (unsigned int)(which_lib.size() - 1));
	std::uniform_real_distribution<double> unif_01(0, 1);
	size_t max_lib_size = which_lib.size();


    gettimeofday(&t2,NULL);
	unsigned long et1 = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
	printf("stage 1 running time = %fs\n", (float)et1/(float)(1000000));


	gettimeofday(&t1,NULL);

	if(enable_cpus){
		omp_set_num_threads(num_cpus);
	}else{
		omp_set_num_threads(1);
	}
	// final result
	vector<double > rhos;

    #pragma omp parallel for
	for(size_t sample = 0; sample < num_samples; sample++){
		unsigned int cpu_thread_id = omp_get_thread_num();
		// cout << "initialize data for thread id: " << cpu_thread_id << " process: " << sample<< endl;

		// sample l size of observation index
		vector<size_t> lib_contain_indices;

		if(replacement){
			// sample with replacement (default)
			for(auto l = 0; l < lib_size; l++)
				lib_contain_indices.push_back(which_lib[lib_sampler(rng)]);
		}else{
			// sample without replacement   (refer from the algorithm from Knuth)
			lib_contain_indices.assign(lib_size, 0);
			size_t m = 0;
			size_t t= 0;
			while(m < lib_size){
				if(double(max_lib_size - t) * unif_01(rng) >= double(lib_size - m)){
					++t;
				}
				else{
					lib_contain_indices[m] = which_lib[t];
					++t; ++m;
				}
			}
		}
		// find nearest neighbor here?


		// initialize predicted
		vector<double> predicted(num_vectors, qnan);
		// simplex prediction to compute predicted
		size_t cur_pred_index, num_ties;
		std::vector<double > weights;
		for(size_t k = 0; k < which_pred.size(); k++){
			cur_pred_index = which_pred[k];

			vector<size_t> lib;
			// TODO: filter the index itself
			std::copy_if(lib_contain_indices.begin(), lib_contain_indices.end(), back_inserter(lib), [&cur_pred_index](size_t i){return i != cur_pred_index;});
			// cout << lib.size() << endl;

			if(!enable_global_sort){
				//find nearest neighbors without global sorted table here: distance_matrix[cur_pred_index]
				std::vector<size_t> neighbors;
				const vector<double>& distances = distance_matrix[cur_pred_index];
				std::sort(lib.begin(), lib.end(), [&distances](size_t i1, size_t i2){return distances[i1] < distances[i2];});
			}else{
				//find nearest neighbors with global sorted table
				// when lib size is large


			}


			// identify tie
			size_t tie_index = min(lib.size()-1, E);
			double tie_distance = distance_matrix[cur_pred_index][lib[tie_index]];
			size_t cur_tie_index = tie_index;
			for(; cur_tie_index < lib.size(); cur_tie_index++){
				if(distance_matrix[cur_pred_index][lib[cur_tie_index]] > tie_distance){
					cur_tie_index -= 1; // is the previous one
					break;
				}
			}
			// 0 - cur_tie_index   in lib   is the neighbor index range

			double min_distance = distance_matrix[cur_pred_index][lib[0]];
			weights.assign(cur_tie_index+1, min_weight);
			for(size_t i = 0; i < cur_tie_index; i++){
				if(distance_matrix[cur_pred_index][lib[i]] == 0){
					cout << "this is special case "<< endl;
					weights[i] = 1;
				}else if(min_distance != 0){
					weights[i] = fmax(exp(-distance_matrix[cur_pred_index][lib[i]] / min_distance), min_weight);
				}
			}

			// identify tie exist and adjust weights
			if(cur_tie_index > tie_index){
				num_ties = 0;
				int left_tie = tie_index-1;
				while(left_tie >= 0 && distance_matrix[cur_pred_index][lib[left_tie]] == tie_distance){
					left_tie--;
					num_ties++;
				}
				int right_tie = tie_index+1;
				while(right_tie <= cur_tie_index && distance_matrix[cur_pred_index][lib[right_tie]] == tie_distance){
					right_tie--;
					num_ties++;
				}
				double tie_adj_factor = double(num_ties  - cur_tie_index + tie_index) / double(num_ties);

				for(size_t t = 0; t <= cur_tie_index; t++){
					if(distance_matrix[cur_pred_index][lib[t]] == tie_distance)
						weights[t] *= tie_adj_factor;
				}
			}

			// make prediction
			double total_weight = accumulate(weights.begin(), weights.end(), 0.0);
			predicted[cur_pred_index] = 0;
			for(size_t t = 0; t <= cur_tie_index; t++){
				predicted[cur_pred_index] += weights[t] * targets[lib[t]];
			}
			// normalized
			predicted[cur_pred_index] = predicted[cur_pred_index] / total_weight;
		}

		// compute rho for every sample between predicted and targets
		size_t num_pred = 0;
		double sum_tar = 0;
		double sum_pred = 0;
		double sum_squared_tar = 0;
		double sum_squared_pred = 0;
		double sum_prod = 0;
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
		double rho = 0;
		double denominator = sqrt((sum_squared_tar * num_pred - sum_tar * sum_tar) * (sum_squared_pred * num_pred - sum_pred * sum_pred));
		double numerator = (sum_prod * num_pred - sum_tar * sum_pred);
		if(denominator != 0)
			rho = numerator / denominator;

		#pragma omp critical
		rhos.push_back(rho);
	 }




        /*
        thrust::host_vector<int> data(DSIZE);
        thrust::generate(data.begin(), data.end(), rand);

        // copy data
        // critical part: should enter into task
        for (unsigned int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            thrust::copy(data.begin(), data.end(), (*(dvecs[i])).begin());
          }

        printf("start sort\n");


        cudaSetDevice(cpu_thread_id);
        thrust::sort((*(dvecs[cpu_thread_id])).begin(), (*(dvecs[cpu_thread_id])).end());
        cudaDeviceSynchronize();
		*/

	cout << " the result size: " << rhos.size() << endl;
	for(size_t i = 0; i < rhos.size(); i++){
		cout << rhos[i] << " ";
	}
	cout << endl;

    gettimeofday(&t2,NULL);
    unsigned long et2 = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
    printf("stage 2 runing time = %fs\n", (float)et2/(float)(1000000));


    printf("total runing time = %fs\n", (float)(et1+et2)/(float)(1000000));

    /*
    unsigned long et = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
    if (cudaSuccess != cudaGetLastError())
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    printf("sort time = %fs\n", (float)et/(float)(1000000));
    // check results
    thrust::host_vector<int> result(DSIZE);
    for (int i = 0; i < num_gpus; i++)
    {
        cudaSetDevice(i);
        thrust::copy((*(dvecs[i])).begin(), (*(dvecs[i])).end(), result.begin());
        for (int j = 0; j < DSIZE; j++)
          if (data[j] != result[j]) { printf("mismatch on device %d at index %d, host: %d, device: %d\n", i, j, data[j], result[j]); return 1;}
    }
    */
    printf("Success\n");
    return 0;
}
