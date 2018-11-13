#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

using namespace std;


template<typename T>
std::vector<std::size_t> tag_sort(const std::vector<T>& v)
{
    std::vector<std::size_t> result(v.size());
    std::iota(std::begin(result), std::end(result), 0);
    std::sort(std::begin(result), std::end(result),
            [&v](const double & lhs, const double & rhs)
            {
                return v[lhs] < v[rhs];
            }
    );
    return result;
}


int main(){
	int vec_size = 10000000;
	vector<double> values;
	vector<int> indices;
	double lower_bound = 0;
	double upper_bound = 10000;
	std::default_random_engine re;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);

	for(int i = 0; i < vec_size; i++){
		double a_random_double = unif(re);
		values.push_back(a_random_double);
		indices.push_back(i);
	}
	// generating values
	//for(double i : values)
	//	cout << i << " ";
	//cout << endl;
	vector<double> values_cpu(values);
	// sort using c++ stl vector
	auto start_cpu = std::chrono::high_resolution_clock::now();
	auto idices_cpu = tag_sort(values_cpu);
	auto finish_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_cpu = finish_cpu - start_cpu;
	cout <<" CPU single thread executing time: " << elapsed_cpu.count() << endl;
	// output
	//for (auto && elem:idxs)
	//	std::cout << elem << " : " << values[elem] << std::endl;

	thrust::device_vector<double> values_gpu(values);
	thrust::device_vector<int> indices_gpu(indices);
	// sort using cuda gpu
	auto start_gpu = std::chrono::high_resolution_clock::now();
	thrust::sort_by_key(values_gpu.begin(), values_gpu.end(), indices_gpu.begin());
	auto finish_gpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_gpu = finish_gpu - start_gpu;
	cout <<" GPU CUDA executing time: " << elapsed_gpu.count() << endl;

	//for(int i = 0; i < vec_size; i++)
	//	cout << indices[i] << " : " << values[i] << endl;
	bool are_equal = true;
	for(int i = 0; i < vec_size; i++)
		if (idices_cpu[i] != indices_gpu[i]){
			are_equal = false;
			break;
		}
	cout << are_equal << endl;


	return 0;
}
