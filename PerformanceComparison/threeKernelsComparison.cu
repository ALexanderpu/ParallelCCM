#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include<cfloat>

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>


using namespace std;

// compare radix sort

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

void compare_sort(int vec_size){
	// int vec_size = 100000;
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
	cout <<" CPU  SP  executing time (ms): " << elapsed_cpu.count()*1000 << endl;
	// output
	//for (auto && elem:idxs)
	//	std::cout << elem << " : " << values[elem] << std::endl;

	thrust::device_vector<double > values_gpu(values);

	thrust::device_vector<int> indices_gpu(indices);
	thrust::host_vector<double > indices_cpu(indices);
	// sort using cuda gpu
	auto start_gpu = std::chrono::high_resolution_clock::now();
	thrust::sort_by_key(values_gpu.begin(), values_gpu.end(), indices_gpu.begin());
	thrust::copy(indices_gpu.begin(), indices_gpu.end(), indices_cpu.begin());
	auto finish_gpu = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed_gpu = finish_gpu - start_gpu;
	cout <<" GPU CUDA executing time (ms): " << elapsed_gpu.count()*1000 << endl;

	//for(int i = 0; i < vec_size; i++)
	//	cout << indices[i] << " : " << values[i] << endl;
	/*
	bool are_equal = true;
	for(int i = 0; i < vec_size; i++)
		if (idices_cpu[i] != indices_gpu[i]){
			are_equal = false;
			break;
		}
	cout << are_equal << endl;
	*/
}

// m should be the power of 2 rank version
#define BlockSize 4  // for keper architecture  maxwell will be 16?




// compare pairwise of eculidean distance for A (n*dim)
__global__ void gpu_euclidian_distances(float *out, float *in, int n, int dim){
	__shared__ float Xs[BlockSize][BlockSize];
	__shared__ float Ys[BlockSize][BlockSize];
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int xBegin = bx * BlockSize * dim;
	int yBegin = by * BlockSize * dim;
	int yEnd = yBegin + dim - 1;
	int x, y, k, outIdx;
	float s = 0.0, tmp;

	for(y = yBegin, x = xBegin; y <= yEnd; y += BlockSize, x += BlockSize){
		Ys[ty][tx] = in[y + ty*dim + tx];
		Xs[tx][ty] = in[x + ty*dim + tx];
		__syncthreads();

		for(k = 0; k < BlockSize; k++){
			tmp = Ys[ty][k] - Xs[k][tx];
			s += tmp*tmp;
		}
		__syncthreads();
	}

	outIdx = by*BlockSize*n + ty*n + bx*BlockSize + tx;
	out[outIdx] = sqrtf(s);

}

void filling(float* matrix, int w, int h, float high){
	float low = 0.0f;
	for(int i = 0; i < w*h; i++){
		matrix[i] = low + static_cast<float>(rand() / static_cast<float>(RAND_MAX/(high - low)));
	}
}

void cpu_euclidian_distances(float* in, int n, int dim, float* out){
	for(int i = 0; i < n; i++){
		out[i*n + i] = 0.0;
		for(int j = i+1; j < n; j++){
			float dist = 0.0;
			for(int k = 0; k < dim; k++){
				float a = in[i*dim + k];
				float b = in[j*dim + k];
				dist += (a-b)*(a-b);
			}
			out[i*n + j] = sqrtf(dist);
			out[j*n + i] = sqrtf(dist);
		}
	}
}

void compare_distances(int vec_size, int E){

	// host memory allocation
	float *h_A, *h_result;
	cudaMallocHost((void**)&h_A, vec_size * E * sizeof(float));
	cudaMallocHost((void**)&h_result, vec_size * vec_size * sizeof(float));
	filling(h_A, vec_size, E, 10.0f);

	// device memory allocation
	float *d_A, *d_result;
	cudaMalloc((void**)&d_A, vec_size * E * sizeof(float));
	cudaMalloc((void**)&d_result, vec_size * vec_size * sizeof(float));

	auto start_gpu = std::chrono::high_resolution_clock::now();
	// data transfer to GPU
	cudaMemcpy(d_A, h_A, vec_size * E * sizeof(float), cudaMemcpyHostToDevice);
	// kernel launch
	dim3 block(BlockSize, BlockSize);
	dim3 grid(vec_size / BlockSize, vec_size / BlockSize);
	gpu_euclidian_distances <<<grid, block>>> (d_result, d_A, vec_size, E);
	// data transfer to CPU
	cudaMemcpy(h_result, d_result, vec_size * vec_size * sizeof(float), cudaMemcpyDeviceToHost);

	auto finish_gpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_gpu = finish_gpu - start_gpu;
	cout <<" GPU CUDA executing time (ms): " << elapsed_gpu.count()*1000 << endl;

	cudaFree(d_A);
	cudaFree(d_result);

	/*
	// check result first
	cout <<"input A: " << endl;
	for(int i = 0; i < vec_size; i++){
		for(int j = 0; j < E; j++){
			cout << h_A[i*E + j] << " ";
		}
		cout << endl;
	}


	cout << "GPU output: " << endl;
	for(int i = 0; i < vec_size; i++){
		for(int j = 0; j < vec_size; j++){
			cout << h_result[i*vec_size + j] << " ";
		}
		cout << endl;
	}
	*/


	// CPU
	auto start_cpu = std::chrono::high_resolution_clock::now();
	cpu_euclidian_distances(h_A, vec_size, E, h_result);
	auto finish_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_cpu = finish_cpu - start_cpu;
	cout <<" CPU  SP  executing time (ms): " << elapsed_cpu.count()*1000 << endl;

	/*
	cout << "CPU output: " << endl;
	for(int i = 0; i < vec_size; i++){
		for(int j = 0; j < vec_size; j++){
			cout << h_result[i*vec_size + j] << " ";
		}
		cout << endl;
	}
	*/
}



__global__ void pairwise_pearson_correlation(float *out, float *in, int n, int dim){
	__shared__ float Xs[BlockSize][BlockSize];
	__shared__ float Ys[BlockSize][BlockSize];
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int xBegin = bx * BlockSize * dim;
	int yBegin = by * BlockSize * dim;
	int yEnd = yBegin + dim - 1;
	int x, y, k, outIdx;
	float sumX, sumY, sumX2, sumY2, sumXY;
	float avgX, avgY, varX, varY, cov, rho;

	sumX = sumY = sumX2 = sumY2 = sumXY = 0.0;

	for(y = yBegin, x = xBegin; y <= yEnd; y += BlockSize, x += BlockSize){
		Ys[ty][tx] = in[y + ty*dim + tx];
		Xs[tx][ty] = in[x + ty*dim + tx];
		__syncthreads();

		for(k = 0; k < BlockSize; k++){
			sumX += Xs[k][tx];
			sumY += Ys[ty][k];
			sumX2 += Xs[k][tx]*Xs[k][tx];
			sumY2 += Ys[ty][k]*Ys[ty][k];
			sumXY += Xs[k][tx]*Ys[ty][k];
		}
		__syncthreads();
	}
	avgX = sumX/dim;
	avgY = sumY/dim;
	varX = (sumX2 - avgX*avgX*dim) / (dim - 1);
	varY = (sumY2 - avgY*avgY*dim) / (dim - 1);
	cov = (sumXY - avgX*avgY*dim) / (dim - 1);
	rho = cov / sqrtf(varX*varY);
	outIdx = by*BlockSize*n + ty*n + bx*BlockSize + tx;
	out[outIdx] = rho;
}


__global__ void gpu_pearson_correlation(float* out, float* A, float* B, int n, int dim){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n){
		// calculate the pearson between A and B
		float sumX = 0.0, sumY = 0.0, sumX2 = 0.0, sumY2 = 0.0, sumXY = 0.0;
		float rho = 0.0;
		for(int k = 0; k < dim; k++){
			sumX += A[idx*dim + k];
			sumY += B[idx*dim + k];
			sumX2 += A[idx*dim + k] * A[idx*dim + k];
			sumY2 += B[idx*dim + k] * B[idx*dim + k];
			sumXY += A[idx*dim + k] * B[idx*dim + k];
		}
		float denominator = sqrtf((sumX2 * dim - sumX * sumX) * (sumY2 * dim - sumY * sumY));
		float numerator = sumXY*dim - sumX*sumY;
		if(abs(denominator) > 1e-10)
			rho = numerator / denominator;
		out[idx] = rho;
	}
}


void cpu_pearson_correlation(float* out, float* A, float* B, int n, int dim){

	float rho;
	for(int i = 0; i < n; i++){
		float sumX = 0.0, sumY = 0.0, sumX2 = 0.0, sumY2 = 0.0, sumXY = 0.0;
		for(int k = 0; k < dim; k++){
			sumX += A[i*dim + k];
			sumY += B[i*dim + k];
			sumX2 += A[i*dim + k]*A[i*dim + k];
			sumY2 += B[i*dim + k]*B[i*dim + k];
			sumXY += A[i*dim + k]*B[i*dim + k];
		}
		rho = 0.0;
		float denominator = sqrtf((sumX2 * dim - sumX * sumX) * (sumY2 * dim - sumY * sumY));
		float numerator = sumXY*dim - sumX*sumY;
		if(abs(denominator) > 1e-10)
			rho = numerator / denominator;

		out[i] = rho;
	}
}

void compare_rho(int vec_size, int d){


	// host memory allocation and data filling
	float *h_A, *h_B, *h_result;
	cudaMallocHost((void**)&h_A, vec_size * d * sizeof(float));
	cudaMallocHost((void**)&h_B, vec_size * d * sizeof(float));
	cudaMallocHost((void**)&h_result, vec_size * sizeof(float));
	filling(h_A, vec_size, d, 10.0f);
	filling(h_B, vec_size, d, 10.0f);

	// device memory allocation
	float *d_A, *d_B, *d_result;
	cudaMalloc((void**)&d_A, vec_size * d * sizeof(float));
	cudaMalloc((void**)&d_B, vec_size * d * sizeof(float));
	cudaMalloc((void**)&d_result, vec_size * sizeof(float));


	auto start_gpu = std::chrono::high_resolution_clock::now();
	// data transfer to GPU
	cudaMemcpy(d_A, h_A, vec_size * d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, vec_size * d * sizeof(float), cudaMemcpyHostToDevice);

	// kernel launch
	dim3 block(BlockSize);
	dim3 grid(vec_size / BlockSize);
	gpu_pearson_correlation <<<grid, block>>> (d_result, d_A, d_B, vec_size, d);
	// data transfer to CPU
	cudaMemcpy(h_result, d_result, vec_size * sizeof(float), cudaMemcpyDeviceToHost);

	auto finish_gpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_gpu = finish_gpu - start_gpu;
	cout <<" GPU CUDA executing time (ms): " << elapsed_gpu.count()*1000 << endl;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_result);

	/*
	// check result first
	cout <<"input A: " << endl;
	for(int i = 0; i < vec_size; i++){
		for(int j = 0; j < d; j++){
			cout << h_A[i*d + j] << " ";
		}
		cout << endl;
	}

	cout <<"input B: " << endl;
	for(int i = 0; i < vec_size; i++){
		for(int j = 0; j < d; j++){
			cout << h_B[i*d + j] << " ";
		}
		cout << endl;
	}


	cout << "GPU output: " << endl;
	for(int i = 0; i < vec_size; i++){
		cout << h_result[i] << " ";
	}
	cout << endl;
	*/



	// CPU
	auto start_cpu = std::chrono::high_resolution_clock::now();
	cpu_pearson_correlation(h_result, h_A, h_B, vec_size, d);
	auto finish_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_cpu = finish_cpu - start_cpu;
	cout <<" CPU  SP  executing time (ms): " << elapsed_cpu.count()*1000 << endl;

	/*
	cout << "CPU output: " << endl;
	for(int i = 0; i < vec_size; i++){
		cout << h_result[i] << " ";
	}
	cout << endl;
	*/

}




int main(){

	vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
	vector<int> dimensions = {2, 4, 8};
	
	cout << "comparing the performance difference of pairwise distances calculation" << endl;
	for(auto dim: dimensions){
		cout << "dimension: " << dim << endl;
		for(auto size: sizes){
			compare_distances(size, dim);
		}
	}

	cout << "comparing the performance difference of radix sort" << endl;
	for(auto size: sizes){
		compare_sort(size);
	}
	

	cout << "comparing the performance difference of Pearson correlation" << endl;
	for(auto dim: dimensions){
		cout << "dimension: " << dim << endl;
		for(auto size: sizes){
			compare_rho(size, dim);
		}
	}
	return 0;
}
