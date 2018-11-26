# CCM Parallization Techniques

## CUDAC++ - GPU accelerate

 GPU is that they do not share the same memory as the CPU. In other words, a GPU does not have direct access to the host memory. The host memory is generally larger, but slower than the GPU memory. To use a GPU, data must therefore be transferred from the main program to the GPU through the PCI bus, which has a much lower bandwidth than either memories. This means that managing data transfer between the host and the GPU will be of paramount importance. Transferring the data and the code onto the device is called offloading

###  1. global sorting  - thrust sort_by_key

In order to compare with the performance of gpu, you have to close the debug mode and compile as follows:

```console
nvcc -Xcompiler -fopenmp -std=c++11 -lgomp -o ccm OpenMP_thrust.cu
```

the performance benchmark:

-- time_series length: 

1000:

gpu sorting running time = 1.072339s

cpu sorting running time = 0.534552s

10000:

gpu sorting running time = 13.196652s

cpu sorting running time = 33.870094s


###  2. alternative tool - OpenACC
  compiler reaches an OpenACC kernels directive, it will analyze the code in order to identify sections that can be parallelized. This often corresponds to the body of the loop. we then need to rely on compiler feedback in order to identify regions it failed to parallelize when using    #pragma acc kernels.
  
  Another way to tell the compiler that loops iterations are independent is to specify it explicitly by using a different directive: loop, with the clause independent  --  #pragma acc loop independent
  
  Parallel loop vs kernel

PARALLEL LOOP 

    It is the programmer's responsibility to ensure that parallelism is safe
    Enables parallelization of sections that the compiler may miss
    Straightforward path from OpenMP

KERNEL

    It is the compiler's responsibility to analyze the code and determine what is safe to parallelize.
    A single directive can cover a large area of code
    The compiler has more room to optimize

## OpenMP  - Multi-threads
shared memory threading:  communicating through shared variables
use synchronization to protect data writing conflicts.  you have to wait writing then read
(Synchronization is used to impose order constraints and to protect access to shared data)

```cpp
#include<omp.h>


omp_lock_t lck;
omp_init_lock(&lck);

omp_set_num_threads(4);
#pragma omp parallel for
for(int sample = 0; sample < num_samples; sample++){
  int ID = omp_get_thread_num();
  
  // Mutual exclusion: Only one thread at a time can enter a critical region.
  #pragma omp critical
  vec.push_back(sample);
  
  // Atomic provides mutual exclusion but only applies to the read/update of a memory location
  #pragma omp atomic
  x += tmp;
  
  // Single is executed by only one thread (differentiate from critical: one thread one time for all threads)
  #pragma omp single
  { exchange_boundaries(); }
  
  // lock is the low level of critical
  omp_set_lock(&lck); //Wait here for your turn
  printf(“%d %d”, id, tmp);
  omp_unset_lock(&lck); //Release the lock so the next thread gets a turn
}

// reduction (op : list) .    The variables in “list” must be shared in the enclosing parallel region. 
double ave=0.0, A[MAX]; int i;
#pragma omp parallel for reduction (+:ave)
for (i=0;i< MAX; i++) {
   ave + = A[i];
}
ave = ave/MAX; 


```

## Spark/MPI   -   Multi-nodes

Mitigating Bottlenecks of cluster computing:  reducing the response time for large L jobs as we have to wait until the last job done in multi nodes.
How to:  build once and query multiple times for nearest neighbors finding
