# CCM Parallization Techniques

## CUDAC++ - GPU accelerate

 GPU is that they do not share the same memory as the CPU. In other words, a GPU does not have direct access to the host memory. The host memory is generally larger, but slower than the GPU memory. To use a GPU, data must therefore be transferred from the main program to the GPU through the PCI bus, which has a much lower bandwidth than either memories. This means that managing data transfer between the host and the GPU will be of paramount importance. Transferring the data and the code onto the device is called offloading

###  1. global sorting  - thrust sort_by_key
performance benchmark:

-- time_series length: 
1000
gpu sorting running time = 1.072339s
cpu sorting running time = 0.534552s

10000  
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
```console
nvcc -Xcompiler -fopenmp -std=c++11 -lgomp -o ccm OpenMP_thrust.cu
```

## Spark/MPI   -   Multi-nodes
