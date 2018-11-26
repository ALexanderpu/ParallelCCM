# CCM Parallization Techniques

## CUDAC++ - GPU accelerate

 GPU is that they do not share the same memory as the CPU. In other words, a GPU does not have direct access to the host memory. The host memory is generally larger, but slower than the GPU memory. To use a GPU, data must therefore be transferred from the main program to the GPU through the PCI bus, which has a much lower bandwidth than either memories. This means that managing data transfer between the host and the GPU will be of paramount importance. Transferring the data and the code onto the device is called offloading

###  global sorting  - thrust sort_by_key

###  alternative tool - OpenACC
  compiler reaches an OpenACC kernels directive, it will analyze the code in order to identify sections that can be parallelized. This often corresponds to the body of the loop

## OpenMP  - Multi-threads

## Spark/MPI   -   Multi-nodes
