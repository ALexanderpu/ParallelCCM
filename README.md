# Applying Parallel Techniques on Convergent Cross Mapping (CCM)
---
## Overview

This repository is the parallel version of R package `rEDM` function `ccm`, which was originally implemented by [ha0ye](https://github.com/ha0ye/rEDM) to infer the causality using time series. Different parallel techniques used in the code: GPU CUDA, Spark and MPI/OpenMP. Other similar computation-intensive algorithms which require parallelizing can also refer this repository. This implementation can improve the execution speed when trying an wide range of parameters: E, tau, lib_sizes. The output is csv file format and there is R script which can transfer the csv file into the density plots. By observing the pattern of these plots, we can infer the causality with confidence.

``` r
install.packages("rEDM")
data(sardine_anchovy_sst)
anchovy_xmap_sst <- ccm(sardine_anchovy_sst, E = 3, lib_column = "anchovy", 
    target_column = "np_sst", lib_sizes = seq(10, 80, by = 10), num_samples = 100, 
    random_libs = TRUE, replace = TRUE)
```

## Project Layout

1. **ccm.cfg** The configuration file defines the input and output file paths and parameters settings for the parallel ccm implementation. This cfg file contains 4 sections (paths, inputs, parameters and options), and in each section there are several key-value pairs (do not change the key name).

``` bash
[paths]
input= # put the input time series csv file path here
output= # put the output csv file path here. By default it should in Result folder
sparkccmlib= # this is a setting only used for pyspark+GPU version
[inputs]
x= # for the input csv file, put the column name as the lib_column in R ccm function argument lists (like anchovy)
y= # for the input csv file, put the column name as the target_column in R ccm function argument lists (like np_sst)
[parameters]
E= # the list of the embedding dimensions. Separate the values by ',' 
tau= # the list of lag steps to construct shadow manifold. Separate the values by ','
num_samples= # the number of samples, which is corresponding to the same argument in R ccm funciton (like 100)
LStart= # specify the beginning value of the lib_sizes sequence
LEnd= # specify the end value of the lib_sizes sequence
LInterval= # specify the interval size of the lib_sizes sequence
[options]
GenerateOutputCSV= # 0 for not generating output csv file (only show mean value in the process); 1 for generating
MultiLsVersion= # this is a setting only used for pyspark+GPU version
```
2. **TestInputCSVData** The diretory for the input time series.
3. **Result** The directory for the output csv (4 fields: E, tau, L, rho)
3. **PerformanceComparison** The GPU CUDA implementations in CCM, which you can test if CUDA installed properly and GPU power on the machine.
4. **CCM** The c++ library of parallel ccm. It is the core part of the parallel implementations using c++ language and GPU CUDA accelerations.
5. **SingleVersion,MPIVersion,SparkVersion** folders contains the compiled program, which can run on single machine, MPI cluster and Spark cluster separately. (not necessary to install GPU on these machines or clusters, you can choose to compile and run without GPU accelerations). These versions of programs have the common library -- **CCM**. These folders are used in *runsinglemachine.sh*,  *runmpic.sh*, *runsparkc.sh* scripts. 
6. **ScalaSpark** The Scala implementation of parallel ccm, which doesn't use the **CCM** library. No GPU acceleration and pure scala code. You can run it on single machine (Specify SPARK_MASTER = local[*]) and yarn cluster (Specify SPARK_MASTER = Yarn)

## Configurations of Different Versions 

### Scala Spark

This version is related to the folder **ScalaSpark**, which is implemented using IntelliJ IDE with sbt.

Use the following command in the **ScalaSpark** folder to assembly a fat-jar, which can be submited to the spark yarn cluster servers.
``` bash
sbt assembly
```

Need to upload config file and input csv file to HDFS
``` bash
hadoop fs -put ./TestInputCSVData/test_float_1000.csv
hadoop fs -put ./TestInputCSVData/test_float_1000.csv
```

Submit the jar using the following command (pass the config file path as the first argument)
```bash
spark-submit --master yarn scalaspark-assembly-1.0.jar ./ccm.cfg
```


### C++ multithreading + CUDA (GPU acceleration)  - single machine

There are three parts in convergent cross mapping algorithm can be accelerated by GPU (CUDA or Thrust). The other part will use openMP library to achieve multi-threading. Compile using the following commands in **PerformanceComparison** folder to test if your machine can support CUDA and openMP. If not, please install CUDA toolkit firstly.

```console
nvcc -Xcompiler -fopenmp -std=c++11 -lgomp -o ccm OpenMP_thrust.cu
```

####  1. Pairwise Euclidean distance kernel

####  2. Distance sorting  - thrust sort_by_key function

####  3. Pearson coefficient correlation kernel

## OpenMP  - Multi-threads
OpenMP supports a shared memory threading:  communicating through shared variables; All threads share an address space, but it can get complicated: Consistency models based on orderings of Reads (R), Writes (W) and Synchronizations (S)
use synchronization to protect data writing conflicts.  you have to wait writing then read
(Synchronization is used to impose order constraints and to protect access to shared data)

```cpp
#include<omp.h>


omp_lock_t lck;
omp_init_lock(&lck);
size_t num_cpus = omp_num_procs();
omp_set_num_threads(num_cpus);
#pragma omp parallel for
for(int sample = 0; sample < num_samples; sample++){
  int ID = omp_get_thread_num();
  // data shareing: local variables inside parallel scope are automatically private;  global variables outside parallel scope are automatically shared
  
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

### PySpark + C++ (CUDA)  -  Cluster


### MPI + C++ (CUDA) - Cluster

MPI is used only for inter-node parallelism, while OpenMP threads control intra-node parallelism

Mitigating Bottlenecks of cluster computing:  reducing the response time for large L jobs as we have to wait until the last job done in multi nodes.
How to:  build once and query multiple times for nearest neighbors finding

The central issue here is the overhead involved in internode communication:

Point-to-Point communication

MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)

MPI_Recv(void* data, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm communicator, MPI_Status* status)

The first argument is the data buffer. The second and third arguments describe the count and type of elements that reside in the buffer. The fourth and fifth arguments specify the rank of the sending/receiving process and the tag of the message. The sixth argument specifies the communicator and the last argument (for MPI_Recv only) provides information about the received message.

```cpp
// Find out rank, size
int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

int number;
if (world_rank == 0) {
    number = -1;
    MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
} else if (world_rank == 1) {
    MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("Process 1 received number %d from process 0\n",
           number);
}
```

Collective communication is that it implies a synchronization point among processes:

MPI_Barrier(MPI_Comm communicator) 

-- MPI has a special function that is dedicated to synchronizing processer, the function forms a barrier, and no processes in the communicator can pass the barrier until all of them call the function.


MPI_Bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)

**broadcast the time series data**

-- A broadcast is one of the standard collective communication techniques. During a broadcast, one process sends the same data to all processes in a communicator. One of the main uses of broadcasting is to send out user input to a parallel program, or send out configuration parameters to all processes.
the root process and receiver processes do different jobs, they all call the same MPI_Bcast function. When the root process (in our example, it was process zero) calls MPI_Bcast, the data variable will be sent to all other processes. When all of the receiver processes call MPI_Bcast, the data variable will be filled in with the data from the root process.

MPI_Scatter(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root,MPI_Comm communicator)

**scatter the parameter combinations**

MPI_Gather(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator)

**gater the rhos results for different parameter**

![images1](https://github.com/ALexanderpu/CCM-Parralization/blob/master/Screenshot%202018-11-27%2019.05.33.png)

#### MPI installation

implementation: MPICH2

#### Compile with cuda code
```console
# compile under c standard
mpicc -o hellow hellow.c

# compile using g++
g++ -I/home/bo/Desktop/mpich-3.3/src/include -L/home/bo/Desktop/mpich-3.3/lib test.cxx -lmpicxx -lmpi -o ccm

# compile with cuda and openmp c++
nvcc -Xcompiler -fopenmp -std=c++11 -lgomp -I/home/bo/Desktop/mpich-3.3/src/include -L/home/bo/Desktop/mpich-3.3/lib test.cu -lmpicxx -lmpi -o ccmwithcuda
```

#### Run using mpirun after compilation
option for mpirun: 
```console
mpirun -np 4 ./hellow
```
-pernode, --pernode
On each node, launch one process -- equivalent to -npernode 1. (deprecated in favor of --map-by ppr:1:node)

-H, -host, --host <host1,host2,...,hostN>
List of hosts on which to invoke processes.
-hostfile, --hostfile <hostfile>
Provide a hostfile to use.
