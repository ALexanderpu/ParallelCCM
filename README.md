# CCM Parallization Techniques

## CUDAC++ - GPU accelerate

Because GPU programming is an art, and it can be very, very challenging to get it right. On the other hand, because GPUs are well-suited only for certain kinds of computations (matrix or algebra problems).

Task parallel: The first one refers, roughly speaking, to problems where several threads are working on their own tasks (not exact the same but more or less independently). 

Data parallel: The second one refers to problems where many threads are all doing the same - but on different parts of the data (GPU are good at: They have many cores, and all the cores do the same, but operate on different parts of the input data)

GPUs are ridiculously fast in terms of theoretical computational power (FLOPS, Floating Point Operations Per Second). But they are often throttled down by the memory bandwidth  (Namely whether problems are memory bound or compute bound)

Memory bound: Vector Additions have to read two data elements, then perform a single addition, and then write the sum into the result vector. You will not see a speedup when doing this on the GPU, because the single addition does not compensate for the efforts of reading/writing the memory

Compute bound:refers to problems where the number of instructions is high compared to the number of memory reads/writes. For example, consider a matrix multiplication: The number of instructions will be O(n^3) when n is the size of the matrix. In this case, one can expect that the GPU will outperform a CPU at a **certain matrix size**

On the GPU, you may encounter challenges on a much lower level: Occupancy, register pressure, shared memory pressure, memory coalescing


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

## Spark/MPI   -   Multi-nodes

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

-- A broadcast is one of the standard collective communication techniques. During a broadcast, one process sends the same data to all processes in a communicator. One of the main uses of broadcasting is to send out user input to a parallel program, or send out configuration parameters to all processes.
the root process and receiver processes do different jobs, they all call the same MPI_Bcast function. When the root process (in our example, it was process zero) calls MPI_Bcast, the data variable will be sent to all other processes. When all of the receiver processes call MPI_Bcast, the data variable will be filled in with the data from the root process.
[images1]: https://www.dropbox.com/s/saminbnq6k6uxxx/Screenshot%202018-11-27%2019.05.33.png?dl=0

![images1]
### MPI installation

implementation: MPICH2

### Compile with cuda code
```console
mpicc -o hellow hellow.c
mpirun -np 4 ./hellow
```


```console
nvcc -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi spaghetti.cu -o program
```
