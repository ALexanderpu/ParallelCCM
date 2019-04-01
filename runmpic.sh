#!/bin/bash

echo "make sure mpich2 has been installed on your system"


echo "compiling the program for MPI distributed version of ccm"

mpicxx -fopenmp -std=c++11 -o ./MPIVersion/MPIInterface ./CCM/MPIInterface.cpp ./CCM/include/cpu_sort.cpp


echo "running mpi distributed version of ccm"
mpirun -np 3 --hosts 10.80.64.110,10.80.64.35,10.80.64.53 ./MPIVersion/MPIInterface