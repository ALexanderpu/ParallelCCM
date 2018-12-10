#!/bin/bash

echo "compiling gpu sort part"
nvcc -Xcompiler -fopenmp -std=c++11 -arch=sm_35 -o ./SingleVersion/sort.o -c ./CCM/include/gpu_sort.cu
echo "compiling cpu multi-thread part"
g++ -fopenmp -lpthread -std=c++11 -o ./SingleVersion/main.o -c ./CCM/SingleMachine.cpp

echo "merge together"
nvcc -Xcompiler -fopenmp -std=c++11 ./SingleVersion/main.o ./SingleVersion/sort.o -o ./SingleVersion/singlemachine 
# g++ -std=c++11 -o ./SingleVersion/singlemachine ./SingleVersion/main.o ./SingleVersion/sort.o -fopenmp -L/usr/local/cuda-10.0/lib64 -lpthread
if [ -e ./SingleVersion/singlemachine ]
then
    echo "compiling successfully, running single machine parallel version of ccm"
    chmod u+x ./SingleVersion/singlemachine
    ./SingleVersion/singlemachine
else
    echo "the compiling process did not success!"
fi