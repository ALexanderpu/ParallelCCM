#!/bin/bash

echo "make sure g++ compiler ( nvcc cuda) has been installed in your system with sudo permission (test on ubuntu desktop system)"
echo "if CUDA (GPU) is not installed, please set options|GPUAcceleration = 0 in the config file "
echo "removing compiled files"
rm -rf ./SingleVersion/*

echo "start to compile executable file: ./SingleVersion/singlemachine"
if [ -d /usr/local/cuda/bin/ ]; then
    echo "detect GPU CUDA"
    echo "compiling gpu sort part"
    nvcc -Xcompiler -fopenmp -std=c++11 -arch=sm_35 -o ./SingleVersion/sort.o -c ./CCM/include/gpu_sort.cu
    echo "compiling cpu multi-thread part"
    g++ -fopenmp -lpthread -std=c++11 -o ./SingleVersion/main.o -c ./CCM/SingleMachine.cpp
    echo "link together"
    nvcc -Xcompiler -fopenmp -std=c++11 ./SingleVersion/main.o ./SingleVersion/sort.o -o ./SingleVersion/singlemachine 
else
    echo "NOT detect GPU CUDA"
    echo "compiling cpu multi-thread part"
    g++ -fopenmp -lpthread -std=c++11 -o ./SingleVersion/sort.o -c ./CCM/include/cpu_sort.cpp
    g++ -fopenmp -lpthread -std=c++11 -o ./SingleVersion/main.o -c ./CCM/SingleMachine.cpp
    echo "link together"
    g++ -fopenmp -lpthread -std=c++11 -o ./SingleVersion/singlemachine ./SingleVersion/main.o ./SingleVersion/sort.o
fi

# g++ -std=c++11 -o ./SingleVersion/singlemachine ./SingleVersion/main.o ./SingleVersion/sort.o -fopenmp -L/usr/local/cuda-10.0/lib64 -lpthread
if [ -e ./SingleVersion/singlemachine ]
then
    echo "compiling successfully, running single machine parallel version of ccm with gpu acceleration enabled"
    chmod u+x ./SingleVersion/singlemachine
    # pass the config file here
    ./SingleVersion/singlemachine ./ccm.cfg # configure file fullpath needs to be replaced
else
    echo "the compiling process failed!"
fi
