#!/bin/bash

echo "compiling the program for spark distributed version of ccm"

nvcc -Xcompiler -fopenmp -std=c++11 arch=compute_35 -c ./CCM/gpusort.cu
g++  -Xcompiler -fopenmp -std=c++11 -c ./CCM/ccmwithgpu.cpp
g++ -std=c++11 -o ./SparkVersion/sparkc gpusort.o ccmwithgpu.o

if [-e ./SparkVersion/sparkc]
then
    echo "compiling successfully, running spark distributed version of ccm"
    chmod u+x ./SparkVersion/sparkc
    spark-submit ./SparkVersion/SparkCCM.py
else
    echo "the script required for spark distributed version fails compiling"
fi

