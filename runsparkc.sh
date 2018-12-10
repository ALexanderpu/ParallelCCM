#!/bin/bash

echo "compiling the program for spark distributed version of ccm"
echo "compiling gpu sort part"
nvcc -Xcompiler -fopenmp -std=c++11 -arch=sm_35 -o ./SparkVersion/sort.o -c ./CCM/include/gpu_sort.cu
echo "compiling cpu multi-thread part"
g++ -fopenmp -lpthread -std=c++11 -o ./SparkVersion/main.o -c ./CCM/SparkInterface.cpp
echo "merge together"
nvcc -Xcompiler -fopenmp -std=c++11 ./SparkVersion/main.o ./SparkVersion/sort.o -o ./SparkVersion/sparkc

if [ -e ./SparkVersion/sparkc ]
then
    echo "compiling successfully, running spark distributed version of ccm"
    chmod u+x ./SparkVersion/sparkc
    spark-submit ./SparkVersion/SparkCCM.py
else
    echo "the script required for spark distributed version fails compiling"
fi

