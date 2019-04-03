#!/bin/bash
echo "ensure spark yarn cluster & hdfs has been installed on your system"
echo "removing compiled files"
rm -rf ./SparkVersion/*

echo "start to compile executable file for pyspark to call: ./SparkVersion/sparkc"
if [ -d /usr/local/cuda/bin/ ]; then
    echo "detect GPU CUDA"
    echo "compiling gpu sort part"
    nvcc -Xcompiler -fopenmp -std=c++11 -arch=sm_35 -o ./SparkVersion/sort.o -c ./CCM/include/gpu_sort.cu
    echo "compiling cpu multi-thread part"
    g++ -fopenmp -lpthread -std=c++11 -o ./SparkVersion/main.o -c ./CCM/SparkInterface.cpp
    echo "link together"
    nvcc -Xcompiler -fopenmp -std=c++11 ./SparkVersion/main.o ./SparkVersion/sort.o -o ./SparkVersion/sparkc 
else
    echo "NOT detect GPU CUDA"
    echo "compiling cpu multi-thread part"
    g++ -fopenmp -lpthread -std=c++11 -o ./SparkVersion/sort.o -c ./CCM/include/cpu_sort.cpp
    g++ -fopenmp -lpthread -std=c++11 -o ./SparkVersion/main.o -c ./CCM/SparkInterface.cpp
    echo "link together"
    g++ -fopenmp -lpthread -std=c++11 -o ./SparkVersion/sparkc ./SparkVersion/main.o ./SparkVersion/sort.o
fi

if [ -e ./SparkVersion/sparkc ]
then
    echo "compiling successfully, running spark distributed version of ccm using spark-submit"
    chmod u+x ./SparkVersion/sparkc
    spark-submit ./SparkCCM.py /home/bo/cloud/CCM-Parralization/ccm.cfg
else
    echo "the script required for spark distributed version fails compiling"
fi

