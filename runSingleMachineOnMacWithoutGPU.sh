
# brew install libomp    if -fopenmp is not supported

rm -rf ./SingleVersion/*
echo "start to compile executable file: ./SingleVersion/singlemachine"

echo "compiling cpu multi-thread part on mac"
/usr/local/bin/g++-9 -fopenmp -lpthread -std=c++11 -o ./SingleVersion/sort.o -c ./CCM/include/cpu_sort.cpp
/usr/local/bin/g++-9 -fopenmp -lpthread -std=c++11 -o ./SingleVersion/singlemachine ./SingleVersion/main.o ./SingleVersion/sort.o
/usr/local/bin/g++-9 -fopenmp -lpthread -std=c++11 -o ./SingleVersion/main.o -c ./CCM/SingleMachine.cpp

if [ -e ./SingleVersion/singlemachine ]
then
    echo "compiling successfully, running single machine parallel version of ccm with gpu acceleration enabled"
    chmod u+x ./SingleVersion/singlemachine
    # pass the config file here
    ./SingleVersion/singlemachine ./ccm.cfg # configure file fullpath needs to be replaced
else
    echo "the compiling process failed!"
fi
