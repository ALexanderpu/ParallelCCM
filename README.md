# Applying Parallel Techniques on Convergent Cross Mapping (CCM)
---

[![Build Status](https://travis-ci.com/ALexanderpu/CCM-Parralization.svg?token=38AAxXGq77ksaSrhHKSS&branch=newest)](https://travis-ci.com/ALexanderpu/CCM-Parralization)

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

1. **ccm.cfg** The configuration file defines the input and output file paths and parameters settings for the parallel ccm implementation. This cfg file contains 4 sections (paths, inputs, parameters and options), and in each section there are several key-value pairs (do not change the key name). (ccm-scala.cfg is a copy of ccm.cfg: basically the same but scala use hdfs directory, which is slight different from other versions - NFS shared directory)

``` bash
[paths]
input= # put the input time series csv file fullpath here
output= # put the output csv file fullpath here. By default it should in Result folder
sparkccmlib= # this is a setting only used for pyspark+GPU version, the extern application fullpath (default it is in the SparkVersion folder)
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
GPUAcceleration= # cuda has already installed can be set as 1, not install cuda must be set as 0
GenerateOutputCSV= # 0 for not generating output csv file (only show mean value in the process); 1 for generating
```
2. **TestInputCSVData** The diretory for the input time series.
3. **Result** The directory for the output csv. (4 fields: E, tau, L, rho)
3. **PerformanceComparison** The GPU CUDA implementations in CCM, which you can test if CUDA installed properly and GPU power on the machine.
4. **CCM** The c++ library of parallel ccm. It is the core part of the parallel implementations using c++ language and GPU CUDA accelerations.
5. **SingleVersion,MPIVersion,SparkVersion** folders contains the compiled C++ program, which can run on single machine, MPI cluster and Spark cluster separately. (not necessary to install GPU on these machines or clusters, you can choose to compile and run without GPU acceleration). These versions of programs have the common library -- **CCM**. These folders are used in *runsinglemachine.sh*,  *runmpic.sh*, *runsparkc.sh* scripts. 
6. **ScalaSpark** The Scala implementation of parallel ccm, which doesn't use the **CCM** library. No GPU acceleration and pure scala code. You can run it on single machine (Specify SPARK_MASTER = local[*]) and yarn cluster (Specify SPARK_MASTER = Yarn)

## Configurations of Different Parallel CCM Versions 


### C++ Single Machine (with/without GPU acceleration)

There are several prerequisites before runing ccm on one single machine. (Suggest Linux system like Ubuntu 16.04). 

1. Check compiler dependencies: G++ and NVCC.
```bash
g++ --version
nvcc --version
```
2. Install G++, NVCC compiler dependencies: NVIDIA CUDA toolkit can be found [here](http://developer.nvidia.com/cuda-downloads). This step can be skip if there is no GPU on your machine. You can directly jump to step 3 (the script will check if you install CUDA). By the way, if you skip this step, please set [options|GPUAcceleration] = 0 in ccm.cfg file.

```bash
# install g++ using one of the two following command
sudo apt install g++
sudo apt install build-essential

# install cuda
# you need: CUDA-capable GPU (hardware), linux with gcc compiler (system) and NIVIDIA CUDA Toolkit (software)
# download Toolkit: cuda_10.0.130_410.48_linux.run to Download directory

# remove everything nvidia component installed  and close desktop (use ssh login to install)
sudo apt-get purge nvidia* 
sudo service lightdm stop
sudo init 3
sudo reboot
chmod u+x ~/Downloads/cuda_10.0.130_410.48_linux.run
# if they need reboot as the reason  An attmept has been made to disable Nouveau
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
## copy following lines
blacklist nouveau
options nouveau modeset=0
##
sudo update-initramfs -u
sudo reboot
# then login in and rerun
sudo bash ~/Downloads/cuda_10.0.130_410.48_linux.run -silent

vim ~/.profile  # add this line below to permanently add into path
export PATH=$PATH:/usr/local/cuda-10.0/bin
export PATH=$PATH:/usr/local/cuda-10.0/lib64  # cublas libraryx`
nvcc --version # nvcc: NVIDIA (R) Cuda compiler driver
nvidia-smi     # check CUDA GPU status
# show desktop again
sudo service lightdm start
```

3. Compile and run parallel CCM in single machine:
```bash
# download the project and cd into the project folder. 
# replace the ccm.cfg full filepath in runsinglemachine.sh and run
bash ./runsinglemachine.sh
```

### MPI Cluster
This version requires 2 things: MPI cluster with a shared file directory.

MPI is simply a standard which others follow in their implementation. Because of this, there are a wide variety of MPI implementations out there. One of the most popular implementations, MPICH2, will be used for all of the examples provided through this version. There are several prerequisites before runing ccm on MPI cluster. (Suggest Linux systems like Ubuntu 16.04). 
1. Install G++ compiler (refer to previous section)

2. Install MPI on single machine: 
* Install compiler dependencies on single machine: MPICH2 (the latest version of MPICH2 is available [here](https://www.mpich.org/)). The version that I will be using is 3.3. Once doing this, you should be able to configure your installation by performing ***./configure --disable-fortran***. When configuration is done, it is time to build and install MPICH2 with ***make; sudo make install***.
```bash
tar -xzf mpich-3.3.tar.gz
cd mpich-3.3
./configure --disable-fortran
make; sudo make install
mpiexec --version
```
* Check compiler dependencies on single machine: If your build was successful, you should be able to type ***mpiexec --version*** and see something similar to this.
```bash
HYDRA build details:
    Version:                         3.3
    Release Date:                    Wed Nov 21 15:02:56 CST 2018
```

3. Install MPI as Cluster within a LAN: 

* Repeating step 1 and 2 to install MPICH2 on every node in the cluster you prepared. Then configure your ***hosts*** file to set up login the nodes (IP addresses have to be replaced) without password (so we can login use ssh host_i). This file is used by your device operating system to map hostnames to IP addresses. 

```bash
# set up ssh login without password and edit the hosts file:  we can login use ssh host_i (replace use any: host0, host1, host2, host3)
sudo vim /etc/hosts
---  add contents in this file:
10.80.64.41     host0
10.80.64.35     host1
10.80.64.53     host2
10.80.64.110    host3
---
sudo apt-get install openssh-server
ssh-keygen
ssh-add
ssh-copy-id youraccountusername@hosti
# verify if you can login without password
ssh host_i 
```


* Setting up NFS server for shared directory: ~/cloud. It involves two steps: Set NFS server on master machine; Set NFS client on other machines in the cluster. You share a directory via NFS in master which the client mounts to exchange data.

```bash
# 1. setting NFS server on master machine: host0
sudo apt-get install -y nfs-kernel-server
cd ~
mkdir cloud  # under ~

sudo vim /etc/exports
---  add an entry in this file: 
/home/bo/cloud *(rw,sync,no_root_squash,no_subtree_check)
---
sudo exportfs -a
sudo service nfs-kernel-server restart


# 2. setting NFS client on worker machines: host1, host2, host3
sudo apt-get install nfs-common
cd ~
mkdir cloud  # under ~
sudo mount -t nfs host1:~/cloud ~/cloud

# 3. check in all the clients in the cluster
df -h
# if success will show: host0:~/cloud  49G   15G   32G  32% /home/youraccountusername/cloud

# 4. modify the permission in the master (if you want to generate csv file)
chmod -R 777 ~/cloud
```

(**Note**: Please put CCM-Parallelization project folder under the shared file directory)

4.  Compile and run parallel CCM in MPI cluster:
```bash
# download the project and cd into the project folder. 
# replace the ccm.cfg full filepath in runmpic.sh and run
bash ./runmpic.sh
```

### PySpark Cluster (with/without GPU acceleration)

This version requires 2 things: Spark Yarn cluster with a shared file direcory. 

1. Install G++, NVCC compiler (refer to single machine section)

2. Installation of Python libraries:

```bash
sudo apt upgrade
# installing python 2.7 and pip for it
sudo apt install python2.7 python-pip

# install python libraries
pip install configparser
pip install numpy
pip install json
pip install pyspark
pip install pandas
```
3. Installation of Spark Yarn Cluster (please refer to  [Ambari](https://github.com/hortonworks/ansible-hortonworks)). Another option is Google Cloud Platform (GCP).

- You can submit python file or fat-jar to Yarn using ***spark-submit*** if Yarn and Spark have been successfully installed.
```bash
spark-submit ./SparkCCM.py
```

- The implementation mainly utilizes ***pipedRDD*** to assign tasks and invoke external application, which you have to specify the application path (using NFS shared directory) in ccm.cfg: [path|sparkccmlib] 


4. Compile and run parallel CCM in Spark Yarn cluster:
```bash
# download the project and cd into the project folder. 
# replace the ccm.cfg full filepath in runmpic.sh and run
bash ./runsparkc.sh
```





### Scala Spark

This version is related to the folder **ScalaSpark**, which is implemented using IntelliJ IDE with sbt.

#### Local Mode

Local Mode is relatively easy. download IntelliJ IDE and import ScalaSpark as a spark project. Modify two places before running
change two things in the ScalaSpark folder: 
1. main.scala: set SPARK_MASTER -> "local[*]"; 
2. build.sbt: remove "provided" in spark-related packages;

Also, it is necessary to pass ccm-scala.cfg file fullpath as the first argument in the program.




#### Yarn Cluster Mode

Installation of Spark Yarn Cluster (please refer to  [Ambari](https://github.com/hortonworks/ansible-hortonworks)). Another option is Google Cloud Platform (GCP).

After setting Spark Yarn Cluster, you can submit the scala project fat-jar using ***spark-submit***.

Use the following command in the **ScalaSpark** folder to assembly a fat-jar, which can be submited to the spark yarn cluster servers.
Before ***sbt assembly***, 2 things should be done:
1. main.scala: set SPARK_MASTER -> "yarn"; 
2. build.sbt: have the 'provided' lines so that the assembly won't include the spark source jars.

``` bash
cd ./ScalaSpark
# before assembling fat-jar: change two things in the ScalaSpark folder: 1. source code: SPARK_MASTER -> "yarn"; 2. build.sbt spark package to "provided"
# if assembly successfully, the fat-jar will be stored at ./target/scala-2.11/scala-spark-ccm-assembly-0.1.jar
sbt assembly

# put/upload local file into hdfs; so you have to change [paths|inputs] in ccm-scala.cfg file to the hdfs path like: /user/bo/test_float_1000.csv
hadoop fs -put ~/cloud/CCM-Parralization/TestInputCSVData/test_float_1000.csv

# pass ccm-scala.cfg file fullpath as the first argument
spark-submit --master yarn ./target/scala-2.11/scala-spark-ccm-assembly-0.1.jar ~/cloud/CCM-Parralization/ccm-scala.cfg
```
