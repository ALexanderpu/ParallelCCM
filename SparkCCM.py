# running under python 2.7 
__author__      = "Bo Pu"

import sys
import ConfigParser
import pandas as pd
from pyspark.sql import SparkSession
import json
import numpy as np
import os

# for single L; which will be not used 
# read parameter combinations config and fill into the objects
class Sample:
    def __init__(self, _observations, _targets, _e, _tau, _l, _samples, _multil, _genoutput):
        self.observations = _observations
        self.targets = _targets
        self.e = _e
        self.tau = _tau
        self.samples = _samples
        self.l = _l
        self.multil = _multil
        self.genoutput = _genoutput

def ccm(LArr, EArr, TauArr, num_samples, time_series, x, y, scriptPath, generateOutput):
    observations, targets = time_series[x].tolist(), time_series[y].tolist()
    paras = []
    for l in LArr:
        for e in EArr:
            for tau in TauArr:
                s = Sample(observations, targets, e, tau, l, num_samples, 0, generateOutput)
                para = json.dumps(vars(s))
                #print para
                paras.append(para)
    # start the spark context 
    spark = SparkSession.builder.appName("PySparkCCM").getOrCreate()
    paraRdd = spark.sparkContext.parallelize(paras)
    piped = paraRdd.pipe(scriptPath)
    result = piped.collect()
    spark.stop()
    return result


# for multi Ls in one task
class SampleMultiL:
    def __init__(self, _observations, _targets, _e, _tau, _samples, _lstart, _lend, _linterval, _multil, _genoutput, _outputpath, _gpu):
        self.observations = _observations
        self.targets = _targets
        self.e = _e
        self.tau = _tau
        self.samples = _samples
        self.lstart = _lstart
        self.lend = _lend
        self.linterval = _linterval
        self.multil = _multil
        self.genoutput = _genoutput
        self.outputpath = _outputpath
        self.gpu = _gpu


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please input the local path of ccm.cfg")
        sys.exit()

    # read input cfg file: the first argument is the file path
    cfgfile = sys.argv[1]
        
    config = ConfigParser.RawConfigParser()
    config.read(cfgfile)
    try:
        input_path = config.get('paths', 'input')
        output_path = config.get('paths', 'output')
        script_path = config.get('paths', 'sparkccmlib')

        E = config.get('parameters', 'E')
        Tau = config.get('parameters', 'tau')
        EArr = map(int, E.split(","))
        TauArr = map(int, Tau.split(","))

        num_samples = config.getint('parameters', 'num_samples')
        
        LStart = config.getint('parameters', 'LStart')
        LEnd = config.getint('parameters', 'LEnd')
        LInterval = config.getint('parameters', 'LInterval')

        xname = config.get('inputs', 'x')
        yname = config.get('inputs', 'y')

        time_series = pd.read_csv(input_path)
        observations, targets = time_series[xname].tolist(), time_series[yname].tolist()

        GenerateOutputCSV = config.getint('options', 'GenerateOutputCSV')
        GPUAcceleration = config.getint('options', 'GPUAcceleration')
        print("GPUAcceleration: " + str(GPUAcceleration))
        # generate para rdd to separate the tasks to different workers
        paras = []
        for e in EArr:
            for tau in TauArr:
                s = SampleMultiL(observations, targets, e, tau, num_samples, LStart, LEnd, LInterval, 1, GenerateOutputCSV, output_path, GPUAcceleration)
                para = json.dumps(vars(s))
                #print para
                paras.append(para)
        # start the spark context 
        
        print("size: " + str(len(paras)))
        
        
        spark = SparkSession.builder.appName("PySparkCCMMultiL").getOrCreate()
        paraRdd = spark.sparkContext.parallelize(paras)
        piped = paraRdd.pipe(script_path)
        result = piped.collect()

        for ele in result:
            print(ele)

        spark.stop()

        # output path in the result
        # with open("outputcsvpath.out", "w") as f:
        #    for record in result:
        #        f.write(record)
    except:
        print("parsing config file error")