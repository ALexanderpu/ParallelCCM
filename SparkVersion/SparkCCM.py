# running under python 2.7 
__author__      = "Bo Pu"

import sys
import ConfigParser
import pandas as pd
from pyspark.sql import SparkSession
import json
import os

# read parameter combinations config and fill into the objects
class Sample:
    def __init__(self, _observation, _target, _e, _tau, _l, _samples):
        self.observation = _observation
        self.target = _target
        self.e = _e
        self.tau = _tau
        self.samples = _samples
        self.l = _l

def ccm(LArr, EArr, TauArr, num_samples, time_series, scriptPath):
    observation, target = time_series['x'].tolist(), time_series['y'].tolist()
    paras = []
    for l in LArr:
        for e in EArr:
            for tau in TauArr:
                s = Sample(observation, target, e, tau, l, num_samples)
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("using default cfg file path")
        cfgfile = "/home/bo/cloud/CCM-Parralization/ccm.cfg"
    else:
        # read input cfg file: the first argument is the file path
        cfgfile = sys.argv[1]
        
    config = ConfigParser.RawConfigParser()
    config.read(cfgfile)
    try:
        input_path = config.get('paths', 'input')
        output_path = config.get('paths', 'output')
        script_path = config.get('paths', 'sparkccmlib')

        E = config.get('parameters', 'E')
        tau = config.get('parameters', 'tau')
        EArr = map(int, E.split(","))
        tauArr = map(int, tau.split(","))

        num_samples = config.getint('parameters', 'num_samples')
        
        LStart = config.getint('parameters', 'LStart')
        LEnd = config.getint('parameters', 'LEnd')
        LInterval = config.getint('parameters', 'LInterval')
        LArr = range(LStart, LEnd+1, LInterval)

        time_series = pd.read_csv(input_path)

        # start to run ccm algorithm
        result = ccm(LArr, EArr, tauArr, num_samples, time_series, script_path)

    except:
        print("parsing config file error")
    
    # save result into the local file: define the format



    
