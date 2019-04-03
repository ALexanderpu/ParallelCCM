# running under python 2.7 
__author__      = "Bo Pu"

import sys
import ConfigParser
import pandas as pd
from pyspark.sql import SparkSession
import json
import numpy as np
import os

# for single L
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


# for multi Ls
class SampleMultiL:
    def __init__(self, _observations, _targets, _e, _tau, _samples, _lstart, _lend, _linterval, _multil, _genoutput):
        self.observations = _observations
        self.targets = _targets
        self.e = _e
        self.tau = _tau
        self.samples = _samples
        self.multil = _multil
        self.genoutput = _genoutput
        self.lstart = _lstart
        self.lend = _lend
        self.linterval = _linterval


def ccmMultiL(LStart, LEnd, LInterval, EArr, TauArr, num_samples, time_series, x, y, scriptPath, generateOutput):
    observations, targets = time_series[x].tolist(), time_series[y].tolist()
    paras = []
    for e in EArr:
        for tau in TauArr:
            s = SampleMultiL(observations, targets, e, tau, num_samples, LStart, LEnd, LInterval, 1, generateOutput)
            para = json.dumps(vars(s))
            #print para
            paras.append(para)
    # start the spark context 
    spark = SparkSession.builder.appName("PySparkCCMMultiL").getOrCreate()
    paraRdd = spark.sparkContext.parallelize(paras)
    piped = paraRdd.pipe(scriptPath)
    result = piped.collect()
    for ele in result:
        print(ele)
    
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
        MultiLsVersion = config.getboolean('options', 'MultiLsVersion')
        GenerateOutputCSV = config.getint('options', 'GenerateOutputCSV')

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

        x = config.get('inputs', 'x')
        y = config.get('inputs', 'y')


        time_series = pd.read_csv(input_path)

        # start to run ccm algorithm
        if MultiLsVersion:
            result = ccmMultiL(LStart, LEnd, LInterval, EArr, tauArr, num_samples, time_series, x, y, script_path, GenerateOutputCSV)
            # output path in the result
            with open("outputcsvpath.out", "w") as f:
                for record in result:
                    f.write(record)
        else:
            result = ccm(LArr, EArr, tauArr, num_samples, time_series, x, y, script_path, GenerateOutputCSV)
            # save csv file to output folder
            # save result into the local file: define the format
            labels = ['E', 'tau', 'L', 'rho']
            taus = []
            es = []
            ls = []
            rhos = []
            for r in result:
                record = json.loads(r)
                rho = record['result']  # a list
                num_samples = len(rho)
                tau = record['tau']
                e = record['e']
                l = record['l']
                rhos.extend(rho)
                taus.extend([tau] * num_samples)
                es.extend([e] * num_samples)
                ls.extend([l] * num_samples)
            result_df = pd.DataFrame(np.column_stack([es, taus, ls, rhos]), columns=labels)
            output_file = output_path + "/sparkccm.csv"
            result_df.to_csv(output_file)
    except:
        print("parsing config file error")