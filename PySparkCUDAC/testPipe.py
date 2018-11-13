from pyspark.sql import SparkSession
import json
from subprocess import call
import random
import os
class Sample:
    def __init__(self, _observation, _target, _e, _tau, _l, _samples):
        self.observation = _observation
        self.target = _target
        self.e = _e
        self.tau = _tau
        self.samples = _samples
        self.l = _l

def testInterfaceC():
    o = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = [9.0, 10,0, 11.1, 12.23, 3.9, 1.8]
    s = Sample(o, t, 2, 1, 100, 200)
    para = json.dumps(vars(s))
    print para
    call(["./testPipe", para])


def generate_time_series(length, dt=0.01, noiseCoeff=0.01):
    """
    length is the array size
    dt is dynamic system min time  0.01
    noiseCoeff:  0.01
    """
    state = [2.0, 4.0, 4.0, 3.0]
    # change parameters: birth rate for prey death rate for predator
    hare1NatalityRate = 0.02
    hare2NatalityRate = 0.025
    lynx1MortalityRate = 0.03
    lynx2MortalityRate = 0.03
    perLynx1AnnualHare1ProbOfGettingCaught = 0.005
    perLynx2AnnualHare2ProbOfGettingCaught = 0.010
    perLynx2AnnualHare1ProbOfGettingCaught = 0.0025
    perHare1NetLynx1BirthRateIncrement = 0.0005
    perHare2NetLynx2BirthRateIncrement = 0.001
    hares1 = [state[0]]
    lynx1 = [state[1]]
    hares2 = [state[2]]
    lynx2 = [state[3]]
    for t in range(1, length):
        new_hares1 = max(0, hares1[t-1] + dt * (hare1NatalityRate * hares1[t-1] - (perLynx1AnnualHare1ProbOfGettingCaught * lynx1[t-1] + perLynx2AnnualHare1ProbOfGettingCaught * lynx2[t-1]) * hares1[t-1] + noiseCoeff * random.random()))
        new_lynx1 = max(0, lynx1[t-1] + dt * (-lynx1MortalityRate * lynx1[t-1] + (perHare1NetLynx1BirthRateIncrement * hares1[t-1]) * lynx1[t-1] + noiseCoeff * random.random()))
        new_hares2 = max(0, hares2[t-1] + dt * (hare2NatalityRate * hares2[t-1] - (perLynx2AnnualHare2ProbOfGettingCaught * lynx2[t-1]) * hares2[t-1] + noiseCoeff * random.random()))
        new_lynx2 = max(0, lynx2[t-1] + dt * (-lynx2MortalityRate * lynx2[t-1] + (perHare2NetLynx2BirthRateIncrement * hares2[t-1]) * lynx2[t-1] + noiseCoeff * random.random()))
        hares1.append(new_hares1)
        lynx1.append(new_lynx1)
        hares2.append(new_hares2)
        lynx2.append(new_lynx2)
    return lynx2, hares2


def ccm(LArr, EArr, TauArr, num_samples):
    observation, target = generate_time_series(100)
    paras = []
    for l in LArr:
        for e in EArr:
            for tau in TauArr:
                s = Sample(observation, target, e, tau, l, num_samples)
                para = json.dumps(vars(s))
                #print para
                paras.append(para)

    spark = SparkSession.builder.appName("CCM").getOrCreate()

    paraRdd = spark.sparkContext.parallelize(paras)
    program = "/Users/alexpb/Desktop/Lab/PySparkCUDAC/testPipe"
    spark.sparkContext.addFile(program)
    piped = paraRdd.pipe(program)
    piped.collect()
    
    for x in piped.collect():
        print x
    
    spark.stop()

ccm([1,2,3], [4,5,6], [7,8,9], 250)

    
