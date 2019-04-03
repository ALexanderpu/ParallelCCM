import random
import numpy as np
import pandas as pd

# float point number   causal/non causal 
def DynamicModelInputGenerator(seqenceLength):
    initState = [2.0, 4.0, 4.0, 3.0]
    meanHare1NatalityRate = 0.02
    meanHare2NatalityRate = 0.025
    meanLynx1MortalityRate = 0.03
    meanLynx2MortalityRate = 0.03

    meanPerLynx1AnnualHare1ProbOfGettingCaught = 0.005
    meanPerLynx2AnnualHare2ProbOfGettingCaught = 0.01
    meanPerLynx2AnnualHare1ProbOfGettingCaught = 0.0025

    meanPerHare1NetLynx1BirthRateIncrement = 0.0005
    meanPerHare2NetLynx2BirthRateIncrement = 0.001

    hare1NatalityRate = meanHare1NatalityRate * (1.0 + 0.01 * random.uniform(0, 1)) #0.02
    hare2NatalityRate = meanHare2NatalityRate * (1.0 + 0.01 * random.uniform(0, 1)) #0.025
    lynx1MortalityRate = meanLynx1MortalityRate * (1.0 + 0.01 * random.uniform(0, 1)) #0.03
    lynx2MortalityRate = meanLynx2MortalityRate * (1.0 + 0.01 * random.uniform(0, 1)) #0.03

    perLynx1AnnualHare1ProbOfGettingCaught = meanPerLynx1AnnualHare1ProbOfGettingCaught * (1.0 + 0.01 * random.uniform(0, 1)) #0.005
    perLynx2AnnualHare2ProbOfGettingCaught = meanPerLynx2AnnualHare2ProbOfGettingCaught * (1.0 + 0.01 * random.uniform(0, 1)) #0.010
    perLynx2AnnualHare1ProbOfGettingCaught = meanPerLynx2AnnualHare1ProbOfGettingCaught * (1.0 + 0.01 * random.uniform(0, 1)) #0.0025
    perHare1NetLynx1BirthRateIncrement = meanPerHare1NetLynx1BirthRateIncrement * (1.0 + 0.01 * random.uniform(0, 1)) #0.0005
    perHare2NetLynx2BirthRateIncrement = meanPerHare2NetLynx2BirthRateIncrement * (1.0 + 0.01 * random.uniform(0, 1)) #0.001

    noiseCoeffHare1 = 0.01
    noiseCoeffLynx1 = 0.01
    noiseCoeffHare2 = 0.01
    noiseCoeffLynx2 = 0.01

    dt = 1e-1
    (hares1, lynx1, hares2, lynx2) = initState
    arrW = [hares1]
    arrX = [lynx1]
    arrY = [hares2]
    arrZ = [lynx2]

    for _ in range(1, seqenceLength):
        arrW.append(max(0, hares1 + dt * (hare1NatalityRate * hares1 - (perLynx1AnnualHare1ProbOfGettingCaught * lynx1 + perLynx2AnnualHare1ProbOfGettingCaught * lynx2) * hares1 + noiseCoeffHare1 * random.uniform(0, 1))))
        arrX.append(max(0, lynx1 + dt * (-lynx1MortalityRate * lynx1 + (perHare1NetLynx1BirthRateIncrement * hares1) * lynx1 + noiseCoeffLynx1 * random.uniform(0, 1))))
        arrY.append(max(0, hares2 + dt * (hare2NatalityRate * hares2 - (perLynx2AnnualHare2ProbOfGettingCaught * lynx2) * hares2 + noiseCoeffHare2 * random.uniform(0, 1))))
        arrZ.append(max(0, lynx2 + dt * (-lynx2MortalityRate * lynx2 + (perHare2NetLynx2BirthRateIncrement * hares2) * lynx2 + noiseCoeffLynx2 * random.uniform(0, 1))))
        # update current variables
        hares1 = arrW[-1]
        lynx1 = arrX[-1]
        hares2 = arrY[-1]
        lynx2 = arrZ[-1]

    return (arrZ, arrY)


# integer number   causal / non causal

def CausalFloatCSV(filename, size):
    (x, y) = DynamicModelInputGenerator(size)
    arr = np.array([x, y])
    df = pd.DataFrame(data = arr.transpose(), columns=['x', 'y'])
    df.to_csv(filename,  header=True)

if __name__ == "__main__":
    CausalFloatCSV("./TestCSVData/test_float_1000.csv", 1000)