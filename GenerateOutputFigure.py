import subprocess
import os
import ConfigParser
import sys
cwd = os.getcwd()


# ccm parameters
if __name__ == "__main__":
    Rfile = cwd + "/Comparison.R"
    if len(sys.argv) < 2:
        print("using default cfg file path")
        cfgfile = "/home/bo/cloud/CCM-Parralization/ccm.cfg"
    else:
        # read input cfg file: the first argument is the file path
        cfgfile = sys.argv[1]
        
    config = ConfigParser.RawConfigParser()
    config.read(cfgfile)

    input_path = config.get('paths', 'input')
    output_path = config.get('paths', 'output')

    E = config.get('parameters', 'E')
    tau = config.get('parameters', 'tau')
    EArr = map(int, E.split(","))
    tauArr = map(int, tau.split(","))

    num_samples = config.getint('parameters', 'num_samples')
    
    LStart = config.getint('parameters', 'LStart')
    LEnd = config.getint('parameters', 'LEnd')
    LInterval = config.getint('parameters', 'LInterval')
    LArr = range(LStart, LEnd+1, LInterval)

    ParallelInput = output_path + "/singlemachine.csv"
    subprocess.call(["Rscript", Rfile, input_path, ParallelInput, str(num_samples), str(tauArr[0]), str(EArr[0]), str(LStart), str(LInterval), str(LEnd)])

    # print("parsing config file error")