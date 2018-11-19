import subprocess
import os
cwd = os.getcwd()
Rfile = cwd + "/ResultVerification/Comparison.R"
CSVInput = "test_float_1000.csv"
ParallelInput = "output_withcuda.csv"
# ccm parameters
samples = "1000"
tau = "3"
E = "3"
L = "1000"

subprocess.call(["/Users/bopu/anaconda3/bin/Rscript", Rfile, CSVInput, ParallelInput, samples, tau, E, L])
