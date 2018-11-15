import subprocess
import os
cwd = os.getcwd()
Rfile = cwd + "/ResultVerification/Comparison.R"
CSVInput = "test_float_1000.csv"
ParallelInput = "output_withcuda.csv"
# ccm parameters
samples = "250"
tau = "3"
E = "3"
L = "100"

subprocess.call(["/Users/bopu/anaconda3/bin/Rscript", Rfile, CSVInput, ParallelInput, samples, tau, E, L])