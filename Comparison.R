library("ggplot2")
library("hexbin")
library("RColorBrewer")
library("MASS")
library("rEDM")
library('reshape2')
library(tictoc)

createGGPlotInternal <- function(ccmXYIncludingNonnegColumn, isDisplayMAE, isTruncatePerRealizationRho)
{
  if (isDisplayMAE)
    gPlot <- ggplot(ccmXYIncludingNonnegColumn, aes(x=lib_size, y=mae))
  else
  {
    if (isTruncatePerRealizationRho)
      gPlot <- ggplot(ccmXYIncludingNonnegColumn, aes(x=lib_size, y=nonnegRho))
    else
      gPlot <- ggplot(ccmXYIncludingNonnegColumn, aes(x=lib_size, y=rho)) 
  }
  return(gPlot)  
}

createPerRealizationDensityPlot <- function(ccmXY, strFileName, strXDescription=ccmXY$lib_column[1], strYDescription=ccmXY$target_column[1], isTruncatePerRealizationRho=FALSE, isDisplayMAE=FALSE, num_samples=250)
{
  E = ccmXY$E[1]
  tau = ccmXY$tau[1]
  vecColours <- colorRampPalette(rev(brewer.pal(10,'Spectral')))(50)
  strTitle = paste(strXDescription, " xmap ", strYDescription, "(E", E, ",", "tau", tau, "," , num_samples , " samples)")
  
  uniqueLibrarySizes <- sort(unique(ccmXY$lib_size))
  maxLibrarySize <- max(uniqueLibrarySizes)
  # because there can be an extra small number in the transition between from the short step size to the longer one, to get the smallest
  # standard step size, get the SECOND smallest one
  smallestStandardStepL <- diff(uniqueLibrarySizes)[2]
  
  countDistinctLibrarySizes <- maxLibrarySize/smallestStandardStepL
  countRhoBins <- 100
  
  gPlot <- createGGPlotInternal(ccmXY, isDisplayMAE, isTruncatePerRealizationRho)
  
  png(strFileName)
  result <- gPlot + stat_bin2d(bins=c(countDistinctLibrarySizes, countRhoBins)) + scale_fill_gradientn(colours=vecColours) + ggtitle(strTitle)
  print(result)
  dev.off()
}

# parse the argument and input
# subprocess.call(["Rscript", Rfile, input_path, ParallelInput, num_samples, tauArr[0], EArr[0], LStart, LInterval, LEnd])
# read csv file name and do comparison

args = commandArgs(trailingOnly=TRUE)
if (length(args) < 7){
	stop("Please input the parameter file name: ", call.=FALSE)
}
# input_csv_file_for_rEDM = "/home/bo/cloud/CCM-Parralization/TestInputCSVData/test_float_1000.csv"
# ccm_csv_file_to_compare = "/home/bo/cloud/CCM-Parralization/Result/singlemachine.csv"
# samples = 250
# LStart = 100
# LEnd = 1000
# LInterval = 100
# E = 2
# tau = 1

# input csv file: full path
input_csv_file_for_rEDM = args[1]
# input custmoized ccm result for plot: full path
ccm_csv_file_to_compare = args[2]
# ccm parameters
samples = as.numeric(as.character(args[3]))
tau = as.numeric(as.character(args[4]))
E = as.numeric(as.character(args[5]))
LStart = as.numeric(as.character(args[6]))
LInterval = as.numeric(as.character(args[7]))
LEnd = as.numeric(as.character(args[8]))

print(paste("read csv file path: ", input_csv_file_for_rEDM, "; ccm c++ version output path: ", ccm_csv_file_to_compare, sep=""))

print(paste("samples #: ", samples, "; tau: ", tau, "; E: ", E, "; LStart: ", LStart, "; LInterval: ", LInterval, "; LEnd: ", LEnd, sep=""))

home_dir = getwd() # run the program in the parent directory

output_dir = paste(home_dir, "/ImageCompareResult/", sep="")

#  call ccm library in R
source = "x"
target = "y"
TimeSeries = read.csv(input_csv_file_for_rEDM)
vecLibrarySizes <- seq(LStart, LEnd, LInterval)

tic("running rEDM ccm libarary")
ccmXY <- ccm(TimeSeries, E=E, tau=tau, lib_column=source, target_column=target, lib_sizes=vecLibrarySizes, num_samples=samples)
toc()
ccmXY$nonnegRho = pmax(0, ccmXY$rho)

#create plot for rEDM
print("start to plot ccm library result")
SaveFileName_REDM = paste(output_dir, "ccm_rEDM", "_samples_", samples, "_E_", E, "_tau_", tau, ".png", sep="")
print(SaveFileName_REDM)
createPerRealizationDensityPlot(ccmXY, SaveFileName_REDM, isTruncatePerRealizationRho=TRUE, num_samples=samples)
print("finish the plot")


# create ccm c++ version parallel plot
print("start to plot ccm parallel result")
ccm_to_compare = read.csv(ccm_csv_file_to_compare)
# revise the col name to plot
colnames(ccm_to_compare) <- c("E", "tau", "lib_size", "rho")
ccm_to_compare$nonnegRho = pmax(0, ccm_to_compare$rho)

SaveFileName_COMPARE = paste(output_dir, "ccm_parallel", "_samples_", samples, "_E_", E, "_tau_", tau, ".png", sep="")
print(SaveFileName_COMPARE)
createPerRealizationDensityPlot(ccm_to_compare, SaveFileName_COMPARE,  strXDescription=ccmXY$lib_column[1], strYDescription=ccmXY$target_column[1], isTruncatePerRealizationRho=TRUE, num_samples=samples)
#createPerRealizationDensityPlot(ccm_to_compare, SaveFileName_COMPARE,  strXDescription="x", strYDescription="y", isTruncatePerRealizationRho=TRUE, num_samples=samples)
print("finish the plot")