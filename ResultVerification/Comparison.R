# read csv file name and do comparison
args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0){
	stop("Please input the csv file name: ", call.=FALSE)
}
# input csv file
csv_file = args[1]
# input custmoized ccm result for plot
plot_file_to_compare = args[2]
# ccm parameters
samples = as.numeric(as.character(args[3]))
tau = as.numeric(as.character(args[4]))
E = as.numeric(as.character(args[5]))
lib_size = as.numeric(as.character(args[6]))

print(paste("read csv file: ", csv_file, "; customized output: ", plot_file_to_compare, "; samples #: ", samples, "; tau: ", tau, "; E: ", E, "; L: ", lib_size, sep=""))

library("ggplot2")
library("hexbin")
library("RColorBrewer")
library("MASS")
library("rEDM")
library('reshape2')

home_dir = getwd() # run the program in the parent directory
ccm_parallel_csv_dir = paste(home_dir, "/ParallelCCMOutput/", sep="")
ccm_csv_dir = paste(home_dir, "/TestCSVData/", sep="")
output_dir = paste(home_dir, "/ImageCompareResult", sep="")

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
createPerRealizationDensityPlot <- function(ccmXY, strFileName, strXDescription=ccmXY$lib_column[1], strYDescription=ccmXY$target_column[1], isTruncatePerRealizationRho=FALSE, isDisplayMAE=FALSE)
{
  # EEmbeddingDimension = ccmXY$E[1] 
  countSamplesPerL = ccmXY$num_pred[1]
  # tau = ccmXY$tau[1] 
  
  vecColours <- colorRampPalette(rev(brewer.pal(10,'Spectral')))(50)
  strTitle = paste(strXDescription, " xmap ", strYDescription, "(E", E, ",", "tau", tau, "," , samples , " samples)")
  
  uniqueLibrarySizes <- unique(ccmXY$lib_size)
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

source = "x"
target = "y"
TimeSeries = read.csv(paste(ccm_csv_dir, csv_file, sep=""))
maxL <- min(lib_size, nrow(TimeSeries[target]))
stepL <- 100
vecLibrarySizes <- seq(50, maxL, stepL)
ccmXY <- ccm(TimeSeries, E=E, tau=tau, lib_column=source, target_column=target, lib_sizes=vecLibrarySizes, num_samples=samples)
ccmXY$nonnegRho = pmax(0, ccmXY$rho)

LongestL = max(ccmXY$lib_size)
print(paste("The longest L is:", LongestL))



