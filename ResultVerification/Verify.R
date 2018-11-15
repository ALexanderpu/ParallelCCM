#read scenario #
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("Please input the scenario #", call.=FALSE)
}
#ScenarioNum = strtoi(args[1])
ScenarioNum = args[1]
# import library
library("ggplot2")
library("hexbin")
library("RColorBrewer")
library("MASS")
library("rEDM")
library('reshape2')

samples = 250
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

createPerRealizationDensityPlot <- function(ccmXY, window, strFileName, strXDescription=ccmXY$lib_column[1], strYDescription=ccmXY$target_column[1], isTruncatePerRealizationRho=FALSE, isDisplayMAE=FALSE)
{
  EEmbeddingDimension = ccmXY$E[1] 
  countSamplesPerL = ccmXY$num_pred[1]
  tau = ccmXY$tau[1] 
  
  vecColours <- colorRampPalette(rev(brewer.pal(10,'Spectral')))(50)
  strTitle = paste(strXDescription, " xmap ", strYDescription, "(E", EEmbeddingDimension, ",", "tau", tau, ",","k", window, "," , samples , " samples)")
  
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





SuicideModel <- function(csv_df, window_size, Edimension){
  delay <- Edimension-1
  x = head(rowSums(embed(csv_df$Suicide.Attempt.Count, window_size)), -(delay+1)) 
  y = tail(csv_df$Suicide.Attempt.Count, -(window_size+delay))
  df <- data.frame(x, y)
  names(df) <- c("AccumulateCount","Count")
  return(df)
}
#"AccumulateCount"
# "Count"
CCMModel <- function(TimeSeries, lib, target, strScenario, E, window_size, saveFigureDir, saveFolderName){
  print(paste("start to run ccm for scenario", strScenario, "with combination E:", E, "window_size:", window_size))
  
  maxL <- nrow(TimeSeries[target])
  stepL <- 100
  vecLibrarySizes <- seq(50, maxL, stepL)
  ccmXY <- ccm(TimeSeries, E=E, tau=1, lib_column=lib, target_column=target, lib_sizes=vecLibrarySizes, num_samples=samples)
  ccmXY$nonnegRho = pmax(0, ccmXY$rho)
  
  # collect the mean value for the longest value
  LongestL = max(ccmXY$lib_size)
  print(paste("The longest L is:", LongestL))
  mean_rho = colMeans(subset(ccmXY, ccmXY$lib_size == LongestL, select=c("nonnegRho")))
  print(paste("mean value with combination E:", E, "window_size:", window_size, "is:", mean_rho))
  
  #create ccm plot
  print("start to plot ccm")
  SaveFileName = paste(saveFigureDir, saveFolderName, strScenario, "_E_", E, "_K_", window_size, ".png", sep="")
  print(SaveFileName)
  createPerRealizationDensityPlot(ccmXY, window_size, SaveFileName, isTruncatePerRealizationRho=TRUE)
  print("finish the plot")
  return(mean_rho)
}


# produce suicide x, y
home_dir = "/home/bo/Desktop/SuicideModels/"
#home_dir = "/Users/alexpb/Desktop/Lab/ABMCCM/CCMWithSuicideModeling/"
subfolder_name = "SuicideResult/"

strScenarioTitle = paste("Scenario_", ScenarioNum, sep="")
CsvFile = paste(home_dir, ScenarioNum, ".csv", sep="")
print(paste("read csv file:", CsvFile))
SuicideCounts = read.csv(CsvFile)

E_candidates <- c(2, 4, 6, 8)
window_size_candidates <- c(7, 14, 30, 90, 180, 360)

# create df for different E
df <- data.frame(window_size_candidates)
for(E in E_candidates){
  ccmVector <- c()  # plot meanRho-windowSize figure
  for(window_size in window_size_candidates){
    TimeSeries = SuicideModel(SuicideCounts, window_size, E)
    meanRho = CCMModel(TimeSeries, "AccumulateCount", "Count", strScenarioTitle, E, window_size, home_dir, subfolder_name)
    ccmVector <-c(ccmVector, meanRho)
  }
  df[[paste(strScenarioTitle, "_E_", E, sep = "")]] <- ccmVector
}

#save df
write.csv(df, file=paste(home_dir, subfolder_name , strScenarioTitle, "_meanRho.csv", sep = ""), row.names=TRUE)

# plot figure
strMeanFileName = paste(home_dir,subfolder_name, strScenarioTitle, "_meanRho.png", sep = "")
df$X<-NULL
melted = melt(df, id.vars="window_size_candidates")
png(strMeanFileName)
figure <- ggplot() + geom_line(data=melted, aes(x=window_size_candidates, y=value, group=variable, colour=variable)) + geom_line(size=2) + ggtitle("Suicide Model") + xlab("Window Size") + ylab("Mean Rho")
print(figure)
dev.off()

