
files <- list.files("./data/")
for(file in files){
  print(file)
  indata <- load(paste("./data/", file, sep=""))
  outputFile = paste("./data/", strsplit(file, split=".",fixed=TRUE)[[1]][1], ".csv", sep = "")
  print(paste("output file:", outputFile))
  write.csv(get(indata), file=outputFile)
}
