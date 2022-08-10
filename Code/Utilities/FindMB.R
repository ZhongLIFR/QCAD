##This script is used to find Markov Blankets
##Author @Zhong Li, August 10, 2022, Leiden
################################################################################
##Example: abalone
################################################################################
#install.packages("bnlearn")

AbsRootDir = '/Users/zlifr/Documents/GitHub' ##you must specify this path by yourself

GenDataSetPath = paste(AbsRootDir, '/QCAD/Data/GenData/abaloneGene.csv', sep = "")
MBDataSetPath = paste(AbsRootDir, '/QCAD/Data/GenData/MB/abaloneGeneTest.csv', sep = "") 

library(bnlearn)
MyDataSet <- read.csv(file = GenDataSetPath)
MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]

for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
  if (typeof(MyDataSet[ , i]) == "integer"){
    MyDataSet[ , i] = as.factor(MyDataSet[ , i])
  }
}

for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
  Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
  sink(MBDataSetPath, append = TRUE)
  cat(paste(Node1, collapse = ';')) ##print without index
  cat("\n")
  sink()
}
rm(list=ls())


