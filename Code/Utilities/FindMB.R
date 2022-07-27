#install.packages("bnlearn")
################################################################################
##File0: abalone
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/abaloneGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/abaloneGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File1: forestFires
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/forestFiresGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/forestFiresGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File2: Energy
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/EnergyGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/EnergyGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File3: heartFailure
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/heartFailureGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/heartFailureGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File4: Hepatitis
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/hepatitisGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/hepatitisGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())
################################################################################
##File5: IndianLiverPatient
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/indianLiverPatientGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/indianLiverPatientGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File6: Maintenance
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/MaintenanceGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/MaintenanceGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File7: QSRanking
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/QSRankingGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/QSRankingGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File8: gasEmission
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/gasEmissionGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/gasEmissionGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File9: synchronousMachine
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/synchronousMachineGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/synchronousMachineGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())


################################################################################
##File10: parkinson
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/parkinsonGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/parkinsonGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File11: bodyfat
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/bodyfatGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/bodyfatGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File12: boston
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/bostonGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/bostonGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File13: yachtHydrodynamics
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/yachtHydrodynamicsGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/yachtHydrodynamicsGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File14: fish
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/fishGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/fishGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File15: airfoil
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/airfoilGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/airfoilGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File16: Concrete
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/ConcreteGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/ConcreteGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File17: toxicityGene
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/toxicityGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/toxicityGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File18: power
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/powerGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/powerGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File19: elnino
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/Data3/elninoGene.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet),ncol(MyDataSet))]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/Data3/MB/elninoGene.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File20: SynDataSet1
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet1.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())
################################################################################
##File21: SynDataSet2
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet2.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet2.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())
################################################################################
##File22: SynDataSet3
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet3.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet3.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())
################################################################################
##File23: SynDataSet4
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet4.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet4.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())
################################################################################
##File24: SynDataSet5
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet5.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet5.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File25: SynDataSet6
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet6.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet6.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File26: SynDataSet7
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet7.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet7.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File27: SynDataSet8
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet8.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet8.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

################################################################################
##File28: SynDataSet9
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet9.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet9.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())
################################################################################
##File29: SynDataSet10
################################################################################
# library(bnlearn)
# MyDataSet <- read.csv(file = '/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet10.csv')
# MyDataSet <- MyDataSet[ , -c(1, ncol(MyDataSet)+1, ncol(MyDataSet)+1)]
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   if (typeof(MyDataSet[ , i]) == "integer"){
#     MyDataSet[ , i] = as.factor(MyDataSet[ , i])
#   }
# }
# 
# for(i in 1:ncol(MyDataSet)) {       # for-loop over columns
#   Node1 <- learn.mb(MyDataSet, node = colnames(MyDataSet)[i], method = "fast.iamb")
#   sink("/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet10.csv", append = TRUE)
#   cat(paste(Node1, collapse = ';')) ##print without index
#   cat("\n")
#   sink()
# }
# rm(list=ls())

