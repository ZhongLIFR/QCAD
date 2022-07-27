#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:40:41 2021

@author: zlifr
"""
import warnings
warnings.filterwarnings("ignore")

###############################################################################
###############################################################################
##Step1 define a function to calculate anomaly score 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import percentile

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import percentile
import numpy as np 

from pyod.models.knn import KNN   # kNN detector
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF ##
from pyod.models.iforest import IForest ##
from pyod.models.ocsvm import OCSVM ##
from pyod.models.sod import SOD ## subspace outlier detection

###############################################################################
###############################################################################
##Step2 generate different dataset to use
from ContextualAnomalyInjectFinal import GenerateData
# from ICAD_QRF import ICAD_QRF
from RICAD_QRF import ICAD_QRF
from PyODTest import PyODModel
from LoPAD import LoPAD
from ROCOD import ROCOD


###############################################################################
##DataSet1:  Abalone
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/abalone.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/abaloneGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/abalone_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/abaloneGene.csv"

# AllCols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
#             'Viscera weight', 'Shell weight', 'Rings']

# AllColsWithTruth = ['Sex', 'Length', 'Diameter', 'Height',
#                     'Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings',
#                     'ground_truth']

# ContextCols = ['Sex', 'Length', 'Diameter', 'Height']

# BehaveCols = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

# NumCols = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

# anomaly_value = 100

# sample_value = 100

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet2:  airfoil
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Airfoil.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/AirfoilGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/AirfoilGene.csv"

# AllCols = ['f', 'alpha', 'c', 'U_infinity', 'delta', 'SSPL']

# AllColsWithTruth = ['f', 'alpha', 'c', 'U_infinity', 'delta', 'SSPL',
#                     'ground_truth']

# ContextCols = ['f', 'alpha', 'c', 'U_infinity', 'delta']

# BehaveCols = ['SSPL']

# NumCols =  ['SSPL']

# anomaly_value = 70

# sample_value = 70

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet3:  bodyfat
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Bodyfat.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/BodyfatGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/BodyfatGene.csv"


# AllCols = ['Density', 'BodyFat', 
#             'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# AllColsWithTruth = ['Density', 'BodyFat', 
#                     'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
#                     'ground_truth']

# ContextCols = ['Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# BehaveCols = ['Density', 'BodyFat']

# NumCols = ['Density', 'BodyFat']

# anomaly_value = 20

# sample_value = 20

# neighbour_value = 126

# num_dataset = 10

###############################################################################
##DataSet4:  boston
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/boston.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/bostonGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/bostonGene.csv"

# AllCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

# AllColsWithTruth = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV',
#                     'ground_truth']

# ContextCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']

# BehaveCols = ['MEDV']

# NumCols = ['MEDV']

# anomaly_value = 40

# sample_value = 40

# neighbour_value = 253

# num_dataset = 10

###############################################################################
##DataSet5:  Concrete 
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Concrete.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/ConcreteGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/ConcreteGene.csv"

# AllCols = ["C1","C2","C3","C4","C5","C6","C7","Age","Strength"]

# AllColsWithTruth = ["C1","C2","C3","C4","C5","C6","C7","Age","Strength",
#                     'ground_truth']

# ContextCols = ["C1","C2","C3","C4","C5","C6","C7","Age",]

# BehaveCols = ['Strength']

# NumCols =  ['Strength']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet6:  elnino 
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Elnino.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/ElninoGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Elnino_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/ElninoGene.csv"

# AllCols = ['Year', 'Month', 'Day', 'Date', 'Latitude','Longitude',
#             'Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp']

# AllColsWithTruth =['Year', 'Month', 'Day', 'Date', 'Latitude','Longitude',
#                     'Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp',
#                     'ground_truth']

# ContextCols = ['Year', 'Month', 'Day', 'Date', 'Latitude','Longitude']

# BehaveCols = ['Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp']

# NumCols =  ['Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp']

# anomaly_value = 200

# sample_value = 200

# neighbour_value = 500

# num_dataset = 5


###############################################################################
##DataSet7:  Energy
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/energy.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/energyGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/energyGene.csv"

# AllCols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']

# AllColsWithTruth = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2',
#                     'ground_truth']

# ContextCols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# BehaveCols = ['Y1', 'Y2']

# NumCols =  ['Y1', 'Y2']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 384

# num_dataset = 10

###############################################################################
##DataSet8:  fish weight
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Fish.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/FishGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/FishGene.csv"


# AllCols = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']

# AllColsWithTruth = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height','Width',
#                     'ground_truth']

# ContextCols = ['Species', 'Length1', 'Length2', 'Length3', 'Height','Width']

# BehaveCols = ['Weight']

# NumCols =  ['Weight']

# anomaly_value = 15

# sample_value = 15

# neighbour_value = 80

# num_dataset = 10

###############################################################################
##DataSet9:  ForestFires
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/forestFires.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFiresGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/forestFiresGene.csv"

# AllCols = ['X', 'Y', 'month', 'day', 
#             'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# AllColsWithTruth = ['X', 'Y', 'month', 'day', 
#                     'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area',
#                     'ground_truth']

# ContextCols = ['X', 'Y', 'month', 'day']

# BehaveCols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# NumCols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 258

# num_dataset = 10

###############################################################################
##DataSet10:  gasEmission 
###############################################################################
FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/GasEmission.csv"
ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmissionGene.csv"
SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_ICAD.csv"
SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_LoPAD.csv"
SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_ROCOD.csv"

SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_COD.csv" ##Too slow!

SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_IForest.csv"
SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_LOF.csv"
SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_KNN.csv"
SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_SOD.csv"
SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/GasEmission_HBOS.csv"

MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/GasEmissionGene.csv"



AllCols = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO','NOX']

AllColsWithTruth = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO','NOX',
                    'ground_truth']

ContextCols = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']

BehaveCols = ['TEY', 'CO','NOX']

NumCols =  ['TEY', 'CO','NOX']

anomaly_value = 100

sample_value = 100

neighbour_value = 500

num_dataset = 5

###############################################################################
##DataSet11:  heartFailure
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/heartFailure.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailureGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/heartFailureGene.csv"

# AllCols = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia",
#             "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time",
#             "DEATH_EVENT"]

# AllColsWithTruth = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia",
#                     "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time",
#                     "DEATH_EVENT",
#                     'ground_truth']

# ContextCols = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia"]

# BehaveCols = ["creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

# NumCols = ["creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

# anomaly_value = 30

# sample_value = 30

# neighbour_value = 150

# num_dataset = 10

###############################################################################
##DataSet12:  hepatitis
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/hepatitis.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitisGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/hepatitisGene.csv"


# AllCols = ['Category', 'Age', 'Sex', 
#             'ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# AllColsWithTruth = ['Category', 'Age', 'Sex', 
#                     'ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT',
#                     'ground_truth']

# ContextCols = ['Category', 'Age', 'Sex']

# BehaveCols = ['ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# NumCols =  ['ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# anomaly_value = 30

# sample_value = 30

# neighbour_value = 308

# num_dataset = 10


###############################################################################
##DataSet13:  indianLiverPatient
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/indianLiverPatient.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatientGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/indianLiverPatientGene.csv"


# AllCols = ['Age', 'Gender', 'Selector',
#             'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# AllColsWithTruth = ['Age', 'Gender', 'Selector',
#                     'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#                     'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio',
#                     'ground_truth']

# ContextCols = ['Age', 'Gender', 'Selector']

# BehaveCols = ['Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#               'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# NumCols =  ['Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# anomaly_value = 30

# sample_value = 30

# neighbour_value = 290

# num_dataset = 10


###############################################################################
##DataSet14:  Maintenance 
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Maintenance.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/MaintenanceGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Maintenance_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/MaintenanceGene.csv"

# AllCols = ["LeverPosition", "ShipSpeed", "GTT", "GTn",
#             "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48",
#             "P1", "P2", "Pexh", "TIC", "mf",
#             "CompressorDecay", "TurbineDecay"]

# AllColsWithTruth = ["LeverPosition", "ShipSpeed", "GTT", "GTn",
#                     "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48",
#                     "P1", "P2", "Pexh", "TIC", "mf",
#                     "CompressorDecay", "TurbineDecay",
#                     'ground_truth']

# ContextCols = ["LeverPosition", "GTT", "GTn",
#                 "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48",
#                 "P1", "P2", "Pexh", "TIC", "mf"]

# BehaveCols = ['ShipSpeed',"CompressorDecay", "TurbineDecay"]

# NumCols =  ['ShipSpeed',"CompressorDecay", "TurbineDecay"]

# anomaly_value = 100

# sample_value = 100

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet15:  parkinson  
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Parkinson.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/ParkinsonGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Parkinson_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/ParkinsonGene.csv"


# AllCols = ['subject', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
#            'Jitter', 'Jitter_Abs', 'Jitter_RAP', 'Jitter_PPQ5', 'Jitter_DDP',
#            'Shimmer', 'Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5',
#            'Shimmer_APQ11', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

# AllColsWithTruth = ['subject', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
#                     'Jitter', 'Jitter_Abs', 'Jitter_RAP', 'Jitter_PPQ5', 'Jitter_DDP',
#                     'Shimmer', 'Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5',
#                     'Shimmer_APQ11', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
#                     'ground_truth']

# ContextCols = ['subject', 'age', 'sex', 'test_time', 
#               'Jitter', 'Jitter_Abs', 'Jitter_RAP', 'Jitter_PPQ5', 'Jitter_DDP',
#               'Shimmer', 'Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5',
#               'Shimmer_APQ11', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

# BehaveCols = ['motor_UPDRS', 'total_UPDRS']

# NumCols =  ['motor_UPDRS', 'total_UPDRS']

# anomaly_value = 100

# sample_value = 100

# neighbour_value = 500

# num_dataset = 5


###############################################################################
##DataSet16:  power plant
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Power.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/PowerGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Power_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/PowerGene.csv"

# AllCols = ['AT', 'V', 'AP', 'RH', 'PE']

# AllColsWithTruth = ['AT', 'V', 'AP', 'RH','PE',
#                     'ground_truth']

# ContextCols = ['AT', 'V', 'AP', 'RH']

# BehaveCols = ['PE']

# NumCols =  ['PE']

# anomaly_value = 100

# sample_value = 100

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet17:  QSRanking 
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/QSRanking.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRankingGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/QSRankingGene.csv"


# AllCols = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
#             'SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
#             'Academic Reputation', 'Employer Reputation', 'Faculty Student',
#             'Citations per Faculty', 'International Faculty',
#             'International Students']

# AllColsWithTruth = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
#                     'SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
#                     'Academic Reputation', 'Employer Reputation', 'Faculty Student',
#                     'Citations per Faculty', 'International Faculty',
#                     'International Students',
#                     'ground_truth']

# ContextCols = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country']

# BehaveCols = ['Academic Reputation', 'Employer Reputation', 'Faculty Student',
#               'Citations per Faculty', 'International Faculty',
#               'International Students']

# NumCols = ['Academic Reputation', 'Employer Reputation', 'Faculty Student',
#             'Citations per Faculty', 'International Faculty', 'International Students']

# anomaly_value = 40

# sample_value = 40

# neighbour_value = 237

# num_dataset = 10


###############################################################################
##DataSet18:  synchronousMachine  
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/SynchronousMachine.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachineGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/SynchronousMachineGene.csv"


# AllCols = ['Iy', 'PF', 'e', 'dIf', 'If']

# AllColsWithTruth = ['Iy', 'PF', 'e', 'dIf', 'If',
#                     'ground_truth']

# ContextCols = ['Iy', 'PF', 'e', 'dIf']

# BehaveCols = ['If']

# NumCols =  ['If']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 275

# num_dataset = 10

###############################################################################
##DataSet19:  Toxicity
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Toxicity.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/ToxicityGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/ToxicityGene.csv"


# AllCols = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50"]

# AllColsWithTruth = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50",
#                     'ground_truth']

# ContextCols = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP"]

# BehaveCols = ['LC50']

# NumCols =  ['LC50']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 450

# num_dataset = 10

###############################################################################
##DataSet20:  yachtHydrodynamics
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/YachtHydrodynamics.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamicsGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/YachtHydrodynamicsGene.csv"


# AllCols = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#             'Beam_draught_ratio', 'Length_beam_ratio','Froude_number', 
#             'resistance']

# AllColsWithTruth = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#                     'Beam_draught_ratio', 'Length_beam_ratio','Froude_number', 
#                     'resistance',
#                     'ground_truth']

# ContextCols = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#                 'Beam_draught_ratio', 'Length_beam_ratio','Froude_number']

# BehaveCols = ['resistance']

# NumCols = ['resistance']

# anomaly_value = 30

# sample_value = 30

# neighbour_value = 155

# num_dataset = 10


###############################################################################
###############################################################################
##Step3 Calculate snomaly score based on given datasets

##def a function to do sensitivity analysis
def AverageTest(FilePath,
                ResultFilePath, 
                MB_dataset_path,
                SaveFilePath_ICAD,
                SaveFilePath_LoPAD,
                SaveFilePath_ROCOD,
                SaveFilePath_IForest,
                SaveFilePath_LOF,
                SaveFilePath_KNN,
                SaveFilePath_SOD,
                SaveFilePath_HBOS,
                MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                NumCols, anomaly_value, sample_value, neighbour_value,
                num_dataset):
    
    import pandas as pd
    myResult_ICAD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
    myResult_LoPAD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])    
    myResult_ROCOD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])    
    myResult_IForest =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
    myResult_LOF =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
    myResult_KNN=  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
    myResult_SOD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])    
    myResult_HBOS =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
   
   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
        MyBehaveList = BehaveCols
           
        #for each dataset, calculate score
        
        ##This is for LoPAD
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = LoPAD(MyDataSet, MB_dataset_path, sample_value)
        myResult_LoPAD.loc[len(myResult_LoPAD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]

        ##This is for ROCOD
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = ROCOD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList,
                                                                            0.9, 0,
                                                                            r'/Users/zlifr/Desktop/HHBOS/TrainedModel/COD/distance_matrix_EnergyGene.npy',
                                                                            0, MyDataSet)

        myResult_ROCOD.loc[len(myResult_ROCOD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]

        ##This is for ICAD_QRF
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, TempDataSet = ICAD_QRF(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                                            neighbour_value, anomaly_value, sample_value)
        
        myResult_ICAD.loc[len(myResult_ICAD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
    
        ##This is for IForest
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = PyODModel(MyDataSet, sample_value, IForest())
        myResult_IForest.loc[len(myResult_IForest)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
        
        ##This is for LOF
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = PyODModel(MyDataSet, sample_value, LOF())
        myResult_LOF.loc[len(myResult_LOF)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
        
        ##This is for KNN
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = PyODModel(MyDataSet, sample_value, KNN())
        myResult_KNN.loc[len(myResult_KNN)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
       
        ##This is for SOD
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = PyODModel(MyDataSet, sample_value, SOD())
        myResult_SOD.loc[len(myResult_SOD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
        
        ##This is for HBOS
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = PyODModel(MyDataSet, sample_value, HBOS())
        myResult_HBOS.loc[len(myResult_HBOS)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
   
    
    
    ## For ICAD
    print("ICAD start")
    myResult2 = myResult_ICAD.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_ICAD = myResult2.drop_duplicates()
    myResult_ICAD.to_csv(SaveFilePath_ICAD, sep=',')
    
    ## For LoPAD
    print("LoPAD start")
    myResult2 = myResult_LoPAD.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_LoPAD = myResult2.drop_duplicates()
    myResult_LoPAD.to_csv(SaveFilePath_LoPAD, sep=',')
    
    ## For ROCOD
    print("ROCOD start")
    myResult2 = myResult_ROCOD.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_ROCOD = myResult2.drop_duplicates()
    myResult_ROCOD.to_csv(SaveFilePath_ROCOD, sep=',')
    
    ##For IForest
    print("IForest start")
    myResult2 = myResult_IForest.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_IForest = myResult2.drop_duplicates()
    myResult_IForest.to_csv(SaveFilePath_IForest, sep=',')

    ##For LOF
    print("LOF start")
    myResult2 = myResult_LOF.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_LOF = myResult2.drop_duplicates()
    myResult_LOF.to_csv(SaveFilePath_LOF, sep=',')

    ##For KNN
    print("KNN start")
    myResult2 = myResult_KNN.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_KNN = myResult2.drop_duplicates()
    myResult_KNN.to_csv(SaveFilePath_KNN, sep=',')

    ##For SOD
    print("SOD start")
    myResult2 = myResult_SOD.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_SOD = myResult2.drop_duplicates()
    myResult_SOD.to_csv(SaveFilePath_SOD, sep=',')

    ##For HBOS
    print("HBOS start")
    myResult2 = myResult_HBOS.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_HBOS = myResult2.drop_duplicates()
    myResult_HBOS.to_csv(SaveFilePath_HBOS, sep=',')    
             
#execute the function
AverageTest(FilePath,
            ResultFilePath,
            MB_dataset_path,
            SaveFilePath_ICAD,
            SaveFilePath_LoPAD,
            SaveFilePath_ROCOD,
            SaveFilePath_IForest,
            SaveFilePath_LOF,
            SaveFilePath_KNN,
            SaveFilePath_SOD,
            SaveFilePath_HBOS,
            AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
            anomaly_value, sample_value, neighbour_value, num_dataset)

