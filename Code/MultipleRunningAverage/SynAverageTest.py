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
# from ContextualAnomalyInject import GenerateData
from ContextualAnomalyInjectFinal import GenerateData
# from ICAD_QRF import ICAD_QRF
from RICAD_QRF import ICAD_QRF
from PyODTest import PyODModel
from LoPAD import LoPAD
from ROCOD import ROCOD
###############################################################################
##DataSet1:  SynDataSet1  done
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet1.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2', 
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']


# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet2:  SynDataSet2  done
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet2.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet2.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet3:  SynDataSet3  done
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet3.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet3.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet4:  SynDataSet4  done
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet4.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet4.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet5:  SynDataSet5 done
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet5.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet5.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10

###############################################################################
##DataSet6:  SynDataSet6
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet6.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet6.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#            'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#            'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#            'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#            'behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                     'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                     'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                     'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4',
#                     'ground_truth']


# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10


###############################################################################
##DataSet7:  SynDataSet7
###############################################################################
FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet7.csv"
ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7Gene.csv"
SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_ICAD.csv"
SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_LoPAD.csv"
SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_ROCOD.csv"

SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_COD.csv" ##Too slow!

SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_IForest.csv"
SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_LOF.csv"
SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_KNN.csv"
SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_SOD.csv"
SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_HBOS.csv"

MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet7.csv"


AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
            'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
            'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
            'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
            'behave_num4']

AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
                    'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
                    'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
                    'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
                    'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
                    'behave_num4',
                    'ground_truth']


ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
                'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
                'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
                'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5']

BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

anomaly_value = 50

sample_value = 50

neighbour_value = 500

num_dataset = 10


# ###############################################################################
# ##DataSet8:  SynDataSet8
# ###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet8.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet8.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#            'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#            'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#            'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#            'behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                     'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                     'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                     'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4',
#                     'ground_truth']


# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10


###############################################################################
##DataSet9:  SynDataSet9
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet9.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet9.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#            'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#            'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#            'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#            'behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                     'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                     'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                     'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4',
#                     'ground_truth']


# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10


###############################################################################
##DataSet10:  SynDataSet10
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet10.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_ICAD.csv"
# SaveFilePath_LoPAD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_LoPAD.csv"
# SaveFilePath_ROCOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_ROCOD.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_COD.csv" ##Too slow!

# SaveFilePath_IForest = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_IForest.csv"
# SaveFilePath_LOF = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_LOF.csv"
# SaveFilePath_KNN = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_KNN.csv"
# SaveFilePath_SOD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_SOD.csv"
# SaveFilePath_HBOS = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_HBOS.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet10.csv"


# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#            'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#            'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#            'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#            'behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                     'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                     'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                     'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4',
#                     'ground_truth']


# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 10


###############################################################################
###############################################################################
##Step3 Calculate snomaly score based on given datasets

##def a function to do sensitivity analysis
def SynAverageTest(FilePath,
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
SynAverageTest(FilePath,
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

