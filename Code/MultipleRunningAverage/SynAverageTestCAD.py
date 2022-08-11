#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:29:14 2021

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
from ContextualAnomalyInject import GenerateData
from ICAD_QRF import ICAD_QRF
from PyODTest import PyODModel
from LoPAD import LoPAD
from ROCOD import ROCOD
from COD import COD
###############################################################################
##DataSet1:  SynDataSet4
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_COD.csv" ##Too slow!



# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3', 'con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 100

# sample_value = 100

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet2:  SynDataSet5
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet2_COD.csv" ##Too slow!



# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 200

# sample_value = 200

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet3:  SynDataSet3
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet3_COD.csv" ##Too slow!

# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet4:  SynDataSet2
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4Gene.csv"


# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet4_COD.csv" ##Too slow!



# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 20

# sample_value = 20

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet5:  SynDataSet1
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet5_COD.csv" ##Too slow!


# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 10

# sample_value = 10

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet6:  SynDataSet6
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet6.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet6_COD.csv" ##Too slow!


# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#             'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#             'con_num11', 'con_num12', 'con_num13', 'con_num14', 'con_num15',
#             'con_num16', 'con_num17', 'con_num_cat0', 'con_num_cat1',
#             'behave_num0','behave_num1', 'behave_num2', 'behave_num3', 'behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                     'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                     'con_num11', 'con_num12', 'con_num13', 'con_num14', 'con_num15',
#                     'con_num16', 'con_num17', 'con_num_cat0', 'con_num_cat1', 
#                     'behave_num0','behave_num1', 'behave_num2', 'behave_num3', 'behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                 'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                 'con_num11', 'con_num12', 'con_num13', 'con_num14', 'con_num15',
#                 'con_num16', 'con_num17', 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 5


###############################################################################
##DataSet7:  SynDataSet7
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet7.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet7_COD.csv" ##Too slow!

# AllCols = ['con_num0', 'con_num1', 'con_num2', 
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0','behave_num1', 'behave_num2', 'behave_num3', 'behave_num4',
#             'behave_num5','behave_num6', 'behave_num7', 'behave_num8', 'behave_num9' ]

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0','behave_num1', 'behave_num2', 'behave_num3', 'behave_num4',
#                     'behave_num5','behave_num6', 'behave_num7', 'behave_num8', 'behave_num9',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',  
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0','behave_num1', 'behave_num2', 'behave_num3', 'behave_num4',
#               'behave_num5','behave_num6', 'behave_num7', 'behave_num8', 'behave_num9']

# NumCols = ['behave_num0','behave_num1', 'behave_num2', 'behave_num3', 'behave_num4',
#             'behave_num5','behave_num6', 'behave_num7', 'behave_num8', 'behave_num9']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 5

# ###############################################################################
# ##DataSet8:  SynDataSet8
# ###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet8.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet8_COD.csv" ##Too slow!

# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#             'behave_num4', 'behave_num5', 'behave_num6', 'behave_num7',
#             'behave_num8', 'behave_num9', 'behave_num10', 'behave_num11',
#             'behave_num12', 'behave_num13', 'behave_num14', 'behave_num15',
#             'behave_num16', 'behave_num17', 'behave_num18', 'behave_num19']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4', 'behave_num5', 'behave_num6', 'behave_num7',
#                     'behave_num8', 'behave_num9', 'behave_num10', 'behave_num11',
#                     'behave_num12', 'behave_num13', 'behave_num14', 'behave_num15',
#                     'behave_num16', 'behave_num17', 'behave_num18', 'behave_num19',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',  
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#               'behave_num4', 'behave_num5', 'behave_num6', 'behave_num7',
#               'behave_num8', 'behave_num9', 'behave_num10', 'behave_num11',
#               'behave_num12', 'behave_num13', 'behave_num14', 'behave_num15',
#               'behave_num16', 'behave_num17', 'behave_num18', 'behave_num19']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#             'behave_num4', 'behave_num5', 'behave_num6', 'behave_num7',
#             'behave_num8', 'behave_num9', 'behave_num10', 'behave_num11',
#             'behave_num12', 'behave_num13', 'behave_num14', 'behave_num15',
#             'behave_num16', 'behave_num17', 'behave_num18', 'behave_num19']

# anomaly_value = 50

# sample_value = 50
# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet9:  SynDataSet9
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet9.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9Gene.csv"

# SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet9_COD.csv" ##Too slow!


# AllCols = ['con_num_cat0', 'con_num_cat1', 'con_num_cat2', 'con_num_cat3',
#             'con_num_cat4', 'con_num_cat5', 'con_num_cat6', 'con_num_cat7',
#             'con_num_cat8', 'con_num_cat9', 'con_num_cat10', 'con_num_cat11',
#             'con_num_cat12', 'con_num_cat13', 'con_num_cat14', 'con_num_cat15',
#             'con_num_cat16', 'con_num_cat17', 'con_num_cat18', 'con_num_cat19',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#             'behave_num4']

# AllColsWithTruth = ['con_num_cat0', 'con_num_cat1', 'con_num_cat2', 'con_num_cat3',
#                     'con_num_cat4', 'con_num_cat5', 'con_num_cat6', 'con_num_cat7',
#                     'con_num_cat8', 'con_num_cat9', 'con_num_cat10', 'con_num_cat11',
#                     'con_num_cat12', 'con_num_cat13', 'con_num_cat14', 'con_num_cat15',
#                     'con_num_cat16', 'con_num_cat17', 'con_num_cat18', 'con_num_cat19',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num_cat0', 'con_num_cat1', 'con_num_cat2', 'con_num_cat3',
#                 'con_num_cat4', 'con_num_cat5', 'con_num_cat6', 'con_num_cat7',
#                 'con_num_cat8', 'con_num_cat9', 'con_num_cat10', 'con_num_cat11',
#                 'con_num_cat12', 'con_num_cat13', 'con_num_cat14', 'con_num_cat15',
#                 'con_num_cat16', 'con_num_cat17', 'con_num_cat18', 'con_num_cat19']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#               'behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#             'behave_num4']

# anomaly_value = 50

# sample_value = 50
# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet10:  SynDataSet10
###############################################################################
FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet10.csv"
ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10Gene.csv"

SaveFilePath_COD = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet10_COD.csv" ##Too slow!

MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/MB/SynDataSet10.csv"


AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
            'con_num_cat0', 'con_num_cat1',
            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
                    'con_num_cat0', 'con_num_cat1',
                    'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
                    'ground_truth']

ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
                'con_num_cat0', 'con_num_cat1']

BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

anomaly_value = 500

sample_value = 500

neighbour_value = 5000

num_dataset = 5

###############################################################################
###############################################################################
##Step3 Calculate snomaly score based on given datasets

##def a function to do sensitivity analysis
def SynAverageTest(FilePath,
                    ResultFilePath, 
                    SaveFilePath_COD,
                    MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                    NumCols, anomaly_value, sample_value, neighbour_value,
                    num_dataset):
    
    import pandas as pd
    myResult_COD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])

   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
        MyBehaveList = BehaveCols
           
        #for each dataset, calculate score
        
        ##This is for COD
        is_model_learned = 0
        FilePath_MappingMatrix = r'/Users/zlifr/Desktop/HHBOS/TrainedModel/COD/test_mapping_matrix.npy'
        num_gau_comp = 5
        # num_gau_comp = 1
        alpha_log = 0.001
        
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = COD(MyDataSet, MyContextList, MyBehaveList, num_gau_comp, alpha_log, is_model_learned, FilePath_MappingMatrix)
        myResult_COD.loc[len(myResult_COD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]


    
    ## For COD
    print("COD start")
    myResult2 = myResult_COD.copy()
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
    myResult_ICAD.to_csv(SaveFilePath_COD, sep=',')
    
             
#execute the function
SynAverageTest(FilePath,
                ResultFilePath,
                SaveFilePath_COD,
                AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
                anomaly_value, sample_value, neighbour_value, num_dataset)

