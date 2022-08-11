#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:40:41 2021

@author: zlifr
"""
##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub'  


import warnings
warnings.filterwarnings("ignore")

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

import sys
ImpleDir = AbsRootDir+ r'/QCAD/Code/Implementation'
UtilityDir = AbsRootDir+ r'/QCAD/Code/Utilities'
sys.path.append(ImpleDir)
sys.path.append(UtilityDir)

from ContextualAnomalyInject import GenerateData
from QCAD import QCAD
from PyODTest import PyODModel
from LoPAD import LoPAD
from ROCOD import ROCOD




# =============================================================================
# #Step1. Define a function to calculate anomaly score based on given datasets
# =============================================================================

def SynAverageTest(FilePath,
                ResultFilePath, 
                MB_dataset_path,
                SaveFilePath_QCAD,
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
    myResult_QCAD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
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

        ##This is for QCAD
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, TempDataSet = QCAD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                                        neighbour_value, sample_value)
        
        myResult_QCAD.loc[len(myResult_QCAD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
    
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
   
    
    
    ## For QCAD
    print("QCAD start")
    myResult2 = myResult_QCAD.copy()
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
        
    myResult_QCAD = myResult2.drop_duplicates()
    myResult_QCAD.to_csv(SaveFilePath_QCAD, sep=',')
    
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
      

# =============================================================================
# Step 2. Example ->  SynDataSet7
# =============================================================================

###############################################################################
## Step 2.1. Parameter setting
###############################################################################

FilePath = AbsRootDir + r'/QCAD/Data/SynData/SynDataSet7.csv'
ResultFilePath = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7Gene.csv'
SaveFilePath_QCAD = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_QCAD.csv'
SaveFilePath_LoPAD = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_LoPAD.csv'
SaveFilePath_ROCOD = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_ROCOD.csv'
SaveFilePath_IForest = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_IForest.csv'
SaveFilePath_LOF = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_LOF.csv'
SaveFilePath_KNN = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_KNN.csv'
SaveFilePath_SOD = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_SOD.csv'
SaveFilePath_HBOS = AbsRootDir + r'/QCAD/Data/TempFiles/SynDataSet7_HBOS.csv'
MB_dataset_path =  AbsRootDir + r'/QCAD/Data/SynData/MB/SynDataSet7.csv'


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

       
###############################################################################
## Step 2.2. Execute the function
###############################################################################
SynAverageTest(FilePath,
               ResultFilePath,
               MB_dataset_path,
               SaveFilePath_QCAD,
               SaveFilePath_LoPAD,
               SaveFilePath_ROCOD,
               SaveFilePath_IForest,
               SaveFilePath_LOF,
               SaveFilePath_KNN,
               SaveFilePath_SOD,
               SaveFilePath_HBOS,
               AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
               anomaly_value, sample_value, neighbour_value, num_dataset)





