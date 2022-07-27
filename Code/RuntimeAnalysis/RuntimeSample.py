#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 09:40:24 2022

@author: zlifr
"""

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# #Step1 import basic functions
# =============================================================================

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


# =============================================================================
# #Step2 import sef-defined funtions
# =============================================================================
# from ContextualAnomalyInject import GenerateData
from ContextualAnomalyInjectFinal import GenerateData
from RICAD_QRF import ICAD_QRF
from PyODTest import PyODModel
from LoPAD import LoPAD
from ROCOD import ROCOD
from COD import COD



# =============================================================================
# ##Step3 Calculate snomaly score based on given datasets
# =============================================================================

##def a function to do sensitivity analysis
def RuntimeSampleSize(FilePath,
                      ResultFilePath, 
                      MB_dataset_path,
                      MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                      NumCols, anomaly_value, sample_value, neighbour_value,
                      num_dataset):
    
    import pandas as pd
    from time import time
    
    result_time_df = pd.DataFrame(columns=["RICAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])
   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        time_vec = []
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
        ##This is for LoPAD
        LoPAD_time_start = time()        
        LoPAD(MyDataSet, MB_dataset_path, sample_value)       
        LoPAD_time_end = time()
        dur_LOPAD = round(LoPAD_time_end - LoPAD_time_start, ndigits=4)
        time_vec.append(dur_LOPAD)
        print("dur_LOPAD")
        print(dur_LOPAD)
        
        ##This is for ICAD_QRF
        ICAD_time_start = time()              
        ICAD_QRF(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                 neighbour_value, anomaly_value, sample_value)
        
        ICAD_time_end = time()
        dur_ICAD = round(ICAD_time_end - ICAD_time_start, ndigits=4)
        time_vec.append(dur_ICAD)
        
        
        ##This is for ROCOD
        ROCOD_time_start = time()       
        ROCOD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList,
              0.9, 0,
              r'',
              0, MyDataSet)        
        ROCOD_time_end = time()
        dur_ROCOD = round(ROCOD_time_end - ROCOD_time_start, ndigits=4)        
        time_vec.append(dur_ROCOD)
        
        ##This is for CAD
        COD_time_start = time()        
        COD(MyDataSet, MyContextList, MyBehaveList,
            5, 0.01, 0, r'/Users/zlifr/Desktop/HHBOS/TrainedModel/COD/mapping_matrix_EnergyGene.npy')        
        COD_time_end = time()
        dur_COD = round(COD_time_end - COD_time_start, ndigits=4)         
        time_vec.append(dur_COD)
        
        ##This is for IForest
        IForest_time_start = time()      
        PyODModel(MyDataSet, sample_value, IForest())
        IForest_time_end = time()
        dur_IForest = round(IForest_time_end - IForest_time_start, ndigits=4)         
        time_vec.append(dur_IForest)
        
        ##This is for LOF
        LOF_time_start = time()      
        PyODModel(MyDataSet, sample_value, LOF())   
        LOF_time_end = time()
        dur_LOF = round(LOF_time_end - LOF_time_start, ndigits=4)         
        time_vec.append(dur_LOF)
        
        ##This is for KNN
        KNN_time_start = time() 
        PyODModel(MyDataSet, sample_value, KNN())
        KNN_time_end = time()
        dur_KNN = round(KNN_time_end - KNN_time_start, ndigits=4)     
        time_vec.append(dur_KNN)
        
        ##This is for SOD
        SOD_time_start = time() 
        PyODModel(MyDataSet, sample_value, SOD())
        SOD_time_end = time()
        dur_SOD = round(SOD_time_end - SOD_time_start, ndigits=4)  
        time_vec.append(dur_SOD)
        
        ##This is for HBOS
        HBOS_time_start = time() 
        PyODModel(MyDataSet, sample_value, HBOS())
        HBOS_time_end = time()
        dur_HBOS = round(HBOS_time_end - HBOS_time_start, ndigits=4)  
        time_vec.append(dur_HBOS) 
                
        ##Add all values        
        time_new_df = pd.DataFrame([time_vec], columns=["RICAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])

        result_time_df = pd.concat([result_time_df, time_new_df])
        
        
    mean_time_list = list(result_time_df.mean(axis=0))

    return mean_time_list
    
# =============================================================================
# #Step4 generate synthetic datasets to test
# =============================================================================
from SyntheticDataFinal import GenSynDataset


result_time_df = pd.DataFrame(columns=["RICAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])
test_range = range(100,3000,100)
for num_sample_size in test_range:
    
    num_con_value = 3
    num_con_cat_value = 1
    num_behave_value = 2
            
    sample_size_value = num_sample_size
    
    MyDataSet = GenSynDataset(num_con =num_con_value , num_con_cat = num_con_cat_value, num_behave = num_behave_value,
                              sample_size = sample_size_value, num_gaussian = 5, my_scheme = "S1")
    MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSet.csv", sep=',')
    
        
    FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSet.csv"
    ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSetGene.csv"
    MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSetMB.csv"
    
    AllCols = list(MyDataSet.columns)
    AllColsWithTruth = AllCols.copy()
    AllColsWithTruth.append('ground_truth')
    
    ContextCols = list(MyDataSet.columns[0:num_con_value])
    BehaveCols = list(MyDataSet.columns[num_con_value:])
    NumCols = BehaveCols
    anomaly_value = 50
    sample_value = 50
    neighbour_value = min(int(sample_size_value/2), 500)
    num_dataset = 1
    
    
    #execute the function
    result_time_list = RuntimeSampleSize(FilePath,
                                       ResultFilePath,
                                       MB_dataset_path,
                                       AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
                                       anomaly_value, sample_value, neighbour_value, num_dataset)
        
    new_df = pd.DataFrame([result_time_list], columns=["RICAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])
    result_time_df = pd.concat([result_time_df, new_df])

result_time_df["SampleSize"] = list(test_range)
result_time_df.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SampleSizeResult.csv", sep=',') 

# =============================================================================
# #to display results
# =============================================================================
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3)

result_time_df2 = pd.read_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SampleSizeResult2.csv", sep=',') 
axs[0, 0].plot(result_time_df2["SampleSize"],result_time_df2["LOPAD"])
axs[0, 0].set_xticks(np.arange(0,3100,500))
# axs[0, 0].set_xlabel('Sample Size')
axs[0, 0].set_ylabel('Runtime of RICAD (sec)')
axs[0, 0].set_title("#C=3, #B=2", fontsize=8)

axs[1, 0].plot(result_time_df2["SampleSize"],result_time_df2["CAD"])
axs[1, 0].set_xticks(np.arange(0,3100,500))
axs[1, 0].set_xlabel('Sample Size')
axs[1, 0].set_ylabel('Runtime of CAD (sec)')
# axs[1, 0].set_title("#C=3, #B=2", fontsize=8)

result_time_df3 = pd.read_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/ContextResult2.csv", sep=',') 
axs[0, 1].plot(result_time_df3["ContextFeatureSize"],result_time_df3["LOPAD"])
axs[0, 1].set_xticks(np.arange(0,210,25))
axs[0, 1].set_title("N=500, #B=2", fontsize=8)


axs[1, 1].plot(result_time_df3["ContextFeatureSize"],result_time_df3["CAD"])
axs[1, 1].set_xticks(np.arange(0,210,25))
axs[1, 1].set_xlabel('Contextual Feature Size')
# axs[1, 1].set_title("N=500, #B=5", fontsize=8)

result_time_df3 = pd.read_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/BehaveResult2.csv", sep=',') 
axs[0, 2].plot(result_time_df3["BehaveFeatureSize"],result_time_df3["LOPAD"])
axs[0, 2].set_xticks(np.arange(0,150,15))
axs[0, 2].set_title("N=500, #C=5", fontsize=8)

axs[1, 2].plot(result_time_df3["BehaveFeatureSize"],result_time_df3["CAD"])
axs[1, 2].set_xticks(np.arange(0,150,15))
axs[1, 2].set_xlabel('Behavioural Feature Size')
# axs[1, 2].set_title("N=500, #C=5", fontsize=8)



