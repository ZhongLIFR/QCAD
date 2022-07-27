
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
test_range = range(5,200,5)

for con_feature_size in test_range:
    
    num_con_value = con_feature_size
    num_con_cat_value = int(con_feature_size/4)
    
    # num_con_value = 3
    # num_con_cat_value = 1
    
    num_behave_value = 2
            
    sample_size_value = 500
    
    MyDataSet = GenSynDataset(num_con =num_con_value , num_con_cat = num_con_cat_value, num_behave = num_behave_value,
                              sample_size = sample_size_value, num_gaussian = 5, my_scheme = "S1")
    MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSet1.csv", sep=',')
    
        
    FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSet1.csv"
    ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSetGene1.csv"
    MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSetMB1.csv"
    
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
    
    # #Generate MB manually
    total_cols = num_con_value + num_behave_value
    
    col_vec = ["con_num0"]*total_cols
        
    MB_df = pd.DataFrame(col_vec, columns=["a"])
    
    MB_df.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/SynDataSetMB1.csv", sep=',', header=False, index=False)  
    
    #execute the function
    result_time_list = RuntimeSampleSize(FilePath,
                                       ResultFilePath,
                                       MB_dataset_path,
                                       AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
                                       anomaly_value, sample_value, neighbour_value, num_dataset)
        
    new_df = pd.DataFrame([result_time_list], columns=["RICAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])
    result_time_df = pd.concat([result_time_df, new_df])

result_time_df["ContextFeatureSize"] = list(test_range)
result_time_df.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Runtime/ContextResult.csv", sep=',') 
 
##to display results
import seaborn as sns
result_time_df_final = result_time_df.melt('ContextFeatureSize', var_name='cols', value_name='vals')
g = sns.catplot(x="ContextFeatureSize", y="vals", hue='cols', style = "cols", data=result_time_df_final, kind='point')

g1 = sns.catplot(x="ContextFeatureSize", y="CAD", data=result_time_df, kind='point') #LoPAD and RICAD are exchanged 


