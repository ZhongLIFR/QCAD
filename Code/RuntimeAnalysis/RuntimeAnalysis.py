#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:18:05 2022

@author: zlifr
"""
##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub'  


import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# #Step1 import basic modules
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
from COD import COD


# =============================================================================
# ##Step3 Define a funtion to compare running time
# =============================================================================

##def a function to do running time analysis
def RuntimeAnalysis(FilePath,
                    ResultFilePath, 
                    MB_dataset_path,
                    MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                    NumCols, anomaly_value, sample_value, neighbour_value,
                    num_dataset):
    """

    Parameters
    ----------
    FilePath : string
        the path of raw dataset.
    ResultFilePath : string
        the path of resulting dataset.
    MB_dataset_path : string
        the path of MB used in LoPAD.
    MyColList : list
        the list of all feature names.
    AllColsWithTruth : list
        the list of all feature names and ground-truth.
    MyContextList : list
        the list of contextual feature names.
    MyBehaveList : list
       the list of behavioural feature names.
    NumCols : list
        the list of numerical contextual feature names.
    anomaly_value : int
        the number of anomalies.
    sample_value : int
        the number of anomalies.
    neighbour_value : int
        the number of neighbours.
    num_dataset : int
        the number of trials.

    Returns
    -------
    mean_time_list : TYPE
        DESCRIPTION.

    """
    
    import pandas as pd
    from time import time
    
    result_time_df = pd.DataFrame(columns=["QCAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])
   
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
        QCAD_time_start = time()              
        QCAD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
             neighbour_value, anomaly_value, sample_value)
        
        QCAD_time_end = time()
        dur_QCAD = round(QCAD_time_end - QCAD_time_start, ndigits=4)
        time_vec.append(dur_QCAD)
        
        
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
        FilePath_MappingMatrix = AbsRootDir + r'/QCAD/Data/TempFiles/temp_CAD_mapping_matrix_RT.npy'
        COD(MyDataSet, MyContextList, MyBehaveList,
            5, 0.01, 0, FilePath_MappingMatrix)        
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
        time_new_df = pd.DataFrame([time_vec], columns=["QCAD", "LOPAD", "ROCOD", "CAD", "IForest", "LOF", "KNN", "SOD", "HBOS"])

        result_time_df = pd.concat([result_time_df, time_new_df])
        
        
    mean_time_list = list(result_time_df.mean(axis=0))

    return mean_time_list
    

# =============================================================================
# ##Step4 Example -> it may take several weeks to accomplish the execution 
#                    by varying the number of contextual features, behavioural features
#                    or the number of samples, respectively.
# =============================================================================







