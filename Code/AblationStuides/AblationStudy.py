#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:55:27 2022

@author: zlifr

##color maps
https://matplotlib.org/stable/tutorials/colors/colormaps.html

##Abalation Study
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


###############################################################################
###############################################################################
##Step2 generate different dataset to use
from ContextualAnomalyInjectFinal import GenerateData

###############################################################################
##DataSet*:  Abalone to do
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/abalone.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/abaloneGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/abalone_ICAD.csv"

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
##DataSet*:  gasEmission to do
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/GasEmission.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/GasEmissionGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/GasEmission_ICAD.csv"


# AllCols = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO','NOX']

# AllColsWithTruth = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO','NOX',
#                     'ground_truth']

# ContextCols = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']

# BehaveCols = ['TEY', 'CO','NOX']

# NumCols =  ['TEY', 'CO','NOX']

# anomaly_value = 100

# sample_value = 100

# neighbour_value = 500

# num_dataset = 5

###############################################################################
##DataSet*:  Maintenance to do
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Maintenance.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/MaintenanceGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/Maintenance_ICAD.csv"

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
##DataSet1:  Energy
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/energy.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/energyGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/energy_ICADtest.csv"

# AllCols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']

# AllColsWithTruth = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2',
#                     'ground_truth']

# ContextCols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# BehaveCols = ['Y1', 'Y2']

# NumCols =  ['Y1', 'Y2']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 384

# num_dataset = 5


###############################################################################
##DataSet2:  ForestFires
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/forestFires.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/forestFiresGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/forestFires_ICADtest.csv"

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

# num_dataset = 5


###############################################################################
##DataSet3:  heartFailure
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/heartFailure.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/heartFailureGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/heartFailure_ICADtest.csv" 

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

# num_dataset = 5 

###############################################################################
##DataSet4:  hepatitis
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/hepatitis.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/hepatitisGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/hepatitis_ICADtest.csv"

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

# num_dataset = 5

###############################################################################
##DataSet5:  indianLiverPatient
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/indianLiverPatient.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/indianLiverPatientGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/indianLiverPatient_ICADtest.csv"

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

# num_dataset = 5


###############################################################################
##DataSet6:  QSRanking  
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/QSRanking.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/QSRankingGene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/QSRanking_ICADtest.csv"

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

# num_dataset = 5

###############################################################################
##DataSet7:  Syn5 
###############################################################################

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet5.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet5Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet5_ICADtest.csv"

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

# num_dataset = 5

###############################################################################
##DataSet8:  Syn10 
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet10.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet10Gene.csv"
# SaveFilePath_ICAD = r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet10_ICADtest.csv"

# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#             'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#             'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#             'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#             'behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                     'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                     'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                     'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3',
#                     'behave_num4',
#                     'ground_truth']


# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4', 'con_num5',
#                 'con_num6', 'con_num7', 'con_num8', 'con_num9', 'con_num10',
#                 'con_num11', 'con_num12', 'con_num13', 'con_num_cat0', 'con_num_cat1',
#                 'con_num_cat2', 'con_num_cat3', 'con_num_cat4', 'con_num_cat5']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 50

# sample_value = 50

# neighbour_value = 500

# num_dataset = 5

###############################################################################
###############################################################################
# =============================================================================
# #Step3 algorithm
# =============================================================================

import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import percentile
import numbers


import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skgarden import RandomForestQuantileRegressor

def ICAD_QRF(RawDataSet, MyColList, MyContextList, MyBehaveList, neighbour_value, anomaly_value, sample_value,
             AblationScaleValue = True,
             AblationClipValue = True,
             upper_bound_value = 100):
    
    ###############################################################################
    ###############################################################################
    ##Step1 set datasets

    MyDataSet = RawDataSet[MyColList]
    MyContextDataSet = MyDataSet[MyContextList]
    
    ###############################################################################
    ###############################################################################
    ##Step2 find  neighbors in contextual space for each point
    import gower
    from time import time
    t0 = time() # to record time 
    
    # calculate gower distance matrix
    MyContextDataSet = MyContextDataSet.astype(str)
    
    distance_matrix = gower.gower_matrix(MyContextDataSet)   
    
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(algorithm='brute', n_neighbors=neighbour_value, metric="precomputed").fit(distance_matrix)
    
    
    indices = knn.kneighbors(distance_matrix,return_distance=False)
    
    indices = pd.DataFrame(indices)
    
    indices['point_index'] = indices.index
    
    t1 = time()  # to record time 
    duration1 = round(t1 - t0, ndigits=4)
    print("matrix calculation processing time (s): ")
    print(duration1)

    ###############################################################################
    ###############################################################################
    ##Step3 calculate  anomaly  score in behavioral  space for each point
    
    def QRF_score(ReferenceDataSet, MyColList, MyContextList, MyBehaveList, step_width, new_point_index,
                  AblationScale = AblationScaleValue):
    
        MyDataSet = ReferenceDataSet[MyColList]
        MyContextDataSet = MyDataSet[MyContextList]
        MyBehaveDataSet = MyDataSet[MyBehaveList]
                
        def QuantilePerCol(X_train, X_test, y_train, y_test, step_width, IsScale = AblationScale):
            
            rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=10, n_estimators=10)
            
            
            rfqr.set_params(max_features= X_train.shape[1] // 1)
            rfqr.fit(X_train, y_train)
            
            quantile_0 = rfqr.predict(X_test, quantile=0.001)
            quantile_25 = rfqr.predict(X_test, quantile=25)
            quantile_75 = rfqr.predict(X_test, quantile=75)
            quantile_100 = rfqr.predict(X_test, quantile=100)
            
            quantile_vec = []
            quantile_location = -1
            scale_factor = (1000/step_width) ##to avoid too samll number in Python
            
            for quantile_num in np.arange(0,100/step_width,1):
                
                quantile_num = quantile_num*step_width                  
                quantile_left = rfqr.predict(X_test, quantile=quantile_num)
                quantile_right = rfqr.predict(X_test, quantile=min(100,quantile_num+step_width))
                quantile_vec.append(quantile_left[0])

                if quantile_left[0] <= y_test.iloc[0,0] and y_test.iloc[0,0] <= quantile_right[0]:
                    quantile_location = int(quantile_num/step_width)
                    
            quantile_vec_diff = np.diff(quantile_vec)
            quantile_vec_diff_rank = [sorted(quantile_vec_diff).index(x) for x in quantile_vec_diff]                 
            
            if (quantile_location == -1) and (y_test.iloc[0,0] < quantile_0):
                
                if IsScale == True:
                    normal_int_len = max(0.000000001,quantile_75-quantile_25)
                    result_temp = (1+abs(y_test.iloc[0,0]-quantile_0)/normal_int_len)*np.max(quantile_vec_diff)
                    quantile_diff =  result_temp[0]*scale_factor                
                    print("result_temp scaled")
                    print(result_temp[0])  
                else:
                    result_temp = np.max(quantile_vec_diff)
                    quantile_diff =  result_temp*scale_factor                
                    print("result_temp non scaled")
                    print(result_temp) 
               
                quantile_rank = 120
                print("A")
                return quantile_location, quantile_diff, quantile_rank
                
            elif (quantile_location == -1) and (y_test.iloc[0,0] > quantile_100):

                if IsScale == True:
                    normal_int_len = max(0.000000001,quantile_75-quantile_25)
                    result_temp = (1+abs(y_test.iloc[0,0]-quantile_100)/normal_int_len)*np.max(quantile_vec_diff)
                    print("result_temp scaled")
                    print(result_temp[0])
                    quantile_diff =  result_temp[0]*scale_factor
                else:                  
                    result_temp = np.max(quantile_vec_diff)
                    quantile_diff =  result_temp*scale_factor                
                    print("result_temp non scaled")
                    print(result_temp)                                 
                
                quantile_rank = 120
                print("B")
                return quantile_location, quantile_diff, quantile_rank
            
            quantile_diff = quantile_vec_diff[quantile_location-1]*scale_factor
            quantile_rank = quantile_vec_diff_rank[quantile_location-1]
            print("C")
            
            return  quantile_location, quantile_diff, quantile_rank
            
            
                
        X_train = MyContextDataSet[~MyContextDataSet.index.isin([new_point_index])]
        X_test = MyContextDataSet[MyContextDataSet.index.isin([new_point_index])]
        y_train_raw = MyBehaveDataSet[~MyBehaveDataSet.index.isin([new_point_index])]
        y_test_raw = MyBehaveDataSet[MyBehaveDataSet.index.isin([new_point_index])]
        
        quantile_bahave = []

        for behave_col in MyBehaveList:
            y_train = y_train_raw[[behave_col]]
            y_test = y_test_raw[[behave_col]]

            quantille_result, quantille_result_diff, quantille_result_diff_rank = QuantilePerCol(X_train, X_test, y_train, y_test, step_width)
           

            ##Three ways to define anomaly scores
            # quantile_score = quantille_result_rank
            quantile_score = quantille_result_diff
            # quantile_score = quantille_result_diff*quantille_result_diff_rank
        
            quantile_bahave.append(quantile_score)
        
        anomaly_behave = quantile_bahave.copy()
        
        return anomaly_behave
        
        
    t2 = time()  #to record time 
    duration2 = round(t2 - t1, ndigits=4)
    
    print("anomaly detection pre-processing time (s): ")
    print(duration2)
    
    all_raw_score_df = pd.DataFrame(columns=MyBehaveList)
    
    #execute our algorithm
    for point_index in range(MyDataSet.shape[0]):
        
        MyRefGroup = indices.loc[indices['point_index'] == point_index]
        
        #get indices of all neighbours which include itself
        MyRefGroup = MyRefGroup.values.tolist()
        MyRefGroup.append(point_index)
        MyRefGroup = set(MyRefGroup[0])
        MyRefGroup = list(MyRefGroup)
                    
        Filter_df  = MyDataSet[MyDataSet.index.isin(MyRefGroup)]    
                         
        new_point_index = point_index
        
        print("---------------")
        print(new_point_index)
        print("---------------")
        
        step_width = 1
        
        ##get raw quantiles for each behaviral column             
        quantile_anomaly_score = QRF_score(Filter_df, MyColList, MyContextList,
                                           MyBehaveList, step_width, new_point_index,
                                           )  

        #print(quantile_anomaly_score)
        all_raw_score_df.loc[len(all_raw_score_df)] = quantile_anomaly_score
        
    
    final_score_df = all_raw_score_df.add_suffix('_score')
           
    if AblationClipValue == True:
        ##Handling raw anomaly scores using an upper pruned sum to avoid dictator effect
        print("Clipped")
        upper_end = upper_bound_value #100 by default
        if len(MyBehaveList) > 1:
            final_score_df = final_score_df.clip(upper=upper_end)
        
    else:
        print("Unclipped")
        
    ##calculate weight vector for goodness of fit
    weight_sum_vec = list()
    for col_name in final_score_df.columns:
        
        weight_to_add =0
        right_end = max(final_score_df[col_name])
        # for bin_index in range(0,100,1):
        for bin_index in list(np.linspace(0,right_end,101)):
            number_in_bin = final_score_df[col_name][(final_score_df[col_name]>bin_index) & (final_score_df[col_name]<bin_index+1)].count()
            weight_to_add += number_in_bin*(bin_index+1)
            
        weight_sum_vec.append(weight_to_add)
        
    # if AblationWeightValue == True:
    #     print("Weighted")
    #     ##This step is applicable when there are more than one behaviral features
    #     if len(MyBehaveList) > 1:
    
    #         ##Using the weighted sum of clipped anomaly scores as a weight      
    #         weight_sum_vec = [1+(max(weight_sum_vec)-x)/max(weight_sum_vec) for x in weight_sum_vec]
    #         print(weight_sum_vec)
    #         for col_index in range(len(final_score_df.columns)):
    #             weight_sum_col = weight_sum_vec[col_index]
    #             col_name = final_score_df.columns[col_index]
    #             final_score_df[col_name] = final_score_df[col_name]*weight_sum_col
            
    # else:
    #     print("Unweighted")
        
                        
    ##We call it weight_score because it is weighted using the ratio of points located between [0,100]
    final_score_df["weight_score"] =  final_score_df[final_score_df.columns[ : final_score_df.shape[1]]].mean(axis=1) ## the aggragation function skip "NaN" automatically
    final_score_df["weight_score"] =  final_score_df["weight_score"].fillna(0) ##it may occur that all behavioral features contain NaN
    
    
    all_raw_score_df = all_raw_score_df.add_suffix('_quantile')
    
    
    from scipy import stats
    
    ##Using weight sum as the rwa anomaly score
    raw_anomaly_score = final_score_df["weight_score"]
    
    percentiles = [stats.percentileofscore(raw_anomaly_score, i) for i in raw_anomaly_score]
    
    MyDataSet["anomaly_score"] = percentiles
    MyDataSet["raw_anomaly_score"] = raw_anomaly_score       
    
    MyDataSet = MyDataSet.sort_values('anomaly_score')
    
    MyDataSet = pd.concat([MyDataSet, all_raw_score_df], axis=1)
    
    MyDataSet = pd.concat([MyDataSet, final_score_df], axis=1)
    
    ###############################################################################
    ###############################################################################
    ##Step4 evaluation of performance by using various metrics: PR AUC(AP), ROC AUC, P@n, F-score
    
    #evaluate anomaly score 1: roc auc score 
    from sklearn.metrics import roc_auc_score
    
    my_roc_score = roc_auc_score(MyDataSet["ground_truth"], MyDataSet["anomaly_score"])
    
    #evaluate anomaly score 2: Precision@n score, Recall@n score and F@n score
    TempDataSet = MyDataSet[["ground_truth","anomaly_score"]]
    
    P_TempDataSet = TempDataSet.sort_values(by=['anomaly_score'], ascending=[False]).head(sample_value)
    
    TP_value = (P_TempDataSet["ground_truth"]== 1).sum()
    
    P_at_n_value = TP_value/sample_value
        
    #evaluate anomaly score 3:prc auc score 
    from sklearn.metrics import precision_recall_curve, auc, roc_curve
    
    y = np.array(MyDataSet["ground_truth"])
    pred = np.array(MyDataSet["anomaly_score"])
    
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1) #to calculate roc auc
    precision, recall, thresholds = precision_recall_curve(y, pred, pos_label=1) #to calculate pr auc
    
    my_pr_auc = auc(recall, precision) #pr auc
    #my_roc_auc = auc(fpr, tpr) #roc auc
        
    t3 = time()  #to record time 
    duration3 = round(t3 - t2, ndigits=4)
    
    print("anomaly detection processing time (s): ")
    print(duration3)
    
    return my_pr_auc,my_roc_score,P_at_n_value,duration1, duration3, MyDataSet

###############################################################################
###############################################################################

# =============================================================================
# #4.1 def a function to compare difference
# =============================================================================
def AblationStudy(FilePath,
                ResultFilePath, 
                SaveFilePath_ICAD,
                MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                NumCols, anomaly_value, sample_value, neighbour_value,
                num_dataset):
    
    import pandas as pd
    
    ##Before
    WithAllPRC = []
    WithAllROC = []
    WithAllPN = []

    ##Ablation study for scale
    NoScalePRC = []
    NoScaleROC = []
    NoScalePN = []
                
    myResult_ICAD =  pd.DataFrame(columns = ["WithAllPRC","WithAllROC","WithAllPN",
                                             "NoScalePRC","NoScaleROC","NoScalePN"])
   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
           
        pr_b, roc_b, pn_b, duration1, duration3, TempDataSet = ICAD_QRF(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                        neighbour_value, anomaly_value, sample_value,
                                                                        AblationScaleValue = True,
                                                                        AblationClipValue = True,)
        WithAllPRC.append(pr_b)
        WithAllROC.append(roc_b)
        WithAllPN.append(pn_b)

        ##Ablation study for scale
        pr_a, roc_a, pn_a, duration1, duration3, TempDataSet = ICAD_QRF(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                    neighbour_value, anomaly_value, sample_value,
                                                                    AblationScaleValue = False,
                                                                    AblationClipValue = True)
        NoScalePRC.append(pr_a)
        NoScaleROC.append(roc_a)
        NoScalePN.append(pn_a)
                    
    
    ## For ICAD
    print("ICAD start")
    myResult_ICAD["WithAllPRC"] = WithAllPRC
    myResult_ICAD["WithAllROC"] = WithAllROC
    myResult_ICAD["WithAllPN"] = WithAllPN
    
    myResult_ICAD["NoScalePRC"] = NoScalePRC
    myResult_ICAD["NoScaleROC"] = NoScaleROC
    myResult_ICAD["NoScalePN"] = NoScalePN
    

    myResult_ICAD.to_csv(SaveFilePath_ICAD, sep=',')
    
  
###############################################################################     
##4.1.1 execute the function
###############################################################################
# AblationStudy(FilePath,
#               ResultFilePath,
#               SaveFilePath_ICAD,
#               AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
#               anomaly_value, sample_value, neighbour_value, num_dataset)


###############################################################################
##4.1.2 Display results
###############################################################################
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# sns.set(font_scale=0.6)

# f, axes = plt.subplots(1, 8)

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/Energy_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s1 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True, yticklabels=True,
#                   vmin=-0.5, vmax=0.5, ax=axes[0],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# s1.set_ylabel('num_data', fontsize = 8)
# s1.set_title(label = 'Energy', fontsize=8)

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/forestFires_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s2 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True, yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[1],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# s2.set_title(label = 'Forest Fires', fontsize=8)



# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/hepatitis_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s3 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True, yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[2],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# s3.set_title(label = 'Hepatitis', fontsize=8)


# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/heartFailure_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s4 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True,yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[3],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# s4.set_title(label = 'Heart Failure', fontsize=8)


# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/indianLiverPatient_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s5 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True,yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[4],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# # s5.set_ylabel('num_data', fontsize = 8)
# s5.set_title(label = 'Indian Liver Patient', fontsize=8)


# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/QSRanking_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s6 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True,yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[5],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# s6.set_title(label = 'QS ranking', fontsize=8)

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet5_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s7 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True,yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[6],
#                   annot_kws={"fontsize":8}, cbar=False) ##RdYlGn, 
# s7.set_title(label = 'Synthetic 5', fontsize=8)


# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet10_ICAD.csv",sep=",")
# RawDataSet["num_data"] = RawDataSet.index
# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 
# s8 = sns.heatmap(New_df[["PRC Diff","ROC Diff","P@n Diff"]], annot=True, fmt=".2f", cmap = "RdYlGn",
#                   xticklabels=True,yticklabels=False,
#                   vmin=-0.5, vmax=0.5, ax=axes[7],
#                   annot_kws={"fontsize":8}, cbar=True) ##RdYlGn, 
# s8.set_title(label = 'Synthetic 10', fontsize=8)
# # s8.set_xticklabels(s8.get_xmajorticklabels(), fontsize = 5)


# =============================================================================
#  Other choice such as barplot
# =============================================================================
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# f, axes = plt.subplots(3, 1)

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/indianLiverPatient_ICAD.csv",sep=",")

# RawDataSet["num_data"] = RawDataSet.index


# New_df = pd.DataFrame(columns=["PRC AUC","ROC AUC","Precision@n"])
# New_df["PRC Diff"] = RawDataSet["WithAllPRC"] - RawDataSet["NoScalePRC"]
# New_df["ROC Diff"] = RawDataSet["WithAllROC"] - RawDataSet["NoScaleROC"]
# New_df["P@n Diff"] = RawDataSet["WithAllPN"] - RawDataSet["NoScalePN"]
# New_df["num_data"]  = RawDataSet["num_data"] 


# ax1 = sns.barplot(data=New_df, x="num_data", y="PRC Diff", ax=axes[0])
# ax1.title.set_text('Indian Liver Patient')
# # for i in ax1.containers:
# #     ax1.bar_label(i,)
    
# ax2 = sns.barplot(data=New_df, x="num_data", y="ROC Diff", ax=axes[1])
# ax3 = sns.barplot(data=New_df, x="num_data", y="P@n Diff", ax=axes[2])


# f, axes = plt.subplots(3, 1)
# ax1 = sns.barplot(data=New_df, x="num_data", y="PRC Diff", ax=axes[0])
# ax1.title.set_text('QSRanking')
# ax1.axhline(0, color="black", clip_on=True)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# # for i in ax1.containers:
# #     ax1.bar_label(i,)
# ax2 = sns.barplot(data=New_df, x="num_data", y="ROC Diff", ax=axes[1])
# ax2.axhline(0, color="black", clip_on=True)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# ax3 = sns.barplot(data=New_df, x="num_data", y="P@n Diff", ax=axes[2])
# ax3.axhline(0, color="black", clip_on=True)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)


###############################################################################
###############################################################################
# =============================================================================
# #4.2 def a function to test value of clip upper bound
# =============================================================================
def AblationStudy2(FilePath,
                ResultFilePath, 
                SaveFilePath_ICAD,
                MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                NumCols, anomaly_value, sample_value, neighbour_value,
                num_dataset):
    
    import pandas as pd
    
    ##define vectors to store results
    AblationPR = []
    AblationROC = []
    AblationPN  = []
    AblationDS = []
    AblationBD = []
 
    myResult_ICAD =  pd.DataFrame(columns = ["PR","ROC","PN","RandomSate"])
   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
                        
        pr_b, roc_b, pn_b, duration1, duration3, TempDataSet = ICAD_QRF(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                neighbour_value, anomaly_value, sample_value,
                                                                AblationClipValue = False)
        
        AblationPR.append(pr_b)
        AblationROC.append(roc_b)
        AblationPN.append(pn_b)
        AblationDS.append(MyRandomState) 
        AblationBD.append("None")
             

        ##test the same dataset by using different upper_bound                                              
        # for upper_end_term in [1,5,10,20,40,60,80,100,120,140,160,180,200,300,400]:
        for upper_end_term in [1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,150,200,300,400]:
            
            pr_b, roc_b, pn_b, duration1, duration3, TempDataSet = ICAD_QRF(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                            neighbour_value, anomaly_value, sample_value,
                                                                            AblationClipValue = True,                                                                            
                                                                            upper_bound_value = upper_end_term)
            AblationPR.append(pr_b)
            AblationROC.append(roc_b)
            AblationPN.append(pn_b)
            AblationDS.append(MyRandomState)
            AblationBD.append(upper_end_term)
        
    myResult_ICAD["PR"] = AblationPR
    myResult_ICAD["ROC"] = AblationROC
    myResult_ICAD["PN"] = AblationPN
    myResult_ICAD["RandomState"] = AblationDS
    myResult_ICAD["upper_bound"] = AblationBD

    myResult_ICAD.to_csv(SaveFilePath_ICAD, sep=',')

###############################################################################     
##4.2.1 execute the function
###############################################################################
# AblationStudy2(FilePath,
#               ResultFilePath,
#               SaveFilePath_ICAD,
#               AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
#               anomaly_value, sample_value, neighbour_value, num_dataset)

###############################################################################     
##4.2.2 Display the results
###############################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# # =============================================================================
# # Dataset1
# # =============================================================================
# f, axes = plt.subplots(3, 4)

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/energy_ICADtest.csv",sep=",")

# RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
#                                         'ROC': 'ROC AUC',
#                                         'PN': "Precision@n",
#                                         'RandomState': 'RandomState'})

# RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
#                                                             44: "Data2", 45: "Data3",
#                                                             46: "Data4"})

# RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
# RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
# RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)

# ax1 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="p", ax=axes[0,0])

# ax1.plot(["3", "3"], [0, 1], color="r")
# ax1.plot(["10", "10"], [0, 1], color="r")

# ax1.title.set_text('Energy')
# ax1.get_legend().remove()
# # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax1.xaxis.label.set_visible(False)
# # ax1.axes.xaxis.set_ticklabels([])

# ax2 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="s", ax=axes[1,0])
# ax2.get_legend().remove()
# # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax2.xaxis.label.set_visible(False)
# # ax2.set_xticks([])

# ax2.plot(["3", "3"], [0, 1], color="r")
# ax2.plot(["10", "10"], [0, 1], color="r")


# ax3 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="d", ax=axes[2,0])
# ax3.get_legend().remove()
# # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax3.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax3.plot(["3", "3"], [0, 1], color="r")
# ax3.plot(["10", "10"], [0, 1], color="r")


# # =============================================================================
# # Dataset2
# # =============================================================================

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/heartFailure_ICADtest.csv",sep=",")

# RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
#                                         'ROC': 'ROC AUC',
#                                         'PN': "Precision@n",
#                                         'RandomState': 'RandomState'})

# RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
#                                                             44: "Data2", 45: "Data3",
#                                                             46: "Data4"})

# RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
# RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
# RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)


# ax4 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="p", ax=axes[0,1])
# ax4.title.set_text('Heart Failure')
# ax4.get_legend().remove()
# # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax4.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)
# ax4.xaxis.label.set_visible(False)
# ax4.yaxis.label.set_visible(False)
# ax4.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# # ax4.set_xticks([])
# ax4.plot(["3", "3"], [0, 1], color="r")
# ax4.plot(["10", "10"], [0, 1], color="r")


# ax5 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="s", ax=axes[1,1])
# ax5.get_legend().remove()
# # ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax5.spines['right'].set_visible(False)
# ax5.spines['top'].set_visible(False)
# ax5.xaxis.label.set_visible(False)
# ax5.yaxis.label.set_visible(False)
# ax5.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# # ax5.set_xticks([])
# ax5.plot(["3", "3"], [0, 1], color="r")
# ax5.plot(["10", "10"], [0, 1], color="r")


# ax6 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="d", ax=axes[2,1])
# ax6.get_legend().remove()
# # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax6.spines['right'].set_visible(False)
# ax6.spines['top'].set_visible(False)
# ax6.yaxis.label.set_visible(False)
# ax6.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax6.plot(["3", "3"], [0, 1], color="r")
# ax6.plot(["10", "10"], [0, 1], color="r")



# # =============================================================================
# # Dataset3
# # =============================================================================

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/indianLiverPatient_ICADtest.csv",sep=",")

# RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
#                                         'ROC': 'ROC AUC',
#                                         'PN': "Precision@n",
#                                         'RandomState': 'RandomState'})

# RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
#                                                             44: "Data2", 45: "Data3",
#                                                             46: "Data4"})

# RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
# RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
# RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)

# ax7 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="p", ax=axes[0,2])
# ax7.title.set_text('Indian Liver Patient')
# ax7.get_legend().remove()
# # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax7.spines['right'].set_visible(False)
# ax7.spines['top'].set_visible(False)
# ax7.xaxis.label.set_visible(False)
# ax7.yaxis.label.set_visible(False)
# ax7.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# # ax4.set_xticks([])
# ax7.plot(["3", "3"], [0, 1], color="r")
# ax7.plot(["10", "10"], [0, 1], color="r")



# ax8 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="s", ax=axes[1,2])
# ax8.get_legend().remove()
# # ax8.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax8.spines['right'].set_visible(False)
# ax8.spines['top'].set_visible(False)
# ax8.xaxis.label.set_visible(False)
# ax8.yaxis.label.set_visible(False)
# ax8.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# # ax5.set_xticks([])
# ax8.plot(["3", "3"], [0, 1], color="r")
# ax8.plot(["10", "10"], [0, 1], color="r")


# ax9 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="d", ax=axes[2,2])
# ax9.get_legend().remove()
# # ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax9.spines['right'].set_visible(False)
# ax9.spines['top'].set_visible(False)
# ax9.yaxis.label.set_visible(False)
# ax9.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax9.plot(["3", "3"], [0, 1], color="r")
# ax9.plot(["10", "10"], [0, 1], color="r")


# # =============================================================================
# # Dataset4
# # =============================================================================

# RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/forestFires_ICADtest.csv",sep=",")

# RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
#                                         'ROC': 'ROC AUC',
#                                         'PN': "Precision@n",
#                                         'RandomState': 'RandomState'})

# RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
#                                                             44: "Data2", 45: "Data3",
#                                                             46: "Data4"})

# RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
# RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
# RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)

# ax10 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="p", ax=axes[0,3])
# ax10.title.set_text('Forest Fires')
# ax10.get_legend().remove()
# # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax10.spines['right'].set_visible(False)
# ax10.spines['top'].set_visible(False)
# ax10.xaxis.label.set_visible(False)
# ax10.yaxis.label.set_visible(False)
# ax10.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# # ax4.set_xticks([])
# ax10.plot(["3", "3"], [0, 1], color="r")
# ax10.plot(["10", "10"], [0, 1], color="r")



# ax11 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="s", ax=axes[1,3])
# ax11.get_legend().remove()
# ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax11.spines['right'].set_visible(False)
# ax11.spines['top'].set_visible(False)
# ax11.xaxis.label.set_visible(False)
# ax11.yaxis.label.set_visible(False)
# ax11.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# # ax5.set_xticks([])
# ax11.plot(["3", "3"], [0, 1], color="r")
# ax11.plot(["10", "10"], [0, 1], color="r")


# ax12 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
#                     hue="RandomState", style="RandomState", marker="d", ax=axes[2,3])
# ax12.get_legend().remove()
# # ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# ax12.spines['right'].set_visible(False)
# ax12.spines['top'].set_visible(False)
# ax12.yaxis.label.set_visible(False)
# ax12.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax12.plot(["3", "3"], [0, 1], color="r")
# ax12.plot(["10", "10"], [0, 1], color="r")

# =============================================================================
# Dataset1
# =============================================================================
f, axes = plt.subplots(3, 4)

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/hepatitis_ICADtest.csv",sep=",")

RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
                                        'ROC': 'ROC AUC',
                                        'PN': "Precision@n",
                                        'RandomState': 'RandomState'})

RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
                                                            44: "Data2", 45: "Data3",
                                                            46: "Data4"})

RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)

ax1 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="p", ax=axes[0,0])

ax1.plot(["3", "3"], [0, 1], color="r")
ax1.plot(["10", "10"], [0, 1], color="r")

ax1.title.set_text('Hepatitis')
ax1.get_legend().remove()
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
ax1.xaxis.label.set_visible(False)
# ax1.axes.xaxis.set_ticklabels([])

ax2 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="s", ax=axes[1,0])
ax2.get_legend().remove()
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
ax2.xaxis.label.set_visible(False)
# ax2.set_xticks([])

ax2.plot(["3", "3"], [0, 1], color="r")
ax2.plot(["10", "10"], [0, 1], color="r")


ax3 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
                    hue="RandomState", style="RandomState", marker="d", ax=axes[2,0])
ax3.get_legend().remove()
# ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
ax3.plot(["3", "3"], [0, 1], color="r")
ax3.plot(["10", "10"], [0, 1], color="r")


# =============================================================================
# Dataset2
# =============================================================================

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/QSRanking_ICADtest.csv",sep=",")

RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
                                        'ROC': 'ROC AUC',
                                        'PN': "Precision@n",
                                        'RandomState': 'RandomState'})

RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
                                                            44: "Data2", 45: "Data3",
                                                            46: "Data4"})

RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)


ax4 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="p", ax=axes[0,1])
ax4.title.set_text('QS Ranking')
ax4.get_legend().remove()
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.xaxis.label.set_visible(False)
ax4.yaxis.label.set_visible(False)
ax4.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax4.set_xticks([])
ax4.plot(["3", "3"], [0, 1], color="r")
ax4.plot(["10", "10"], [0, 1], color="r")


ax5 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="s", ax=axes[1,1])
ax5.get_legend().remove()
# ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.xaxis.label.set_visible(False)
ax5.yaxis.label.set_visible(False)
ax5.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax5.set_xticks([])
ax5.plot(["3", "3"], [0, 1], color="r")
ax5.plot(["10", "10"], [0, 1], color="r")


ax6 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
                    hue="RandomState", style="RandomState", marker="d", ax=axes[2,1])
ax6.get_legend().remove()
# ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.yaxis.label.set_visible(False)
ax6.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
ax6.plot(["3", "3"], [0, 1], color="r")
ax6.plot(["10", "10"], [0, 1], color="r")



# =============================================================================
# Dataset3
# =============================================================================

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet5_ICADtest.csv",sep=",")

RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
                                        'ROC': 'ROC AUC',
                                        'PN': "Precision@n",
                                        'RandomState': 'RandomState'})

RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
                                                            44: "Data2", 45: "Data3",
                                                            46: "Data4"})

RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)

ax7 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="p", ax=axes[0,2])
ax7.title.set_text('Synthetic 5')
ax7.get_legend().remove()
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax7.spines['right'].set_visible(False)
ax7.spines['top'].set_visible(False)
ax7.xaxis.label.set_visible(False)
ax7.yaxis.label.set_visible(False)
ax7.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax4.set_xticks([])
ax7.plot(["3", "3"], [0, 1], color="r")
ax7.plot(["10", "10"], [0, 1], color="r")



ax8 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="s", ax=axes[1,2])
ax8.get_legend().remove()
# ax8.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)
ax8.xaxis.label.set_visible(False)
ax8.yaxis.label.set_visible(False)
ax8.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax5.set_xticks([])
ax8.plot(["3", "3"], [0, 1], color="r")
ax8.plot(["10", "10"], [0, 1], color="r")


ax9 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
                    hue="RandomState", style="RandomState", marker="d", ax=axes[2,2])
ax9.get_legend().remove()
# ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
ax9.yaxis.label.set_visible(False)
ax9.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
ax9.plot(["3", "3"], [0, 1], color="r")
ax9.plot(["10", "10"], [0, 1], color="r")


# =============================================================================
# Dataset4
# =============================================================================

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/AblationTempData/SynDataSet10_ICADtest.csv",sep=",")

RawDataSet = RawDataSet.rename(columns={'PR': 'PRC AUC',
                                        'ROC': 'ROC AUC',
                                        'PN': "Precision@n",
                                        'RandomState': 'RandomState'})

RawDataSet['RandomState'] = RawDataSet['RandomState'].map({42: "Data0", 43: "Data1",
                                                            44: "Data2", 45: "Data3",
                                                            46: "Data4"})

RawDataSet["upper_bound"] = ["None","0.1","0.2","0.5","1","2","3","4","5","6","7","8","9","10","11","12","15","20","30","40"]*5
RawDataSet["Order"] = [20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]*5
RawDataSet = RawDataSet.sort_values(by=["RandomState","Order"], ascending=True)

ax10 = sns.lineplot(data=RawDataSet, x="upper_bound", y="PRC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="p", ax=axes[0,3])
ax10.title.set_text('Synthetic 10')
ax10.get_legend().remove()
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax10.spines['right'].set_visible(False)
ax10.spines['top'].set_visible(False)
ax10.xaxis.label.set_visible(False)
ax10.yaxis.label.set_visible(False)
ax10.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax4.set_xticks([])
ax10.plot(["3", "3"], [0, 1], color="r")
ax10.plot(["10", "10"], [0, 1], color="r")



ax11 = sns.lineplot(data=RawDataSet, x="upper_bound", y="ROC AUC", palette = "hls",
                    hue="RandomState", style="RandomState", marker="s", ax=axes[1,3])
ax11.get_legend().remove()
ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)
ax11.xaxis.label.set_visible(False)
ax11.yaxis.label.set_visible(False)
ax11.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
# ax5.set_xticks([])
ax11.plot(["3", "3"], [0, 1], color="r")
ax11.plot(["10", "10"], [0, 1], color="r")


ax12 = sns.lineplot(data=RawDataSet, x="upper_bound", y="Precision@n", palette = "hls",
                    hue="RandomState", style="RandomState", marker="d", ax=axes[2,3])
ax12.get_legend().remove()
# ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
ax12.spines['right'].set_visible(False)
ax12.spines['top'].set_visible(False)
ax12.yaxis.label.set_visible(False)
ax12.set_xticklabels(RawDataSet["upper_bound"], fontsize=6, rotation=90)
ax12.plot(["3", "3"], [0, 1], color="r")
ax12.plot(["10", "10"], [0, 1], color="r")