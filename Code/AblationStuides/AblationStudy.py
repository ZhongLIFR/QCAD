#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:55:27 2022

@author: zlifr

##color maps
https://matplotlib.org/stable/tutorials/colors/colormaps.html

##Abalation Study
"""

##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub'  

# =============================================================================
# #Step1 Import modules
# =============================================================================


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


import sys
UtilityDir = AbsRootDir+ r'/QCAD/Code/Utilities'
sys.path.append(UtilityDir)

from ContextualAnomalyInject import GenerateData


# =============================================================================
# #Step2 Modify QCAD to perform ablation study 
# =============================================================================



import six
import sys
sys.modules['sklearn.externals.six'] = six

from skgarden import RandomForestQuantileRegressor

def QCAD(RawDataSet, MyColList, MyContextList, MyBehaveList, neighbour_value, sample_value,
         AblationScaleValue = True,
         AblationClipValue = True,
         upper_bound_value = 100):
    """
    

    Parameters
    ----------
    RawDataSet : dataframe
        dataframe containing raw dataset after preprocessing.
    MyColList : list
        the list of all feature names.
    MyContextList : list
        the list of contextual feature names.
    MyBehaveList : list
       the list of behavioural feature names.
    neighbour_value : int
        the number of neighbours.
    sample_value : int
        the number of anomalies.
    AblationScaleValue : True or False, optional
        indicate whether test the scaling module. The default is True.
    AblationClipValue : True or False, optional
        indicate whether test the clipping module. The default is True.
    upper_bound_value : int, optional
        clipping upper bound threshold by default. The default is 100.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
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


# =============================================================================
# #Step3 Define a function to perform ablation study on scaling
# =============================================================================
def AblationStudy(FilePath,
                ResultFilePath, 
                SaveFilePath_QCAD,
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
                
    myResult_QCAD =  pd.DataFrame(columns = ["WithAllPRC","WithAllROC","WithAllPN",
                                             "NoScalePRC","NoScaleROC","NoScalePN"])
   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
           
        pr_b, roc_b, pn_b, duration1, duration3, TempDataSet = QCAD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                    neighbour_value, sample_value,
                                                                    AblationScaleValue = True,
                                                                    AblationClipValue = True,)
        WithAllPRC.append(pr_b)
        WithAllROC.append(roc_b)
        WithAllPN.append(pn_b)

        ##Ablation study for scale
        pr_a, roc_a, pn_a, duration1, duration3, TempDataSet = QCAD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                    neighbour_value, sample_value,
                                                                    AblationScaleValue = False,
                                                                    AblationClipValue = True)
        NoScalePRC.append(pr_a)
        NoScaleROC.append(roc_a)
        NoScalePN.append(pn_a)
                    
    
    ## For ICAD
    print("QCAD start")
    myResult_QCAD["WithAllPRC"] = WithAllPRC
    myResult_QCAD["WithAllROC"] = WithAllROC
    myResult_QCAD["WithAllPN"] = WithAllPN
    
    myResult_QCAD["NoScalePRC"] = NoScalePRC
    myResult_QCAD["NoScaleROC"] = NoScaleROC
    myResult_QCAD["NoScalePN"] = NoScalePN
    

    myResult_QCAD.to_csv(SaveFilePath_QCAD, sep=',')
    


# =============================================================================
# #STep4 Define another function to perform ablation study on values of clipping upper bound
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
                        
        pr_b, roc_b, pn_b, duration1, duration3, TempDataSet = QCAD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
                                                                    neighbour_value, sample_value,
                                                                    AblationClipValue = False)
        
        AblationPR.append(pr_b)
        AblationROC.append(roc_b)
        AblationPN.append(pn_b)
        AblationDS.append(MyRandomState) 
        AblationBD.append("None")
             

        ##test the same dataset by using different upper_bound                                              
        for upper_end_term in [1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,150,200,300,400]:
            
            pr_b, roc_b, pn_b, duration1, duration3, TempDataSet = QCAD(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, 
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


# =============================================================================
# ##Step4 Example -> Energy, Uncomment the following code 
# =============================================================================
    

# FilePath =  AbsRootDir+r'/QCAD/Data/RawData/Energy.csv'

# ResultFilePath = AbsRootDir+r'/QCAD/Data/TempFiles/EnergyAblationGene.csv'

# SaveFilePath_QCAD = AbsRootDir+r'/QCAD/Data/TempFiles/EnergyAblation1.csv'

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


# AblationStudy(FilePath,
#               ResultFilePath,
#               SaveFilePath_QCAD,
#               AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
#               anomaly_value, sample_value, neighbour_value, num_dataset)


# SaveFilePath_QCAD = AbsRootDir+r'/QCAD/Data/TempFiles/EnergyAblation2.csv'

# AblationStudy2(FilePath,
#               ResultFilePath,
#               SaveFilePath_QCAD,
#               AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
#               anomaly_value, sample_value, neighbour_value, num_dataset)




