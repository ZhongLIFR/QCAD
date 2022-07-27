#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:50:33 2022

@author: zlifr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:36:39 2021

@author: zlifr
"""
import warnings
warnings.filterwarnings("ignore")

###############################################################################
###############################################################################
##Step1 load dataset and description analysis

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
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skgarden import RandomForestQuantileRegressor


# =============================================================================
# # Step 1: Load dataset
# =============================================================================

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/forestFires.csv", sep=",")


# MyColList = ['X', 'Y', 'month', 'day', 
#               'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# MyContextList = ['X', 'Y', 'month', 'day']

# MyBehaveList = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# neighbour_value = 260


RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/IRIS.csv", sep=",")

##Encode string to number 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col_index in range(RawDataSet.shape[1]):
    col_name = RawDataSet.columns.tolist()[col_index]
    RawDataSet[col_name] = label_encoder.fit_transform(RawDataSet[col_name])


MyColList = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
MyContextList = ['sepal_length', 'sepal_width', 'species']

# MyColList = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# MyContextList = ['sepal_length', 'sepal_width']

MyBehaveList = ['petal_length', 'petal_width']

neighbour_value = 50



# =============================================================================
# # Step 2: Define function
# =============================================================================


def ICAD_QRF(RawDataSet,
             MyColList,
             MyContextList,
             MyBehaveList,
             neighbour_value,
             AblationScale = True,
             AblationClip = True,
             AblationWeight = True):
    
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
                  AblationScale = True,
                  AblationClip = True,
                  AblationWeight = True):
    
        MyDataSet = ReferenceDataSet[MyColList]
        MyContextDataSet = MyDataSet[MyContextList]
        MyBehaveDataSet = MyDataSet[MyBehaveList]
                
        def QuantilePerCol(X_train, X_test, y_train, y_test, step_width,
                           AblationScale = True):
            
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
                
                if AblationScale == True:                 
                    result_temp = (1+abs(y_test.iloc[0,0]-quantile_0)/(quantile_75-quantile_25))*np.max(quantile_vec_diff)
                else:
                    result_temp = np.max(quantile_vec_diff)
                
                quantile_diff =  result_temp[0]*scale_factor
                quantile_rank = 120
                print("A")
                return quantile_location, quantile_diff, quantile_rank
                
            elif (quantile_location == -1) and (y_test.iloc[0,0] > quantile_100):
                if AblationScale == True:  
                    normal_int_len = max(0.000000001,quantile_75-quantile_25)
                    result_temp = (1+abs(y_test.iloc[0,0]-quantile_100)/normal_int_len)*np.max(quantile_vec_diff)
                else:
                    result_temp = np.max(quantile_vec_diff)
                    
                quantile_diff =  result_temp[0]*scale_factor
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
        quantile_anomaly_score = QRF_score(Filter_df, MyColList, MyContextList, MyBehaveList, step_width, new_point_index)  

        #print(quantile_anomaly_score)
        all_raw_score_df.loc[len(all_raw_score_df)] = quantile_anomaly_score
        
    
    final_score_df = all_raw_score_df.add_suffix('_score')
           
    if AblationClip == True:
        ##Handling raw anomaly scores using an upper pruned sum to avoid dictator effect
        upper_end = 100
        if len(MyBehaveList) > 1:
            final_score_df = final_score_df.clip(upper=upper_end)
        
        
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
        
    # if AblationWeight == True:
    #     ##This step is applicable when there are more than one behaviral features
    #     if len(MyBehaveList) > 1:
    
    #         ##Using the weighted sum of clipped anomaly scores as a weight      
    #         weight_sum_vec = [1+(max(weight_sum_vec)-x)/max(weight_sum_vec) for x in weight_sum_vec]
    #         print(weight_sum_vec)
    #         for col_index in range(len(final_score_df.columns)):
    #             weight_sum_col = weight_sum_vec[col_index]
    #             col_name = final_score_df.columns[col_index]
    #             final_score_df[col_name] = final_score_df[col_name]*weight_sum_col
            
                        
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
    
    return MyDataSet, distance_matrix

# =============================================================================
# # Step 3: execute function
# =============================================================================
MyDataSet, GowerDistanceMatrix = ICAD_QRF(RawDataSet, MyColList, MyContextList, MyBehaveList, neighbour_value)

GDM_df = pd.DataFrame(GowerDistanceMatrix)


# =============================================================================
# # Step 4: explain results
# =============================================================================

# ====================================================
# #Step 4.1 provide explanation for reported values
# ====================================================


from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(algorithm='brute', n_neighbors=neighbour_value, metric="precomputed").fit(GowerDistanceMatrix)

indices = knn.kneighbors(GowerDistanceMatrix,return_distance=False)

indices = pd.DataFrame(indices)

indices['point_index'] = indices.index

MyDataSet['point_index'] = MyDataSet.index

neighbours_selected = indices.loc[indices['point_index'] == 70]

neighbours_selected = list(neighbours_selected.iloc[0].values)

##Now we add the species part
RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/IRIS.csv", sep=",")

ContextDataset = RawDataSet[['sepal_length', 'sepal_width', 'species']]
NeighboursDataset  = ContextDataset[ContextDataset.index.isin(neighbours_selected)]
NeighboursDataset["label"] = "Neighbours"


NeighboursDataset.loc[[70],"label"] = "Target"


RestDataset  = ContextDataset[~ContextDataset.index.isin(neighbours_selected)]

RestDataset["label"] = "Non Neighbours"

AllDatasSet = pd.concat([NeighboursDataset, RestDataset])

import seaborn as sns

ax1 = sns.scatterplot(data=AllDatasSet, x="sepal_length", y="sepal_width", hue="species", palette = ["r","g","b"],
                style="label",  size='label', sizes={"Neighbours":50,"Target":100,"Non Neighbours":40}, markers=['s', '$\clubsuit$','d'])

ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})




MyDataSet['label'] = np.where(MyDataSet['weight_score'] > 90, "Abnormal", "Normal")

ax2 = sns.scatterplot(data=MyDataSet, x="petal_length", y="petal_width", hue="species",
                      palette = ["r","g","b"], style="label")






# ====================================================
# ##Step 4.2.2 demonstrate its raw anomaly score
# ====================================================

# AnomalyScoreDataSet = MyDataSet.loc[68,['petal_length_quantile', 'petal_width_quantile']]

# AnomalyScoreDataSet = AnomalyScoreDataSet.to_frame()
# AnomalyScoreDataSet = AnomalyScoreDataSet.T
# AnomalyScoreDataSet.plot.bar(rot=0)
# plt.title("Raw matched conditional quantile of object 68 on each atrribute")

# MyDataSet['petal_length_score'].hist(bins=50)
# # plt.text(-152, 5, "location: -152.02",rotation=90)
# plt.title('Histogram of matched conditional quantiles of all objects on FFMC')

# MyDataSet['petal_width_score'].hist(bins=50)
# # plt.text(-125, 5, "location: -125.00",rotation=90)
# plt.title('Histogram of matched conditional quantiles of all objects on ISI')



