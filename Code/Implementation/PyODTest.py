#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:39:44 2021

@author: zlifr
"""

# =============================================================================
# Step I. call off-the-shelf models that are implemented in PyOD
# =============================================================================

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


def PyODModel(RawDataSet, sample_value, ADetector):
    
    # X_train =  RawDataSet.iloc[:, 1:-2] ##Not for average
    X_train =  RawDataSet.iloc[:, 0:-1]
    ##training
    # clf = IForest()
    # clf = LOF()
    # clf = KNN()
    # clf = SOD()
    # clf = HBOS()
    clf = ADetector
    
    MyDataSet = RawDataSet.copy()
    clf.fit(X_train)
    
    y_train_scores = clf.decision_scores_  # raw outlier scores
    
    MyDataSet["Raw_anomaly_score"] = y_train_scores
    
    raw_anomaly_score = list(y_train_scores)
    
    from scipy import stats
    
    percentiles = [stats.percentileofscore(raw_anomaly_score, i) for i in raw_anomaly_score]
    
    MyDataSet["raw_anomaly_score"] = raw_anomaly_score  
    MyDataSet["anomaly_score"] = percentiles     
    
          
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
    print(my_pr_auc,my_roc_score,P_at_n_value)
    
    duringTime = 0
    duringTime2 = 0
    
    return my_pr_auc,my_roc_score,P_at_n_value,duringTime,duringTime2


# =============================================================================
# Step II. testing AD with real-world dataset
# =============================================================================

# ##########################################
# ## Step1: load dataset and set parameters
# ##########################################

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data3/powerGene.csv", sep=",")
# X_train =  RawDataSet.iloc[:, 1:-2]
# sample_value = 500

# ##########################################
# ## Step2: call PyODModel function to get results
# ##########################################
    
# PyODModel(RawDataSet, sample_value, HBOS())


