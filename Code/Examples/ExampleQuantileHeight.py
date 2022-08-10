#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:47:32 2021

@author: zlifr
"""

##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub' 

# =============================================================================
# Step1. Load and preprocess dataset
# =============================================================================
import pandas as pd 
import math

RawDataSetPath = AbsRootDir + r'/QCAD/Data/Examples/weight-height.csv'

RawDataSet = pd.read_csv(RawDataSetPath, sep=",")

RawDataSet = RawDataSet.dropna()  #remove missing values

RawDataSet = RawDataSet.head(1000)

RawDataSet = RawDataSet[RawDataSet.Height < 78]

new_row = {'Gender':'Male', 'Height': 70, 'Weight':180, 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)


RawDataSet["label"] = "Train"

RawDataSet["Weight"] = RawDataSet["Weight"]

X_train = RawDataSet[["Height"]]
y_train = RawDataSet[["Weight"]]

new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)

X_test = RawDataSet[["Height"]].iloc[1000:1001,:]
y_test = RawDataSet[["Weight"]].iloc[1000:1001,:]

new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)

new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)

X_test2 = RawDataSet[["Height"]].iloc[1001:1002,:]
y_test2 = RawDataSet[["Weight"]].iloc[1001:1002,:]

X_test3 = RawDataSet[["Height"]].iloc[1002:1003,:]
y_test3 = RawDataSet[["Weight"]].iloc[1002:1003,:]

# =============================================================================
# Step 2. Generate quantile plots
# =============================================================================

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from numpy import percentile
import numbers

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skgarden import RandomForestQuantileRegressor

rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
rfqr.set_params(max_features= X_train.shape[1])
rfqr.fit(X_train, y_train)

quantile_vec = []
for i in range(100):
    quantile_vec.append(rfqr.predict(X_test, quantile=i)[0])
    
quantile_vec2 = []
for i in range(100):
    quantile_vec2.append(rfqr.predict(X_test2, quantile=i)[0])

quantile_vec3 = []
for i in range(100):
    quantile_vec3.append(rfqr.predict(X_test3, quantile=i)[0])
    
plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1000,:], fit_reg=False, scatter_kws={'alpha':0.1})
plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False) 

plt1.set(xlim=(58, 82), ylim=(110, 280))

plt2 = plt.boxplot(quantile_vec, positions=[68], notch = True, widths = 0.4, patch_artist=True)
plt3 = plt.boxplot(quantile_vec2, positions=[74], notch = True, widths = 0.4, patch_artist=True)
plt4 = plt.boxplot(quantile_vec3, positions=[79], notch = True, widths = 0.4, patch_artist=True)

plt.text(68.2, 130.2, "A: tau<0", fontsize=8)
plt.text(79.2, 237.2, "B: tau=0.96", fontsize=8)  
plt.text(74.2, 204.2, "C: tau=0.14", fontsize=8)  

