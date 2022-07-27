#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:47:32 2021

@author: zlifr
"""
########
##1.0 comparions of condional mean and condional quantiles
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)


# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':180, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# RawDataSet["label"] = "Train"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]




# new_row = {'Gender':'Male', 'Height': 60, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# #####to test

# new_row = {'Gender':'Male', 'Height': 60, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 62, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 64, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 66, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 72, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 74, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 76, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 78, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 80, 'Weight':130, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)



# X1 = RawDataSet[["Height"]].iloc[1003:1004,:]
# X2 = RawDataSet[["Height"]].iloc[1004:1005,:]
# X3 = RawDataSet[["Height"]].iloc[1005:1006,:]
# X4 = RawDataSet[["Height"]].iloc[1006:1007,:]
# X5 = RawDataSet[["Height"]].iloc[1007:1008,:]
# X6 = RawDataSet[["Height"]].iloc[1008:1009,:]
# X7 = RawDataSet[["Height"]].iloc[1009:1010,:]
# X8 = RawDataSet[["Height"]].iloc[1010:1011,:]
# X9 = RawDataSet[["Height"]].iloc[1011:1012,:]
# X10 = RawDataSet[["Height"]].iloc[1012:1013,:]
# X11 = RawDataSet[["Height"]].iloc[1013:1014,:]


# import seaborn as sns


# import pandas as pd
# import matplotlib.pyplot as plt
# from numpy import percentile
# import numbers

# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose

# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from skgarden import RandomForestQuantileRegressor

# rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
# rfqr.set_params(max_features= X_train.shape[1])
# rfqr.fit(X_train, y_train)

# q1 = []
# for i in range(10):
#     q1.append(rfqr.predict(X1, quantile=i*10)[0])
    
# q2 = []
# for i in range(10):
#     q2.append(rfqr.predict(X2, quantile=i*10)[0])

# q3 = []
# for i in range(10):
#     q3.append(rfqr.predict(X3, quantile=i*10)[0])
    
# q4 = []
# for i in range(10):
#     q4.append(rfqr.predict(X4, quantile=i*10)[0])
    
# q5 = []
# for i in range(10):
#     q5.append(rfqr.predict(X5, quantile=i*10)[0])

# q6 = []
# for i in range(10):
#     q6.append(rfqr.predict(X6, quantile=i*10)[0])    
    
# q7 = []
# for i in range(10):
#     q7.append(rfqr.predict(X7, quantile=i*10)[0])
    
# q8 = []
# for i in range(10):
#     q8.append(rfqr.predict(X8, quantile=i*10)[0])

# q9 = []
# for i in range(10):
#     q9.append(rfqr.predict(X9, quantile=i*10)[0])    
    
# q10 = []
# for i in range(10):
#     q10.append(rfqr.predict(X10, quantile=i*10)[0])
    
# q11 = []
# for i in range(10):
#     q11.append(rfqr.predict(X11, quantile=i*10)[0])
   
    
# quantile_data =  pd.DataFrame()

# quantile_data["60"] = q1
# quantile_data["62"] = q2
# quantile_data["64"] = q3
# quantile_data["66"] = q4
# quantile_data["68"] = q5
# quantile_data["70"] = q6
# quantile_data["72"] = q7
# quantile_data["74"] = q8
# quantile_data["76"] = q9
# quantile_data["78"] = q10
# quantile_data["80"] = q11
    
    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1003,:], fit_reg=True, scatter_kws={'alpha':0.2})
# # plt2 = sns.scatterplot(data=RawDataSet.iloc[0:1003,:], x="Height", y="Weight", hue = "label", style= "label", legend = False) 

# plt1.set(xlim=(58, 88), ylim=(110, 280))

# plt.text(79.2, 244.2, "Conditional mean", fontsize=8)
# plt.text(80.2, 226.2, "Conditional quantiles", fontsize=8)


# test_dataframe = pd.DataFrame()

# test_dataframe["Height"] = range(60,82,2)
# test_dataframe["Q1"] = quantile_data.loc[0, :].values.tolist()
# test_dataframe["Q2"] = quantile_data.loc[1, :].values.tolist()
# test_dataframe["Q3"] = quantile_data.loc[2, :].values.tolist()
# test_dataframe["Q4"] = quantile_data.loc[3, :].values.tolist()
# test_dataframe["Q5"] = quantile_data.loc[4, :].values.tolist()
# test_dataframe["Q6"] = quantile_data.loc[5, :].values.tolist()
# test_dataframe["Q7"] = quantile_data.loc[6, :].values.tolist()
# test_dataframe["Q8"] = quantile_data.loc[7, :].values.tolist()
# test_dataframe["Q9"] = quantile_data.loc[8, :].values.tolist()
# test_dataframe["Q10"] = quantile_data.loc[9, :].values.tolist()


# sns.lineplot(data=test_dataframe, x="Height", y="Q10", color='tab:cyan', label="tau= 0.9")
# sns.lineplot(data=test_dataframe, x="Height", y="Q9", color='tab:gray', label="tau= 0.8")
# sns.lineplot(data=test_dataframe, x="Height", y="Q8", color='tab:pink', label="tau= 0.7")
# sns.lineplot(data=test_dataframe, x="Height", y="Q7", color='tab:brown', label="tau= 0.6")
# sns.lineplot(data=test_dataframe, x="Height", y="Q6", color='tab:purple', label="tau= 0.5")
# sns.lineplot(data=test_dataframe, x="Height", y="Q5", color='tab:red', label="tau= 0.4")
# sns.lineplot(data=test_dataframe, x="Height", y="Q4", color='tab:green', label="tau= 0.3")
# sns.lineplot(data=test_dataframe, x="Height", y="Q3", color='tab:orange', label="tau= 0.2")
# sns.lineplot(data=test_dataframe, x="Height", y="Q2", color='tab:olive', label="tau= 0.1")
###############################################################################################
##1.1 Generate Point Anomalies:

# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# RawDataSet["label"] = "Normal"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]


# # new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"Abnormal"}

# # RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# # new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"Abnormal"}

# # RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# # new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"Normal"}

# # RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# import seaborn as sns
# import matplotlib.pyplot as plt

    
# # plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, hue='label', fit_reg=False, scatter_kws={'alpha':0.8},  markers=['o', 's'])
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, fit_reg=False, scatter_kws={'alpha':0.1})

# new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False) 

# plt1.set(xlim=(58, 82), ylim=(110, 280))
# plt.text(68.2, 130.2, "A: Abnormal", fontsize=8)
# plt.text(79.2, 237.2, "B: Abnormal", fontsize=8)  
# plt.text(74.2, 204.2, "C: Normal", fontsize=8)  

# #Graph settings
# top=0.97,
# bottom=0.097,
# left=0.097,
# right=0.947,
# hspace=0.2,
# wspace=0.2

# top=0.97,
# bottom=0.1,
# left=0.12,
# right=0.95,
# hspace=0.2,
# wspace=0.2
    
###############################################################################################   
##1.2 Generate Contextual Anomalies:   
    
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# RawDataSet["label"] = "Normal"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# import seaborn as sns
# import matplotlib.pyplot as plt

    
# # plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, hue='label', fit_reg=True, scatter_kws={'alpha':0.8},  markers=['o', 'x'])
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, fit_reg=True, scatter_kws={'alpha':0.1})

# new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False) 

# plt1.set(xlim=(58, 82), ylim=(110, 280))
# plt.text(68.2, 130.2, "A: Abnormal", fontsize=8)
# plt.text(79.2, 237.2, "B: Normal", fontsize=8)  
# plt.text(74.2, 204.2, "C: Normal", fontsize=8)  

###############################################################################################   
##1.2 belta Generate Contextual Anomalies with Age:   
    
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# RawDataSet["label"] = "Normal"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# def f(row):
#     if row['Weight'] > 140:
#         val = "Adult"
#     else:
#         val = "Children"
#     return val

# RawDataSet["Age"]  = RawDataSet.apply(f, axis=1)


# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]


# new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"Normal", 'Age':"Children"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':250, 'label':"Normal", 'Age':"Adult"}


# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# import seaborn as sns
# import matplotlib.pyplot as plt

    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, hue='Age', fit_reg=True, scatter_kws={'alpha':0.8})
# # plt1._legend.remove()

# new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"Abnormal", 'Age':"Children"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "Age", style= "label",legend = False) 


# # #plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, hue='label', fit_reg=True, scatter_kws={'alpha':0.8}, style = 'Age')
# plt1.set(xlim=(58, 80), ylim=(110, 280))
# plt.text(68.2, 130.2, "A: Normal", fontsize=8)
# plt.text(79.2, 250.2, "B: Normal", fontsize=8)  
# plt.text(74.2, 204.2, "C: Abnormal", fontsize=8)  


###############################################################################################  
##1.3 Generate OLS Regression Anomalies:  
    
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# RawDataSet["label"] = "Normal"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]


# new_row = {'Gender':'Male', 'Height': 75, 'Weight':160, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':250, 'label':"Normal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# import seaborn as sns
# import matplotlib.pyplot as plt

    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet,  fit_reg=True, scatter_kws={'alpha':0.8})
# plt1.set(xlim=(58, 80), ylim=(110, 280))

###############################################################################################
##1.4 Generate Quantile Regression Anomalies:  
    
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# RawDataSet["label"] = "Normal"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]


# new_row = {'Gender':'Male', 'Height': 75, 'Weight':160, 'label':"Abnormal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':250, 'label':"Normal"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# import seaborn as sns
# import matplotlib.pyplot as plt

    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet,  fit_reg=False, scatter_kws={'alpha':0.8})
# plt1.set(xlim=(58, 82), ylim=(110, 280))

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
# #make this example reproducible
# df = RawDataSet[["Height","Weight"]]
# Height =  df[["Height"]]
# Weight =  df[["Weight"]]

# #get y values
# get_y = lambda a, b: a + b * Height


# plt.text(79.5, 228, "5-quantile", fontsize=8)
# plt.text(79.5, 236, "25-quantile", fontsize=8)
# plt.text(79.5, 244, "50-quantile", fontsize=8)
# plt.text(79.5, 252, "75-quantile", fontsize=8)
# plt.text(79.5, 260, "95-quantile", fontsize=8)

# model_05 = smf.quantreg('Weight ~ Height', df).fit(q=0.05)
# Weight_pred_05 = get_y(model_05.params['Intercept'], model_05.params['Height'])
# plt05 = plt.plot(Height, Weight_pred_05, color='black')

# model_25 = smf.quantreg('Weight ~ Height', df).fit(q=0.25)
# Weight_pred_25 = get_y(model_25.params['Intercept'], model_25.params['Height'])
# plt25 = plt.plot(Height, Weight_pred_25, color='orange')

# model_50 = smf.quantreg('Weight ~ Height', df).fit(q=0.50)
# Weight_pred_50 = get_y(model_50.params['Intercept'], model_50.params['Height'])
# plt50 = plt.plot(Height, Weight_pred_50, color='yellow')


# model_75 = smf.quantreg('Weight ~ Height', df).fit(q=0.75)
# Weight_pred_75 = get_y(model_75.params['Intercept'], model_75.params['Height'])
# plt75 = plt.plot(Height, Weight_pred_75, color='green')

# model_95 = smf.quantreg('Weight ~ Height', df).fit(q=0.95)
# Weight_pred_95 = get_y(model_95.params['Intercept'], model_95.params['Height'])
# plt95 = plt.plot(Height, Weight_pred_95, color='red')


# ###############################################################################################
##1.5 Generate Quantile Regression for Anomalies Detection:  
import pandas as pd 
import math
RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

RawDataSet = RawDataSet.dropna()  #remove missing values

RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet.head(999)

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':280, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

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

# ###############################################################################################
##1.5 gamma Generate Eculidean Distance for Anomalies Detection:  
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet[RawDataSet.Height < 78]

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':180, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# RawDataSet["label"] = "Train"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]


# new_row = {'Gender':'Male', 'Height': 68, 'Weight':183.42624754238344, 'label':"Train"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# new_row = {'Gender':'Male', 'Height': 68, 'Weight':130, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# new_row = {'Gender':'Male', 'Height': 74, 'Weight':204, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 74, 'Weight':218.16516972144996, 'label':"Train"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':237, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 79, 'Weight':228.37763778980653, 'label':"Train"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# import seaborn as sns


# import pandas as pd
# import matplotlib.pyplot as plt
# from numpy import percentile
# import numbers

# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose

# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from skgarden import RandomForestQuantileRegressor

# rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
# rfqr.set_params(max_features= X_train.shape[1])
# rfqr.fit(X_train, y_train)


    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1000,:], fit_reg=False, scatter_kws={'alpha':0.25})
# plt2 = sns.scatterplot(data=RawDataSet.iloc[1000:1006,:], x="Height", y="Weight", hue = "label", style= "label", legend = False) 

# plt1.set(xlim=(58, 82), ylim=(110, 280))
# plt.plot([68,68], [130,183.4], color="red")
# plt.plot([74,74], [204,218.165], color="red")
# plt.plot([79,79], [237,228.37], color="red")

# plt.text(68.2, 130.2, "A: actual", fontsize=8)
# plt.text(68.2, 156.9, "(d = 53.4)", fontsize=8) 
# plt.text(68.2, 183.6, "A: expected", fontsize=8)
# plt.text(79.2, 237.2, "B: actual", fontsize=8)  
# plt.text(79.2, 232.9, "(d = 8.6)", fontsize=8) 
# plt.text(79.2, 228.6, "B: expected", fontsize=8)  
# plt.text(74.2, 204.2, "C: actual", fontsize=8)  
# plt.text(74.2, 211.3, "(d = 14.2)", fontsize=8) 
# plt.text(74.2, 218.4, "C: expected", fontsize=8)  



###############################################################################################
##1.6 Generate Quantile Regression for Anomalies Detection showing unrobustness:  
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# # RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet.head(999)

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':275, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# RawDataSet["label"] = "Train"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]

# X_train_clean = RawDataSet[["Height"]].iloc[0:999,:]
# y_train_clean = RawDataSet[["Weight"]].iloc[0:999,:]

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':224, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test = RawDataSet[["Height"]].iloc[1000:1001,:]
# y_test = RawDataSet[["Weight"]].iloc[1000:1001,:]

# new_row = {'Gender':'Male', 'Height': 69.90, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test2 = RawDataSet[["Height"]].iloc[1001:1002,:]
# y_test2 = RawDataSet[["Weight"]].iloc[1001:1002,:]

# new_row = {'Gender':'Male', 'Height': 69.92, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test3 = RawDataSet[["Height"]].iloc[1002:1003,:]
# y_test3 = RawDataSet[["Weight"]].iloc[1002:1003,:]

# new_row = {'Gender':'Male', 'Height': 69.94, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test4 = RawDataSet[["Height"]].iloc[1003:1004,:]
# y_test4 = RawDataSet[["Weight"]].iloc[1003:1004,:]

# new_row = {'Gender':'Male', 'Height': 69.96, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test5 = RawDataSet[["Height"]].iloc[1004:1005,:]
# y_test5 = RawDataSet[["Weight"]].iloc[1004:1005,:]


# new_row = {'Gender':'Male', 'Height': 69.98, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test6 = RawDataSet[["Height"]].iloc[1005:1006,:]
# y_test6 = RawDataSet[["Weight"]].iloc[1005:1006,:]

# new_row = {'Gender':'Male', 'Height': 70.02, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test7 = RawDataSet[["Height"]].iloc[1006:1007,:]
# y_test7 = RawDataSet[["Weight"]].iloc[1006:1007,:]

# new_row = {'Gender':'Male', 'Height': 70.04, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test8 = RawDataSet[["Height"]].iloc[1007:1008,:]
# y_test8 = RawDataSet[["Weight"]].iloc[1007:1008,:]

# new_row = {'Gender':'Male', 'Height': 70.06, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test9 = RawDataSet[["Height"]].iloc[1008:1009,:]
# y_test9 = RawDataSet[["Weight"]].iloc[1008:1009,:]


# new_row = {'Gender':'Male', 'Height': 70.08, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test10 = RawDataSet[["Height"]].iloc[1009:1010,:]
# y_test10 = RawDataSet[["Weight"]].iloc[1009:1010,:]

# new_row = {'Gender':'Male', 'Height': 70.10, 'Weight':180, 'label':"test"}
# RawDataSet = RawDataSet.append(new_row, ignore_index=True)
# X_test11 = RawDataSet[["Height"]].iloc[1010:1011,:]
# y_test11 = RawDataSet[["Weight"]].iloc[1010:1011,:]

# import seaborn as sns


# import pandas as pd
# import matplotlib.pyplot as plt
# from numpy import percentile
# import numbers

# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose

# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from skgarden import RandomForestQuantileRegressor


# rfqr_clean = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
# rfqr_clean.set_params(max_features= X_train_clean.shape[1])
# rfqr_clean.fit(X_train_clean, y_train_clean)

# quantile_vec_clean = []
# for i in range(100):
#     quantile_vec_clean.append(rfqr_clean.predict(X_test, quantile=i)[0])
    
    

# rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
# rfqr.set_params(max_features= X_train.shape[1])
# rfqr.fit(X_train, y_train)

# quantile_vec = []
# for i in range(100):
#     quantile_vec.append(rfqr.predict(X_test, quantile=i)[0])
    
# quantile_vec2 = []
# for i in range(100):
#     quantile_vec2.append(rfqr.predict(X_test2, quantile=i)[0])
    
# quantile_vec3 = []
# for i in range(100):
#     quantile_vec3.append(rfqr.predict(X_test3, quantile=i)[0])

# quantile_vec4 = []
# for i in range(100):
#     quantile_vec4.append(rfqr.predict(X_test4, quantile=i)[0])
    
# quantile_vec5 = []
# for i in range(100):
#     quantile_vec5.append(rfqr.predict(X_test5, quantile=i)[0])

# quantile_vec6 = []
# for i in range(100):
#     quantile_vec6.append(rfqr.predict(X_test6, quantile=i)[0])
    
# quantile_vec7 = []
# for i in range(100):
#     quantile_vec7.append(rfqr.predict(X_test7, quantile=i)[0])

# quantile_vec8 = []
# for i in range(100):
#     quantile_vec8.append(rfqr.predict(X_test8, quantile=i)[0])
    
# quantile_vec9 = []
# for i in range(100):
#     quantile_vec9.append(rfqr.predict(X_test9, quantile=i)[0])
    
# quantile_vec10 = []
# for i in range(100):
#     quantile_vec10.append(rfqr.predict(X_test10, quantile=i)[0])
    
# quantile_vec11 = []
# for i in range(100):
#     quantile_vec11.append(rfqr.predict(X_test11, quantile=i)[0])    
    
# import statistics
# quantile_vec_noise = [statistics.median(k) for k in zip(quantile_vec,quantile_vec4,quantile_vec5,quantile_vec6,quantile_vec7,quantile_vec8,quantile_vec9)]


###############################################################################################
##1.6 Alpha Show principles   
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1001,:], hue='label', fit_reg=False, scatter_kws={'alpha':0.8}, markers=['o', 's'])
# ## plt1.set(xlim=(58, 82), ylim=(110, 280))
# ## plt1.set(xlim=(69.93, 70.10), ylim=(110, 280))
# plt1.set(xlim=(69.93, 70.07), ylim=(110, 280))

# ## plt.text(70.5, 275.5, "outlier", fontsize=8)

# ## plt1 = plt.boxplot(quantile_vec, positions=[70], notch = True, widths = 0.1, patch_artist=True)
# plt2 = plt.boxplot(quantile_vec, positions=[70], notch = True, widths = 0.005, patch_artist=True)
# ## plt3 = plt.boxplot(quantile_vec2, positions=[69.90], notch = True, widths = 0.005, patch_artist=True)
# ## plt4 = plt.boxplot(quantile_vec3, positions=[69.92], notch = True, widths = 0.005, patch_artist=True)
# plt5 = plt.boxplot(quantile_vec4, positions=[69.94], notch = True, widths = 0.005, patch_artist=True)
# plt6 = plt.boxplot(quantile_vec5, positions=[69.96], notch = True, widths = 0.005, patch_artist=True)
# plt7 = plt.boxplot(quantile_vec6, positions=[69.98], notch = True, widths = 0.005, patch_artist=True)
# plt8 = plt.boxplot(quantile_vec7, positions=[70.02], notch = True, widths = 0.005, patch_artist=True)
# plt9 = plt.boxplot(quantile_vec8, positions=[70.04], notch = True, widths = 0.005, patch_artist=True)
# plt10 = plt.boxplot(quantile_vec9, positions=[70.06], notch = True, widths = 0.005, patch_artist=True)
# ## plt11 = plt.boxplot(quantile_vec10, positions=[70.08], notch = True, widths = 0.005, patch_artist=True)
# ## plt12 = plt.boxplot(quantile_vec11, positions=[70.10], notch = True, widths = 0.005, patch_artist=True)

###############################################################################################
##1.6 Beta Show comparisons

# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1001,:], hue='label', fit_reg=False, scatter_kws={'alpha':0.8}, markers=['o', 's'])

# plt1.set(xlim=(69.98, 70.08), ylim=(110, 280))
  
# plt2 = plt.boxplot(quantile_vec, positions=[70], notch = True, widths = 0.005, patch_artist=True)
  
# plt20 = plt.boxplot(quantile_vec_clean, positions=[70.02], notch = True, widths = 0.005, patch_artist=True)

# plt30 = plt.boxplot(quantile_vec_noise, positions=[70.04], notch = True, widths = 0.005, patch_artist=True)

# plt.text(70.001, 238, "with outlier", fontsize=8)
# plt.text(70.02, 220, "without outlier", fontsize=8)
# plt.text(70.04, 223, "average  with outlier", fontsize=8)

# for box in plt20['boxes']:
#     # change outline color
#     box.set(color='green', linewidth=2)
#     # change fill color
#     box.set(facecolor = 'green' )
#     # change hatch
#     #box.set(hatch = '/')
# for box in plt30['boxes']:
#     # change outline color
#     box.set(color='black', linewidth=2)
#     # change fill color
#     box.set(facecolor = 'black' )
#     # change hatch
#     #box.set(hatch = '/')
    
    
###############################################################################################
##1.7 Show the robustness of Quantile Regression Forest:  
    
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# # RawDataSet = RawDataSet.head(1000)

# RawDataSet = RawDataSet.head(999)

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':265, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# RawDataSet["label"] = "Train"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':224, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test = RawDataSet[["Height"]].iloc[1000:1001,:]
# y_test = RawDataSet[["Weight"]].iloc[1000:1001,:]

# new_row = {'Gender':'Male', 'Height': 69.95, 'Weight':224, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test2 = RawDataSet[["Height"]].iloc[1001:1002,:]
# y_test2 = RawDataSet[["Weight"]].iloc[1001:1002,:]


# new_row = {'Gender':'Male', 'Height': 70.05, 'Weight':224, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test3 = RawDataSet[["Height"]].iloc[1002:1003,:]
# y_test3 = RawDataSet[["Weight"]].iloc[1002:1003,:]

# new_row = {'Gender':'Male', 'Height': 78.1, 'Weight':245, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test4 = RawDataSet[["Height"]].iloc[1003:1004,:]
# y_test4 = RawDataSet[["Weight"]].iloc[1003:1004,:]



# import seaborn as sns


# import pandas as pd
# import matplotlib.pyplot as plt
# from numpy import percentile
# import numbers

# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose

# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from skgarden import RandomForestQuantileRegressor

# rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
# rfqr.set_params(max_features= X_train.shape[1])
# rfqr.fit(X_train, y_train)

# quantile_vec = []
# for i in range(100):
#     quantile_vec.append(rfqr.predict(X_test, quantile=i)[0])
    
# quantile_vec2 = []
# for i in range(100):
#     quantile_vec2.append(rfqr.predict(X_test2, quantile=i)[0])


# quantile_vec3 = []
# for i in range(100):
#     quantile_vec3.append(rfqr.predict(X_test3, quantile=i)[0])
    
# quantile_vec4 = []
# for i in range(100):
#     quantile_vec4.append(rfqr.predict(X_test4, quantile=i)[0])   
    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1001,:], hue='label', fit_reg=False, scatter_kws={'alpha':0.8}, markers=['o', 's'])

# plt2 = plt.boxplot(quantile_vec, positions=[70], notch = True, widths = 0.8, patch_artist=False)

# plt3 = plt.boxplot(quantile_vec2, positions=[69.95], notch = True, widths = 0.8, patch_artist=False)

# plt4 = plt.boxplot(quantile_vec3, positions=[70.05], notch = True, widths = 0.8, patch_artist=False)

# plt5 = plt.boxplot(quantile_vec4, positions=[78.1], notch = True, widths = 0.8, patch_artist=False)

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
# #make this example reproducible
# df = RawDataSet[["Height","Weight"]]
# Height =  df[["Height"]]
# Weight =  df[["Weight"]]

# #get y values
# get_y = lambda a, b: a + b * Height


# plt.text(79.5, 228, "5-quantile", fontsize=8)
# plt.text(79.5, 236, "25-quantile", fontsize=8)
# plt.text(79.5, 244, "50-quantile", fontsize=8)
# plt.text(79.5, 252, "75-quantile", fontsize=8)
# plt.text(79.5, 260, "95-quantile", fontsize=8)

# model_05 = smf.quantreg('Weight ~ Height', df).fit(q=0.05)
# Weight_pred_05 = get_y(model_05.params['Intercept'], model_05.params['Height'])
# plt05 = plt.plot(Height, Weight_pred_05, color='black')

# model_25 = smf.quantreg('Weight ~ Height', df).fit(q=0.25)
# Weight_pred_25 = get_y(model_25.params['Intercept'], model_25.params['Height'])
# plt25 = plt.plot(Height, Weight_pred_25, color='orange')

# model_50 = smf.quantreg('Weight ~ Height', df).fit(q=0.50)
# Weight_pred_50 = get_y(model_50.params['Intercept'], model_50.params['Height'])
# plt50 = plt.plot(Height, Weight_pred_50, color='yellow')


# model_75 = smf.quantreg('Weight ~ Height', df).fit(q=0.75)
# Weight_pred_75 = get_y(model_75.params['Intercept'], model_75.params['Height'])
# plt75 = plt.plot(Height, Weight_pred_75, color='green')

# model_95 = smf.quantreg('Weight ~ Height', df).fit(q=0.95)
# Weight_pred_95 = get_y(model_95.params['Intercept'], model_95.params['Height'])
# plt95 = plt.plot(Height, Weight_pred_95, color='red')
    
 ###############################################################################################   
    
# import pandas as pd 
# import math
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/weight-height.csv", sep=",")

# RawDataSet = RawDataSet.dropna()  #remove missing values

# RawDataSet = RawDataSet.head(1000)

# # RawDataSet = RawDataSet.head(999)

# # new_row = {'Gender':'Male', 'Height': 70, 'Weight':280, 'label':"test"}

# # RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# RawDataSet["label"] = "Train"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]

# new_row = {'Gender':'Male', 'Height': 70, 'Weight':224, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test = RawDataSet[["Height"]].iloc[1000:1001,:]
# y_test = RawDataSet[["Weight"]].iloc[1000:1001,:]

# new_row = {'Gender':'Male', 'Height': 62, 'Weight':180, 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# X_test2 = RawDataSet[["Height"]].iloc[1001:1002,:]
# y_test2 = RawDataSet[["Weight"]].iloc[1001:1002,:]


# import seaborn as sns

# # sns.lmplot("Height", "Weight", data=RawDataSet, hue='Gender', fit_reg=True, scatter_kws={'alpha':0.4})
# # g = sns.lmplot("Height", "Weight", data=RawDataSet, hue='label', fit_reg=True, scatter_kws={'alpha':0.4})

# # g.set(ylim=(100, 260))


# import pandas as pd
# import matplotlib.pyplot as plt
# from numpy import percentile
# import numbers

# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose

# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from skgarden import RandomForestQuantileRegressor

# rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=30, n_estimators=100)
# rfqr.set_params(max_features= X_train.shape[1])
# rfqr.fit(X_train, y_train)

# quantile_vec = []
# for i in range(100):
#     quantile_vec.append(rfqr.predict(X_test, quantile=i)[0])
    
# quantile_vec2 = []
# for i in range(100):
#     quantile_vec2.append(rfqr.predict(X_test2, quantile=i)[0])
    
    
# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet.iloc[0:1001,:], hue='label', fit_reg=True, scatter_kws={'alpha':0.4}, markers=['o', 's'])

# # axes = plt1.axes

# # axes[0,0].set_ylim(100,260)
# # axes[0,0].set_xlim(60,80)

# plt2 = plt.boxplot(quantile_vec, positions=[70], notch = True, widths = 0.4, patch_artist=False)

# plt3 = plt.boxplot(quantile_vec2, positions=[62], notch = True, widths = 0.4, patch_artist=False)
# # fig, ax = plt.subplots()
# # plt1 = plt.scatter(X_train,y_train, alpha=0.5)
# # plt2 = plt.scatter(X_test,y_test, alpha=0.5)
# # plt3 = plt.boxplot(quantile_vec, positions=[62], notch = True, widths = 0.4)
# # plt.legend((plt1,plt2), ('train', 'test'))
# # ax.set_ylim([100,260])
# # ax.set_xlim([60,80])
# # ax.set_ylabel("Weight")
# # plt.savefig("boxplot.png")
# # plt.close()

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
# #make this example reproducible
# df = RawDataSet[["Height","Weight"]].iloc[0:1000,:]
# Height =  df[["Height"]]
# Weight =  df[["Weight"]]

# #get y values
# get_y = lambda a, b: a + b * Height

# model_75 = smf.quantreg('Weight ~ Height', df).fit(q=0.75)
# Weight_pred_75 = get_y(model_75.params['Intercept'], model_75.params['Height'])
# plt75 = plt.plot(Height, Weight_pred_75, color='green', label ="75")

# plt.text(78.5, 224, "5-quantile", fontsize=8)
# plt.text(78.5, 232, "25-quantile", fontsize=8)
# plt.text(78.5, 240, "50-quantile", fontsize=8)
# plt.text(78.5, 248, "75-quantile", fontsize=8)
# plt.text(78.5, 256, "95-quantile", fontsize=8)


# model_25 = smf.quantreg('Weight ~ Height', df).fit(q=0.25)
# Weight_pred_25 = get_y(model_25.params['Intercept'], model_25.params['Height'])
# plt25 = plt.plot(Height, Weight_pred_25, color='orange')

# model_95 = smf.quantreg('Weight ~ Height', df).fit(q=0.95)
# Weight_pred_95 = get_y(model_95.params['Intercept'], model_95.params['Height'])
# plt95 = plt.plot(Height, Weight_pred_95, color='red')

# model_05 = smf.quantreg('Weight ~ Height', df).fit(q=0.05)
# Weight_pred_05 = get_y(model_05.params['Intercept'], model_05.params['Height'])
# plt05 = plt.plot(Height, Weight_pred_05, color='black')

###############################################################################################
##1.8 Plot skewed normal distribution

# from numpy import random
# import matplotlib.pyplot as plt
# import seaborn as sns 
# from scipy.stats import skewnorm
# import numpy as np
# from scipy.stats import skewnorm

# sample = random.normal(size=10000, loc=50, scale=20)

# X= np.linspace(min(sample), max(sample))
# Y= np.linspace(min(sample), max(sample))

# sample2 = skewnorm.pdf(X, 1, 20, 24) 
# sample3 = skewnorm.pdf(Y, 1, 60, 24) 

# sns.distplot(sample, hist=True, label="skewness = 0")
# # sns.distplot(sample2, hist=False, label="skewness > 0")
# plt.plot(X, sample2, label="skewness > 0")
# plt.plot(Y, sample3, label="skewness < 0")
# plt.legend()    
# plt.show()

# # top=0.88,
# # bottom=0.11,
# # left=0.13,
# # right=0.9,
# # hspace=0.2,
# # wspace=0.2


# from numpy import random
# import matplotlib.pyplot as plt
# import seaborn as sns 
# from scipy.stats import skewnorm
# import numpy as np
# from scipy.stats import skewnorm

# sample = random.normal(size=10000, loc=50, scale=20)
# sample2 = random.normal(size=10000, loc=50, scale=10)
# sample3 = random.normal(size=10000, loc=50, scale=30)

# sns.distplot(sample, hist=True, label="variance = 20")
# sns.distplot(sample2, hist=False, label="variance = 10")
# sns.distplot(sample3, hist=False, label="variance = 30")
# plt.legend()    
# plt.show()
