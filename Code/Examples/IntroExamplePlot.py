#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:11:55 2021

@author: zlifr
"""
# import pandas as pd 
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Examples/fishOriginal.csv", sep=",")

# RawDataSet

# RawDataSet = RawDataSet.rename(columns={'Length1': 'Vertical Length',
#                                         'Length2': 'Diagnonal Length',
#                                         'Length3': 'Cross Length'})

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt2 = sns.scatterplot(data=RawDataSet, x="Vertical Length", y="Weight", hue = "Species", legend = True) 


####################################################################################
##1. Compare conditional mean and conditional quantiles
####################################################################################

# import seaborn as sns
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import random
# import math

# np.random.seed(43)

# my_attr_mean_vec = [149,63]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 300

# x,y = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

# my_attr_mean_vec = [169,73]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 300

# z,w = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

# my_attr_mean_vec = [180,80]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 400

# m,n = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T


# # plt.scatter(z, w)

# # plt.scatter(x, y)

# # plt.scatter(m, n)


# height = np.concatenate((x, z, m))

# weight = np.concatenate((y, w, n))


# RawDataSet = pd.DataFrame()
# RawDataSet["Height"] = height
# RawDataSet["Weight"] = weight
# RawDataSet["Gender"] = "Male"
# RawDataSet["label"] = "train"


# new_row = {'Gender':'Male', 'Height': 145, 'Weight':60, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 145, 'Weight':60, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 145, 'Weight':60, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# RawDataSet["label"] = "Train"

# RawDataSet["Weight"] = RawDataSet["Weight"]

# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]



# #####to test

# new_row = {'Gender':'Male', 'Height': 140, 'Weight':40, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 145, 'Weight':45, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 150, 'Weight':50, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 155, 'Weight':55, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 160, 'Weight':60, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 165, 'Weight':65, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 170, 'Weight':70, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 175, 'Weight':75, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 180, 'Weight':80, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 185, 'Weight':85, 'label':"train"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 190, 'Weight':90, 'label':"train"}

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

# new_row = {'Gender':'Male', 'Height': 158.5, 'Weight':69.5, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 146.3, 'Weight':65.5, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 175, 'Weight':64, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 185, 'Weight':80, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# plt2 = sns.scatterplot(data=RawDataSet.iloc[1014:,:], x="Height", y="Weight", hue = "label", style= "label", legend = False) 

# plt1.set(xlim=(135, 210), ylim=(50, 95))

# plt.text(190.2, 84.2, "Conditional mean", fontsize=8)
# plt.text(190.2,82.2, "Conditional quantiles", fontsize=8)


# test_dataframe = pd.DataFrame()

# test_dataframe["Height"] = range(140,195,5)
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


# sns.lineplot(data=test_dataframe, x="Height", y="Q10", color='tab:cyan', label="tau= 0.9", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q9", color='tab:gray', label="tau= 0.8", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q8", color='tab:pink', label="tau= 0.7", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q7", color='tab:brown', label="tau= 0.6", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q6", color='tab:purple', label="tau= 0.5", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q5", color='tab:red', label="tau= 0.4", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q4", color='tab:green', label="tau= 0.3", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q3", color='tab:orange', label="tau= 0.2", alpha= 0.6)
# sns.lineplot(data=test_dataframe, x="Height", y="Q2", color='tab:olive', label="tau= 0.1", alpha= 0.6)


# plt.legend([],[], frameon=False)

####################################################################################
##2. Compare conditional mean and conditional quantiles
####################################################################################

# import seaborn as sns
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import random
# import math

# np.random.seed(43)

# my_attr_mean_vec = [149,63]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 300

# x,y = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

# my_attr_mean_vec = [169,73]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 300

# z,w = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

# my_attr_mean_vec = [180,80]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 400

# m,n = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T


# # plt.scatter(z, w)

# # plt.scatter(x, y)

# # plt.scatter(m, n)

# height = np.concatenate((x, z, m))

# weight = np.concatenate((y, w, n))


# RawDataSet = pd.DataFrame()
# RawDataSet["Height"] = height
# RawDataSet["Weight"] = weight
# RawDataSet["Gender"] = "Male"
# RawDataSet["label"] = "train"



# new_row = {'Gender':'Male', 'Height': 158.5, 'Weight':69.5, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 146.3, 'Weight':65.5, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 175, 'Weight':64, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 185, 'Weight':80, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# plt1 = sns.lmplot("Height", "Weight", data=RawDataSet, fit_reg=True, scatter_kws={'alpha':0.1})


# plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False) 

# plt1.set(xlim=(135, 200), ylim=(50, 95))

# plt.plot([146.3,139], [65.5,73.9], color="black", ls = "dashed")
# plt.plot([158.5,160], [69.5,61], color="black", ls = "dashed")
# plt.plot([185,187], [80,76], color="black", ls = "dashed")

# plt.text(137, 74, "A: Normal", fontsize=8)
# plt.text(160, 60, "B: Normal", fontsize=8, color = "red")  
# plt.text(176, 64, "C: Abnormal", fontsize=8)  
# plt.text(187, 75, "D: Normal", fontsize=8)  

# top=0.97,
# bottom=0.185,
# left=0.11,
# right=0.97,
# hspace=0.2,
# wspace=0.2
#328-352


####################################################################################
##3. Locate quantiles as anomaly scores
####################################################################################

# import seaborn as sns
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import random
# import math

# np.random.seed(43)

# my_attr_mean_vec = [149,63]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 300

# x,y = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

# my_attr_mean_vec = [169,73]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 300

# z,w = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

# my_attr_mean_vec = [180,80]
# corr_matrix = [[5,10],[10,5]]

# size_each_gaussian = 400

# m,n = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T


# # plt.scatter(z, w)

# # plt.scatter(x, y)

# # plt.scatter(m, n)

# height = np.concatenate((x, z, m))

# weight = np.concatenate((y, w, n))


# RawDataSet = pd.DataFrame()
# RawDataSet["Height"] = height
# RawDataSet["Weight"] = weight
# RawDataSet["Gender"] = "Male"
# RawDataSet["label"] = "train"


# X_train = RawDataSet[["Height"]]
# y_train = RawDataSet[["Weight"]]

# new_row = {'Gender':'Male', 'Height': 158.5, 'Weight':69.5, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 146.3, 'Weight':65.5, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 175, 'Weight':64, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)

# new_row = {'Gender':'Male', 'Height': 185, 'Weight':80, "Gender": "Male", 'label':"test"}

# RawDataSet = RawDataSet.append(new_row, ignore_index=True)


# X_test = RawDataSet[["Height"]].iloc[1000:1001,:]
# y_test = RawDataSet[["Weight"]].iloc[1000:1001,:]

# X_test2 = RawDataSet[["Height"]].iloc[1001:1002,:]
# y_test2 = RawDataSet[["Weight"]].iloc[1001:1002,:]

# X_test3 = RawDataSet[["Height"]].iloc[1002:1003,:]
# y_test3 = RawDataSet[["Weight"]].iloc[1002:1003,:]

# X_test4 = RawDataSet[["Height"]].iloc[1003:1004,:]
# y_test4 = RawDataSet[["Weight"]].iloc[1003:1004,:]


# plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False, alpha= 0.9) 

# plt2.set(xlim=(135, 200), ylim=(50, 95))

# plt.plot([146.3,139], [65.5,73.9], color="black", ls = "dashed")
# plt.plot([158.5,146], [69.5,78], color="black", ls = "dashed")
# plt.plot([185,187], [80,76], color="black", ls = "dashed")

# plt.text(137, 74, r"A: $= \hat{\tau}_{0.90}$", fontsize=8)
# plt.text(146, 78, r"B: $= \hat{\tau}_{0.85}$", fontsize=8)  
# plt.text(176, 64, r"C: $< \hat{\tau}_{0.0}$", fontsize=8)  
# plt.text(187, 75, r"D: $= \hat{\tau}_{0.35}$", fontsize=8)  


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
#     quantile_vec3.append(rfqr.predict(X_test4, quantile=i)[0])
    
    
# plt2 = plt.boxplot(quantile_vec, positions=[158.5], notch = True, widths = 1.5, patch_artist=True, showfliers=False)
# plt3 = plt.boxplot(quantile_vec2, positions=[146.3], notch = True, widths = 1.5, patch_artist=True, showfliers=False)
# plt4 = plt.boxplot(quantile_vec3, positions=[175], notch = True, widths = 1.5, patch_artist=True, showfliers=False)
# plt5 = plt.boxplot(quantile_vec3, positions=[185], notch = True, widths = 1.5, patch_artist=True, showfliers=False)


# plt6 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False, alpha= 0.0) 

# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

##############################################################################
##Example4  Beanplot
##############################################################################
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

np.random.seed(43)

my_attr_mean_vec = [149,63]
corr_matrix = [[5,10],[10,5]]

size_each_gaussian = 300

x,y = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

my_attr_mean_vec = [169,73]
corr_matrix = [[5,10],[10,5]]

size_each_gaussian = 300

z,w = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T

my_attr_mean_vec = [180,80]
corr_matrix = [[5,10],[10,5]]

size_each_gaussian = 400

m,n = np.random.multivariate_normal(my_attr_mean_vec, corr_matrix, size_each_gaussian).T


height = np.concatenate((x, z, m))

weight = np.concatenate((y, w, n))


RawDataSet = pd.DataFrame()
RawDataSet["Height"] = height
RawDataSet["Weight"] = weight
RawDataSet["Gender"] = "Male"
RawDataSet["label"] = "train"


X_train = RawDataSet[["Height"]]
y_train = RawDataSet[["Weight"]]

new_row = {'Gender':'Male', 'Height': 158.5, 'Weight':69.5, "Gender": "Male", 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)

new_row = {'Gender':'Male', 'Height': 146.3, 'Weight':65.5, "Gender": "Male", 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)

new_row = {'Gender':'Male', 'Height': 175, 'Weight':64, "Gender": "Male", 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)

new_row = {'Gender':'Male', 'Height': 185, 'Weight':80, "Gender": "Male", 'label':"test"}

RawDataSet = RawDataSet.append(new_row, ignore_index=True)


X_test = RawDataSet[["Height"]].iloc[1000:1001,:]
y_test = RawDataSet[["Weight"]].iloc[1000:1001,:]

X_test2 = RawDataSet[["Height"]].iloc[1001:1002,:]
y_test2 = RawDataSet[["Weight"]].iloc[1001:1002,:]

X_test3 = RawDataSet[["Height"]].iloc[1002:1003,:]
y_test3 = RawDataSet[["Weight"]].iloc[1002:1003,:]

X_test4 = RawDataSet[["Height"]].iloc[1003:1004,:]
y_test4 = RawDataSet[["Weight"]].iloc[1003:1004,:]


plt2 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False, alpha= 0.7) 

plt2.set(xlim=(135, 200), ylim=(50, 95))

plt.plot([146.3,139], [65.5,73.9], color="black", ls = "dashed")
plt.plot([158.5,146], [69.5,78], color="black", ls = "dashed")
plt.plot([185,187], [80,76], color="black", ls = "dashed")

plt.text(138, 74, r"A: Normal", fontsize=8)
plt.text(145.5, 78, r"B: Normal", fontsize=8)  
plt.text(176, 64.5, r"C: Abnormal", fontsize=8)  
plt.text(187, 75, r"D: Normal", fontsize=8)  


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
for i in range(101):
    quantile_vec.append(rfqr.predict(X_test, quantile=i)[0])
    
quantile_vec2 = []
for i in range(101):
    quantile_vec2.append(rfqr.predict(X_test2, quantile=i)[0])

quantile_vec3 = []
for i in range(101):
    quantile_vec3.append(rfqr.predict(X_test3, quantile=i)[0])
    
quantile_vec4 = []
for i in range(101):
    quantile_vec4.append(rfqr.predict(X_test4, quantile=i)[0])
    
    
# plt2 = plt.boxplot(quantile_vec, positions=[158.5], notch = True, widths = 1.5, patch_artist=True, showfliers=False)
# plt3 = plt.boxplot(quantile_vec2, positions=[146.3], notch = True, widths = 1.5, patch_artist=True, showfliers=False)
# plt4 = plt.boxplot(quantile_vec3, positions=[175], notch = True, widths = 1.5, patch_artist=True, showfliers=False)
# plt5 = plt.boxplot(quantile_vec4, positions=[185], notch = True, widths = 1.5, patch_artist=True, showfliers=False)


plt6 = sns.scatterplot(data=RawDataSet, x="Height", y="Weight", hue = "label", style= "label", legend = False, alpha= 0.0) 

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.grid(False)

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




X = quantile_vec2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def beanplot_variant(actual_value, position, quantile_vec, X_test):
    
    ##Paramter 1
    actual_value = actual_value
    
    ##Paramter 2
    position = position
    
    ##Paramter 3
    quantile_vec = quantile_vec
    
    ##Paramter 4
    X_test = X_test
    
    feature_name = "Weight"
    quantile_vec_diff = np.diff(quantile_vec)
    len_interval = 1/len(quantile_vec)
    density_vec = [len_interval/x for x in quantile_vec_diff]
    print(np.max(density_vec))
    # density_vec = [min(0.08, x) for x in density_vec]
    density_vec = [min(0.11, x) for x in density_vec]
    density_vec = [ x*0.5 for x in density_vec]
    print(np.max(density_vec))
    
    len_mark = np.max(density_vec)/2 #length of black line
    len_basic = (np.max(quantile_vec)-np.min(quantile_vec))/8000 #length of blue line 
    len_box = len_basic*6
    density_vec.append(0)
    
    ##other paramters
    scale_factor = 60
    len_box = len_box*scale_factor
    len_mark = len_mark*scale_factor
    len_basic = len_basic*scale_factor
    
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    
    # plt.ylabel("behavioural feature: "+feature_name)
    plt.ylabel("Weight (kg)")
    plt.xlabel("Height (cm)")
    
    data_to_plot = pd.DataFrame()
    data_to_plot["x_axis"] =  quantile_vec
    data_to_plot["y_axis"] = [x*scale_factor  + position for x in density_vec]
    data_to_plot["z_axis"] = [-x*scale_factor + position for x in density_vec]
    
    # plt.plot(data_to_plot["y_axis"],data_to_plot["x_axis"],color="orange")
    # plt.plot(data_to_plot["z_axis"],data_to_plot["x_axis"],color="orange")
    plt.fill(np.append(data_to_plot["y_axis"], data_to_plot["z_axis"][::-1]),
             np.append(data_to_plot["x_axis"], data_to_plot["x_axis"][::-1]),
             color='red', alpha=0.6)
    
    
    ##Plot basic blue line segments
    for quantile_value in quantile_vec:
        plt.plot([-len_basic+position, len_basic+position], [quantile_value, quantile_value], color="blue")
      
        
    ##Plot terget object   
    plt.plot([-len_mark+position,len_mark+position], [actual_value, actual_value], color="black", alpha=1)
    
    
    ##Plot quartiles
    q25 = rfqr.predict(X_test, quantile=25)
    q50 = rfqr.predict(X_test, quantile=50)
    q75 = rfqr.predict(X_test, quantile=75)
    
    
    plt.plot([-len_box+position, len_box+position], [q25, q25], color="cyan")
    plt.plot([-len_box+position, len_box+position], [q50, q50], color="cyan")
    plt.plot([-len_box+position, len_box+position], [q75, q75], color="cyan")
    plt.plot([-len_box+position, -len_box+position], [q25, q75], color="cyan") #connecting them with a box
    plt.plot([len_box+position, len_box+position], [q25, q75], color="cyan") #connecting them with a box
    
beanplot_variant(69.5, 158.5, quantile_vec, X_test)
beanplot_variant(65.5, 146.3, quantile_vec2, X_test2)
beanplot_variant(64, 175, quantile_vec3, X_test3)
beanplot_variant(80, 185, quantile_vec4, X_test4)