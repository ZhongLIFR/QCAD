#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 09:14:46 2021

@author: zlifr
"""
##############################################################################
##Example 1, compute gower's distance matrix
##############################################################################
# import pandas as pd
# import re
# import itertools
# import numpy as np
# RawDataSet = pd.read_excel("/Users/zlifr/Desktop/Chunyuan/FishExample.xlsx")

# import gower

# distance_matrix = gower.gower_matrix(RawDataSet)

# distance_matrix = np.round(distance_matrix,2)

# distance_matrix = pd.DataFrame(distance_matrix).T

# distance_matrix.to_excel("/Users/zlifr/Desktop/Chunyuan/FishExampleResults.xlsx", engine='xlsxwriter')

##############################################################################
##Example 2, estimate density from quantiles 
##############################################################################

# import pandas as pd 
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/Concrete.csv", sep=",")

# import numpy as np
# import seaborn as sns

# sns.set_style('whitegrid')
# sns.kdeplot(RawDataSet["Age"], bw=0.5)

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

N=2000
mu, sigma = 0.36, 0.10
mu2, sigma2 = 0.7, 0.065
X1 = np.random.normal(mu, sigma, N)
X2 = np.random.normal(mu2, sigma2, N)
X = np.concatenate([X1, X2])

# my_kde = sns.kdeplot(X, bw=0.2)
# # plt.xlabel("Weight (g) conditioned on Vertical Length = 20 cm of Perch fish (Synthetic data)")
# plt.xlabel("Weight conditioned on Height = 0.7 (normalized data)")

# plt.xlim(0, 1)

# line = my_kde.lines[0]
# x, y = line.get_data()
# my_data = pd.DataFrame()
# my_data["x"] = x
# my_data["y"] = y

# my_kde.spines['right'].set_visible(False)
# my_kde.spines['top'].set_visible(False)

##############################################################################
##Step 2.1
##############################################################################

# print(np.quantile(X, 0.1, axis=0))
# print(np.quantile(X, 0.2, axis=0))
# print(np.quantile(X, 0.3, axis=0))
# print(np.quantile(X, 0.4, axis=0))
# print(np.quantile(X, 0.5, axis=0))
# print(np.quantile(X, 0.6, axis=0))
# print(np.quantile(X, 0.7, axis=0))
# print(np.quantile(X, 0.8, axis=0))
# print(np.quantile(X, 0.9, axis=0))


# plt.plot([0.276, 0.276], [0, 1.39], color='blue')
# plt.plot([0.332, 0.332], [0, 1.86], color='orange')
# plt.plot([0.382, 0.382], [0, 1.88], color='green')
# plt.plot([0.441, 0.441], [0, 1.34], color='red')
# plt.plot([0.569, 0.569], [0, 0.88], color='purple')
# plt.plot([0.643, 0.643], [0, 2.14], color='brown')
# plt.plot([0.680, 0.680], [0, 2.59], color='pink')
# plt.plot([0.714, 0.714], [0, 2.58], color='gray')
# plt.plot([0.752, 0.752], [0, 2.07], color='olive')

# my_kde.fill_between(x, y, where=(x<0.276) & (y>0), interpolate=True, color='blue', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.276) & (x<0.332) & (y>0), interpolate=True, color='orange', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.332) & (x<0.382) & (y>0), interpolate=True, color='green', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.382) & (x<0.441) & (y>0), interpolate=True, color='red', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.441) & (x<0.569) & (y>0), interpolate=True, color='purple', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.569) & (x<0.643) & (y>0), interpolate=True, color='brown', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.643) & (x<0.680) & (y>0), interpolate=True, color='pink', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.680) & (x<0.714) & (y>0), interpolate=True, color='gray', alpha=0.2)
# my_kde.fill_between(x, y, where=(x>0.714) & (x<0.752) & (y>0), interpolate=True, color='olive', alpha=0.2)
# my_kde.fill_between(x, y, where= (x>0.752) & (y>0), interpolate=True, color='cyan', alpha=0.2)

# # plt.text(0.276, 1.39, r"$\tau_{0.1}$", fontsize=8)
# # plt.text(0.332, 1.84, r"$\tau_{0.2}$", fontsize=8)
# # plt.text(0.382, 1.86, r"$\tau_{0.3}$", fontsize=8)
# # plt.text(0.441, 1.32, r"$\tau_{0.4}$", fontsize=8)
# # plt.text(0.569, 0.86, r"$\tau_{0.5}$", fontsize=8)
# # plt.text(0.643, 2.12, r"$\tau_{0.6}$", fontsize=8)
# # plt.text(0.680, 2.57, r"$\tau_{0.7}$", fontsize=8)
# # plt.text(0.714, 2.56, r"$\tau_{0.8}$", fontsize=8)
# # plt.text(0.752, 2.05, r"$\tau_{0.9}$", fontsize=8)

# # top=0.98,
# # bottom=0.135,
# # left=0.08,
# # right=0.955,
# # hspace=0.2,
# # wspace=0.2

# plt.text(0.276, 0.02, r"$\tau_{0.1}$", fontsize=8)
# plt.text(0.332, 0.02, r"$\tau_{0.2}$", fontsize=8)
# plt.text(0.382, 0.02, r"$\tau_{0.3}$", fontsize=8)
# plt.text(0.441, 0.02, r"$\tau_{0.4}$", fontsize=8)
# plt.text(0.569, 0.02, r"$\tau_{0.5}$", fontsize=8)
# plt.text(0.643, 0.02, r"$\tau_{0.6}$", fontsize=8)
# plt.text(0.680, 0.02, r"$\tau_{0.7}$", fontsize=8)
# plt.text(0.714, 0.02, r"$\tau_{0.8}$", fontsize=8)
# plt.text(0.752, 0.02, r"$\tau_{0.9}$", fontsize=8)

# plt.text(0.20, 0.25, "10%", fontsize=8)
# plt.text(0.28, 0.50, "10%", fontsize=8)
# plt.text(0.335, 0.75, "10%", fontsize=8)
# plt.text(0.385, 0.50, "10%", fontsize=8)
# plt.text(0.48, 0.25, "10%", fontsize=8)
# plt.text(0.58, 0.50, "10%", fontsize=8)
# plt.text(0.643, 0.75, "10%", fontsize=8)
# plt.text(0.680, 1.0, "10%", fontsize=8)
# plt.text(0.714, 0.75, "10%", fontsize=8)
# plt.text(0.79, 0.25, "10%", fontsize=8)

##############################################################################
##Step 2.2 
##############################################################################


# print(np.quantile(X, 0.41, axis=0))
# print(np.quantile(X, 0.42, axis=0))
# print(np.quantile(X, 0.43, axis=0))
# print(np.quantile(X, 0.44, axis=0))
# print(np.quantile(X, 0.45, axis=0))
# print(np.quantile(X, 0.46, axis=0))
# print(np.quantile(X, 0.47, axis=0))
# print(np.quantile(X, 0.48, axis=0))
# print(np.quantile(X, 0.49, axis=0))


# plt.plot([0.448, 0.448], [0, 1.29], color='red', alpha =0.4)
# plt.plot([0.455, 0.455], [0, 1.20], color='red', alpha =0.4)
# plt.plot([0.463, 0.463], [0, 1.13], color='red', alpha =0.4)
# plt.plot([0.472, 0.472], [0, 1.02], color='red', alpha =0.4)
# plt.plot([0.487, 0.487], [0, 0.90], color='red', alpha =0.4)
# plt.plot([0.499, 0.499], [0, 0.82], color='red', alpha =0.4)
# plt.plot([0.513, 0.513], [0, 0.74], color='red', alpha =0.4)
# plt.plot([0.531, 0.531], [0, 0.70], color='red', alpha =0.4)
# plt.plot([0.550, 0.550], [0, 0.74], color='red', alpha =0.4)

# plt.plot([0.442, 0.442], [0, 1.34], color='red')
# plt.plot([0.5715, 0.5715], [0, 0.89], color='purple')


# plt.xlim(0.440, 0.578)

# my_kde.fill_between(x, y, where=(x>0.440) & (x<0.5715) & (y>0), interpolate=True, color='purple', alpha=0.2)



# plt.text(0.443, 0.50, "1%", fontsize=8)
# plt.text(0.449, 0.50, "1%", fontsize=8)
# plt.text(0.456, 0.50, "1%", fontsize=8)
# plt.text(0.466, 0.50, "1%", fontsize=8)
# plt.text(0.475, 0.50, "1%", fontsize=8)
# plt.text(0.490, 0.50, "1%", fontsize=8)
# plt.text(0.504, 0.50, "1%", fontsize=8)
# plt.text(0.520, 0.50, "1%", fontsize=8)
# plt.text(0.538, 0.50, "1%", fontsize=8)
# plt.text(0.560, 0.50, "1%", fontsize=8)


# plt.text(0.442, 0.04, r"$\tau_{0.40}$", fontsize=8)
# plt.text(0.449, 0.04, r"$\tau_{0.41}$", fontsize=8)
# plt.text(0.456, 0.04, r"$\tau_{0.42}$", fontsize=8)
# plt.text(0.464, 0.04, r"$\tau_{0.43}$", fontsize=8)
# plt.text(0.473, 0.04, r"$\tau_{0.44}$", fontsize=8)
# plt.text(0.488, 0.04, r"$\tau_{0.45}$", fontsize=8)
# plt.text(0.500, 0.04, r"$\tau_{0.46}$", fontsize=8)
# plt.text(0.514, 0.04, r"$\tau_{0.47}$", fontsize=8)
# plt.text(0.532, 0.04, r"$\tau_{0.48}$", fontsize=8)
# plt.text(0.551, 0.04, r"$\tau_{0.49}$", fontsize=8)
# plt.text(0.572, 0.04, r"$\tau_{0.50}$", fontsize=8)

# plt.ylim(0, 1.5)


# import matplotlib.patches as patches
# rect1 = patches.Rectangle((0.442, 0), 0.006, 1.35, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect1)
# rect2 = patches.Rectangle((0.448, 0), 0.007, 1.29, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect2)
# rect3 = patches.Rectangle((0.455, 0), 0.008, 1.20, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect3)
# rect4 = patches.Rectangle((0.463, 0), 0.009, 1.13, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect4)
# rect5 = patches.Rectangle((0.472, 0), 0.015, 1.02, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect5)
# rect6 = patches.Rectangle((0.487, 0), 0.012, 0.90, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect6)
# rect7 = patches.Rectangle((0.499, 0), 0.014, 0.82, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect7)
# rect8 = patches.Rectangle((0.513, 0), 0.018, 0.74, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect8)
# rect9 = patches.Rectangle((0.531, 0), 0.019, 0.70, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect9)
# rect10 = patches.Rectangle((0.550, 0), 0.0215, 0.74, linewidth=2, edgecolor='blue', facecolor='none', linestyle ="--")
# my_kde.add_patch(rect10)

##############################################################################
##Step 2.3 Beanplot
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Paramter 1
actual_value = 0.17

##Paramter 2
my_data_vec = X

##Paramter 3
quantile_vec = []
for i in np.arange(0,101,1):
    quantile_num = i/100
    quantile_value = np.quantile(my_data_vec, quantile_num, axis=0)
    quantile_vec.append(quantile_value)   

feature_name = "Weight"
print(quantile_vec)
quantile_vec_diff = np.diff(quantile_vec)
print(quantile_vec_diff)
len_interval = 1/len(quantile_vec)
density_vec = [len_interval/x for x in quantile_vec_diff]
print(density_vec)
len_mark = np.max(density_vec)/2 #length of black line
len_basic = (np.max(quantile_vec)-np.min(quantile_vec))/20 #length of blue line 
len_box = len_basic*6
density_vec.append(0)

##Scale density vec to fit the variable


print(density_vec)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.ylabel("behavioural feature: "+feature_name)

data_to_plot = pd.DataFrame()
data_to_plot["x_axis"] = quantile_vec
data_to_plot["y_axis"] = density_vec
data_to_plot["z_axis"] = [-x for x in density_vec]

# plt.plot(data_to_plot["y_axis"],data_to_plot["x_axis"],color="orange")
# plt.plot(data_to_plot["z_axis"],data_to_plot["x_axis"],color="orange")
plt.fill(np.append(data_to_plot["y_axis"], data_to_plot["z_axis"][::-1]),
         np.append(data_to_plot["x_axis"], data_to_plot["x_axis"][::-1]),
         color='red', alpha=0.6)


##Plot basic blue line segments
for quantile_value in quantile_vec:
    plt.plot([-len_basic, len_basic], [quantile_value, quantile_value], color="blue")
  
    
##Plot terget object   
plt.plot([-len_mark,len_mark], [actual_value, actual_value], color="black", alpha=0.5)


##Plot quartiles
q25 = np.quantile(my_data_vec, 0.25, axis=0)
q50 = np.quantile(my_data_vec, 0.50, axis=0)
q75 = np.quantile(my_data_vec, 0.75, axis=0)

plt.plot([-len_box, len_box], [q25, q25], color="cyan")
plt.plot([-len_box, len_box], [q50, q50], color="cyan")
plt.plot([-len_box, len_box], [q75, q75], color="cyan")
plt.plot([-len_box, -len_box], [q25, q75], color="cyan") #connecting them with a box
plt.plot([len_box, len_box], [q25, q75], color="cyan") #connecting them with a box

##Add comments
# plt.plot([len_box, len_box*20], [q75, q75*1.3], linestyle='dashed', color="pink") 
# plt.text(len_box*20, q75*1.3, r"Q3", fontsize=10)

# plt.plot([len_box, len_box*20], [q50, q50*1.3], linestyle='dashed', color="pink") 
# plt.text(len_box*20, q50*1.3, r"Q2", fontsize=10)
# plt.plot([len_box, len_box*20], [q25, q25*1.3], linestyle='dashed', color="pink") 
# plt.text(len_box*20, q25*1.3, r"Q1", fontsize=10)
