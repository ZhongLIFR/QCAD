#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to generate the BeanPlot example

Created on Mon Dec 27 09:14:46 2021

@author: zlifr, z.li@liacs.leidenuniv.nl
"""

##############################################################################
##Step1. generate synthetic data 
##############################################################################

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


##############################################################################
##Step2. generate BeanPlot
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
