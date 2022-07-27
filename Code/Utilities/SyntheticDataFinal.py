#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 16:27:57 2022


@author: zlifr

-Description1: \cite{Calikus2021 ArXiv}
The synthetic dataset is generated as the mixtures of multivariate Gaussians using the CAD model (Song et al., 2007). 
The number of contextual attributes and behavioral attributes is chosen to be 5. The number of Gaussians for contextual 
and behavioral attributes is also set to be 5. The centroids of the Gaussians are randomly generated. The diagonal 
of the covariance matrix is set to 1/4 of the average distance between the centroids in each dimension. 
In total, we sampled 25, 000 data points and injected 1% contextual anomalies using the perturbation scheme.

"""
######################################################################################################################
######################################################################################################################
##step1 write a function to generate synthetic datsets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

random.seed(42)
np.random.seed(42)

def GenSynDataset(num_con, num_con_cat, num_behave, sample_size, num_gaussian, my_scheme = "S3"):
    
    """Internal function to generate data samples.
    Parameters
    ----------
    num_con : int
        the number of contextual attributes
    num_con : int
        the number of categorical contextual attributes
    num_behave : int
        the number of contextual attributes
    sample_size : int
        the number of rows of generated data set
    num_gaussian : int
        the number of gaussians used to generate data
    my_scheme : choice from {"S1","S2","S3","S4","S5"}
        the scheme used to generate dataset of different forms of dependencies
        
    Returns
    -------
    syn_data_set : synthetic dataframe of shape (contextual attributes, behavioral attribute, lable),
    which contains only inliers of gaussian 
    
    """
    ###############################################################################
    
    syn_data_set = pd.DataFrame()
    
    ##Step1 generate numeric contextual attributes
    num_con_numeric = num_con-num_con_cat
    for i in range(num_con_numeric):
        con_num = "con_num" + str(i) #attribue name
        attr_mean = np.random.uniform(low=0, high=1, size=(num_gaussian,1))
        # attr_mean = np.random.uniform(low=1200, high=1300, size=(num_gaussian,1))
        
        ##calculate variance in a dimension
        var_distance = 0
        for j in range(num_gaussian):
            for k in range(j+1, num_gaussian):
                var_distance = var_distance + abs(attr_mean[k]-attr_mean[j])
        # attr_var = var_distance/(num_gaussian-1)
        attr_var = var_distance/4
        
        attr_value = []
        
        size_each_gaussian = int(sample_size/num_gaussian)
        
        for gau_item in range(num_gaussian):
            new_attr_value = np.random.normal(attr_mean[gau_item], attr_var, size = size_each_gaussian)
            new_attr_value = new_attr_value.tolist()
            attr_value = attr_value + new_attr_value
            # attr_value.append(new_attr_value)
 
        syn_data_set[con_num] = attr_value

 
    ##Step2 generate categorical contextual attributes

    for r in range(num_con_cat):
        con_num_cat = "con_num_cat" + str(r) #attribue name
        attr_mean_cat = np.random.uniform(low=1, high=10, size=(num_gaussian,1))
        
        ##calculate variance in a dimension
        var_distance_cat = 0
        for s in range(num_gaussian):
            for t in range(s+1, num_gaussian):
                var_distance_cat = var_distance_cat + abs(attr_mean_cat[s]-attr_mean_cat[t])
        # attr_var_cat = var_distance_cat/(num_gaussian-1)
        attr_var_cat = var_distance_cat/4
        
        attr_value_cat = []
        
        size_each_gaussian = int(sample_size/num_gaussian)
        
        for gau_item in range(num_gaussian):
            lowerbound = max(0,int(attr_mean_cat[gau_item]-attr_var_cat))
            upperbound = min(20,int(attr_mean_cat[gau_item]+attr_var_cat))
            
            if lowerbound == upperbound:
                upperbound = upperbound+1
                
            new_attr_value_cat = np.random.randint(lowerbound, upperbound, size = size_each_gaussian)
            new_attr_value_cat = new_attr_value_cat.tolist()
            
            attr_value_cat = attr_value_cat + new_attr_value_cat
 
        syn_data_set[con_num_cat] = attr_value_cat  
       
    ##Step3 generate behavioral attributes
    for l in range(num_behave):
        behave_num = "behave_num" + str(l) #attribue name
        
        attr_value_behave =  list(np.zeros(sample_size))
        print("-------------------------------------------------------------")
        print(behave_num)
        
        for ind_num in range(num_con_numeric):
            
            print("-----------------------")
            ##generate a random number from U(0,1)
            coff_ind_num = random.uniform(0, 1)
            print(coff_ind_num)
            
            ##generate a random number to replace the coff_ind_num by zero
            indicator_censor = random.uniform(0, 1)
            print(indicator_censor)
            if indicator_censor > 2/3:
                coff_ind_num = 0
               
            print(coff_ind_num)   
            ##add this column 
            con_num = "con_num" + str(ind_num) #attribue name
            
            if my_scheme == "S1" :
                attr_value_behave += coff_ind_num*syn_data_set[con_num]
            elif my_scheme == "S2" :
                attr_value_behave += coff_ind_num*np.power(syn_data_set[con_num],3)
            elif my_scheme == "S3" :
                attr_value_behave += coff_ind_num*np.sin(syn_data_set[con_num])
            elif my_scheme == "S4" :
                attr_value_behave += coff_ind_num*np.log(1+abs(syn_data_set[con_num]))
                print(attr_value_behave)
            else:
                ##We need to generate 5 coeffients
                coff_ind_num1 = random.uniform(0, 1)
                coff_ind_num2 = random.uniform(0, 1)
                coff_ind_num3 = random.uniform(0, 1)
                coff_ind_num4 = random.uniform(0, 1)
                
                ##generate a random number to replace the coff_ind_num by zero
                indicator_censor = random.uniform(0, 1)
                if indicator_censor > 2/3:
                    coff_ind_num1 = 0
                    coff_ind_num2 = 0
                    coff_ind_num3 = 0
                    coff_ind_num4 = 0
                    
                attr_value_behave += coff_ind_num1*syn_data_set[con_num] +\
                            coff_ind_num2*np.power(syn_data_set[con_num],3) +\
                            coff_ind_num3*np.sin(math.pi*syn_data_set[con_num]) +\
                            coff_ind_num4*np.log(1+abs(syn_data_set[con_num]))
            
            print("-----------------------")
            
        for ind_cat in range(num_con_cat):
            
            print("-----------------------")
            ##generate a random number from U(0,1)
            coff_ind_cat= random.uniform(0, 1)
            print(coff_ind_cat)
            
            ##generate a random number to replace the coff_ind_num by zero
            indicator_censor = random.uniform(0, 1)
            print(indicator_censor)
            if indicator_censor > 2/3:
                coff_ind_cat = 0
               
            print(coff_ind_cat)   
            ##add this column 
            con_cat = "con_num_cat"  + str(ind_cat) #attribue name
            
            if my_scheme == "S1" :
                attr_value_behave += coff_ind_cat*syn_data_set[con_cat]
            elif my_scheme == "S2" :
                attr_value_behave += coff_ind_cat*np.power(syn_data_set[con_cat],3)
            elif my_scheme == "S3" :
                attr_value_behave += coff_ind_cat*np.sin(syn_data_set[con_cat])
            elif my_scheme == "S4" :
                attr_value_behave += coff_ind_cat*np.log(1+abs(syn_data_set[con_cat]))
            else:
                ##We need to generate 5 coeffients
                coff_ind_cat1 = random.uniform(0, 1)
                coff_ind_cat2 = random.uniform(0, 1)
                coff_ind_cat3 = random.uniform(0, 1)
                coff_ind_cat4 = random.uniform(0, 1)
                
                ##generate a random number to replace the coff_ind_num by zero
                indicator_censor = random.uniform(0, 1)
                if indicator_censor > 2/3:
                    coff_ind_cat1 = 0
                    coff_ind_cat2 = 0
                    coff_ind_cat3 = 0
                    coff_ind_cat4 = 0
                    
                attr_value_behave += coff_ind_cat1*syn_data_set[con_cat] +\
                            coff_ind_cat2*np.power(syn_data_set[con_cat],3) +\
                            coff_ind_cat3*np.sin(syn_data_set[con_cat]) +\
                            coff_ind_cat4*np.log(1+abs(syn_data_set[con_cat]))

            print("-----------------------")
        
        ##Generate epsilon as noises
        epsilon_vec = list(np.random.uniform(0, 0.05, sample_size))
        print(attr_value_behave)
        # syn_data_set[behave_num] = attr_value_behave + epsilon_vec
        syn_data_set[behave_num] = [a + b for a, b in zip(attr_value_behave.values, epsilon_vec)]
    return syn_data_set


##test 


# ## SynDataSet 1
# MyDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S1")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv", sep=',')
# ## SynDataSet 2
# MyDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S2")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet2.csv", sep=',')
# ## SynDataSet 3
# MyDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S3")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet3.csv", sep=',')
# ## SynDataSet 4
# MyDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S4")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet4.csv", sep=',')
# ## SynDataSet 5
# MyDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S5")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet5.csv", sep=',')

# ## SynDataSet 6
# MyDataSet = GenSynDataset(num_con =20 , num_con_cat = 6, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S1")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet6.csv", sep=',')
# ## SynDataSet 7
# MyDataSet = GenSynDataset(num_con =20 , num_con_cat = 6, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S2")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet7.csv", sep=',')
# ## SynDataSet 8
# MyDataSet = GenSynDataset(num_con =20 , num_con_cat = 6, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S3")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet8.csv", sep=',')
# ## SynDataSet 9
# MyDataSet = GenSynDataset(num_con =20 , num_con_cat = 6, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S4")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet9.csv", sep=',')
# ## SynDataSet 10
# MyDataSet = GenSynDataset(num_con =20 , num_con_cat = 6, num_behave = 5,
#                           sample_size = 2000, num_gaussian = 5, my_scheme = "S5")
# MyDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet10.csv", sep=',')

######################################################################################################################
######################################################################################################################
##step2 use ContextualAnomaltGene script to inject outliers

# from ContextualAnomalyGene import GenerateData

# # # num_con_value = 3
# # # num_con_cat_value = 1
# # # num_behave_value = 10


# num_con_value = 5
# num_con_cat_value = 2
# num_behave_value = 5
# sample_size_value = 1000
# num_gaussian_value = 5
# anomaly_ratio = 0.1

# anomaly_number  = int(anomaly_ratio*sample_size_value)

# RawDataSet = GenSynDataset(num_con = num_con_value , num_con_cat = num_con_cat_value, 
#                             num_behave = num_behave_value, sample_size = sample_size_value, num_gaussian = num_gaussian_value)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset1.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset1.csv"

# AllCols = RawDataSet.columns.tolist()

# ContextCols = AllCols[0:num_con_value]

# BehaveCols = AllCols[num_con_value:]

# NumCols = AllCols[0:(num_con_value-num_con_cat_value)] + BehaveCols

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, anomaly_number) 

# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset1Gene.csv", sep=',')


######################################################################################################################
######################################################################################################################
##step3 test accuracy when using different number of neighbours and algorithms 

# from ICAD_sensitivity import SensitivityAnalysisOfK


# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset1Gene.csv"
# SaveFilePath1 = r"/Users/zlifr/Desktop/HHBOS/SynResult/Dataset1ResultMean1.csv"
# SaveFilePath2= r"/Users/zlifr/Desktop/HHBOS/SynResult/Dataset1ResultMean2.csv"


# AllColsWithTruth = AllCols + ['ground_truth']

# anomaly_value = anomaly_number

# sample_value = anomaly_number

# depth_value = 1

# SaveFilePath = SaveFilePath1

# alpha_value = 10

# min_k = 15

# max_k = 200

# step_k = 5

# num_dataset = 10

# SensitivityAnalysisOfK(FilePath, ResultFilePath, SaveFilePath,
#                         AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
#                         anomaly_value, sample_value, depth_value, alpha_value,
#                         min_k, max_k, step_k, num_dataset)