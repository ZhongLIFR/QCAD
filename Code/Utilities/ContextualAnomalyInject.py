#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the script used to generate contextual anomalies.

Created on Wed Dec 29 22:33:52 2021

@author: zlifr
"""


"""
A de-facto mechanism for injecting contextual anomalies is presented in \cite{song2007conditional}. 
However, \cite{song2007conditional} has critised this mechanism as follows:

"We do not follow this scheme for several reasons. First, swapping the attribute values may not always
obtain desired outliers. It is likely that most of the swaps could result in normal data. 
Second, as we observe many extreme outliers in the real-world datasets, swapping values between 
samples in a clean data is less likely to produce this extreme difference between yi and yj. 
Here we present another way to generate outliers and we explore different types of outliers 
where we give controls to where and how many outliers are injected or its degree of outlierness."

Besides, they propose the following new mechanism for injecting contextual anomalies 

"To inject q×N outliers into a dataset with N data samples, we randomly select q×N records z®i = ( ®xi
,yi)to be perturbed. Let yi be the target attribute for perturbation.Let x®i be the rest of attributes. 
For all selected records, a random number from (0, α) is added up to yi as y′i. Then we add new
sample z®′ = ( ®xi,y′i) into the original dataset and flag it as outlier. Note that original N data 
samples are flagged as non-outlier. In the experiments, we standardized the target attribute to range (18,
30) which are the min and max value of the behavioral attribute in Elnino dataset. Set α as 50 by default."

However, as pointed out in our paper, Guo's method also has limitations and we refine it as follows. 
(See our QCAD paper for more detail)

"""


# =============================================================================
# ##Step1. define a function to inject outliers and output result as a csv file
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy
import random

def GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, OutlierNum = 50, RandomState = 42):
    """
    Parameters
    ----------
    FilePath : string
        the location of dataset without contextual anomalies.
    AllCols : list
        the list of all feature names.
    ContextCols : list
        the list of contextual feature names.
    BehaveCols : list
        the list of behavioural feature names..
    NumCols : list
        the list of contextual feature names where the feratures are numerical.
    OutlierNum : int, optional
        the number of injected outliers. The default is 50.
    RandomState : TYPE, optional
        the value of random seeds. The default is 42.

    Returns
    -------
    MyDataSet : dataframe
        dataset containing injected contextual anomalies.

    """

    ###############################################################################
    ###############################################################################
    ##Step1.1 load dataset and description analysis
    
    
    numpy.random.seed(RandomState)
    random.seed(RandomState)
    
    RawDataSet = pd.read_csv(FilePath, sep=",")
    
    RawDataSet = RawDataSet.dropna()  #remove missing values
    
    ##downsampling strategy for elnino datasetlen
    if RawDataSet.shape[0]>20000:
        print("downsampling")
        RawDataSet = RawDataSet.sample(n=20000)
    
    MyDataSet = RawDataSet[AllCols]
            
    ###############################################################################
    ###############################################################################
    ##Step1.2 inject contextual outliers 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    MyScaler = MinMaxScaler()
    
    Behavioural_list = BehaveCols
    
    ##Only scale Behavioural attributes
    # MyDataSet[Behavioural_list] = MyScaler.fit_transform(MyDataSet[Behavioural_list])
    
    ##Scale all numerical attributes
    if len(NumCols) > 0:
        MyDataSet[NumCols] = MyScaler.fit_transform(MyDataSet[NumCols])
    
    ##we should min_max all numerical attributes
    
    inject_data = pd.DataFrame()
    
    random.seed(RandomState)
    
    random_indices = random.sample(range(1, MyDataSet.shape[0]), OutlierNum)
    
    print(random_indices)
    
    for num_index in random_indices:
        print(num_index)  
    
        alpha_value = 0.5
        
        for col_index in Behavioural_list:
            luck_number = random.choice([-1,1])* random.uniform(0.1, alpha_value)
            MyDataSet[col_index].iloc[num_index] = MyDataSet[col_index].iloc[num_index] + luck_number
            # MyDataSet[col_index].clip(lower=0)
            # MyDataSet[col_index].clip(upper=1)
            
        print(MyDataSet[Behavioural_list].iloc[num_index])
       
    MyDataSet["ground_truth"] = 0
    
    for num_index in random_indices:
        MyDataSet["ground_truth"].iloc[num_index] = 1
    
    
    return MyDataSet


# =============================================================================
# ##Step2. apply above defined function to generate injected contextual outliers
# =============================================================================


import pandas as pd
import numpy as np

##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub' 


###############################################################################
##An example, please uncomment the following code
###############################################################################

# RawDataSetPath = AbsRootDir+r'/QCAD/Data/RawData/abalone.csv'
# RawDataSet = pd.read_csv(RawDataSetPath, sep=",") #it contains 4177 rows
# RawDataSet = RawDataSet.dropna()
# FilePath0 = r"/Users/zlifr/Desktop/HHBOS/Data/abalone.csv"
# AllCols0 = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
#             'Viscera weight', 'Shell weight', 'Rings']

# ContextCols0 = ['Sex', 'Length', 'Diameter', 'Height']
# BehaveCols0 = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']
# NumCols0 = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

# FinalDataSet0 = GenerateData(FilePath0, AllCols0, ContextCols0, BehaveCols0, NumCols0,  418)
# GenDataSetPath = AbsRootDir+r'/QCAD/Data/GenData/abaloneGene2.csv'
# FinalDataSet0.to_csv(GenDataSetPath, sep=',')

