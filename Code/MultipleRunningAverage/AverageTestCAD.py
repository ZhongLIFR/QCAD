#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:52:48 2021

@author: zlifr
"""


##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub'  


import warnings
warnings.filterwarnings("ignore")

import sys
ImpleDir = AbsRootDir+ r'/QCAD/Code/Implementation'
UtilityDir = AbsRootDir+ r'/QCAD/Code/Utilities'
sys.path.append(ImpleDir)
sys.path.append(UtilityDir)

from ContextualAnomalyInject import GenerateData
from CAD import CAD

# =============================================================================
# #Step1. Define a function to calculate anomaly score based on given datasets
# =============================================================================

##def a function to do sensitivity analysis
def AverageTestCAD(FilePath,
                   ResultFilePath, 
                   SaveFilePath_CAD,
                   MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                   NumCols, anomaly_value, sample_value, neighbour_value,
                   num_dataset):
    
    import pandas as pd
    myResult_CAD =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])

   
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
        MyBehaveList = BehaveCols
           
        #for each dataset, calculate score
        
        ##This is for CAD
        is_model_learned = 0
        FilePath_MappingMatrix = AbsRootDir + r'/QCAD/Data/TempFiles/abalone_CAD_mapping_matrix.npy'
        num_gau_comp = 5
        alpha_log = 0.001
        
        my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3 = CAD(MyDataSet, MyContextList, MyBehaveList, num_gau_comp, alpha_log, is_model_learned, FilePath_MappingMatrix)
        myResult_CAD.loc[len(myResult_CAD)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]


    
    ## For CAD
    print("CADstart")
    myResult2 = myResult_CAD.copy()
    myResult2['pr_auc_value_mean'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['roc_auc_value_mean'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['p_at_n_value_mean'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration1_mean'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('mean')
    myResult2['duration3_mean'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('mean')
    
    myResult2['pr_auc_value_std'] = myResult2['pr_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['roc_auc_value_std'] = myResult2['roc_auc_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['p_at_n_value_std'] = myResult2['p_at_n_value'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration1_std'] = myResult2['duration1'].groupby(myResult2['neighbour_value']).transform('std')
    myResult2['duration3_std'] = myResult2['duration3'].groupby(myResult2['neighbour_value']).transform('std')
        
    myResult_ICAD = myResult2.drop_duplicates()
    myResult_ICAD.to_csv(SaveFilePath_CAD, sep=',')
    
# =============================================================================
# Step 2. Example ->  Abalone
# =============================================================================

###############################################################################
## Step 2.1. Parameter setting
###############################################################################

FilePath = AbsRootDir + r'/QCAD/Data/RawData/abalone.csv'
ResultFilePath = AbsRootDir + r'/QCAD/Data/TempFiles/abaloneGene.csv'
SaveFilePath_CAD = AbsRootDir + r'/QCAD/Data/TempFiles/abalone_CAD.csv'

AllCols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
            'Viscera weight', 'Shell weight', 'Rings']

AllColsWithTruth = ['Sex', 'Length', 'Diameter', 'Height',
                    'Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings',
                    'ground_truth']

ContextCols = ['Sex', 'Length', 'Diameter', 'Height']

BehaveCols = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

NumCols = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

anomaly_value = 100

sample_value = 100

neighbour_value = 500

num_dataset = 5
    

###############################################################################
## Step 2.2. Execute the function
###############################################################################

AverageTestCAD(FilePath,
               ResultFilePath,
               SaveFilePath_CAD,
               AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
               anomaly_value, sample_value, neighbour_value, num_dataset)
    
