#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 07:55:50 2021

@author: zlifr
"""

##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub'  


# =============================================================================
# #Step1 Import related modules
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import percentile

import sys
ImpleDir = AbsRootDir+ r'/QCAD/Code/Implementation'
UtilityDir = AbsRootDir+ r'/QCAD/Code/Utilities'
sys.path.append(ImpleDir)
sys.path.append(UtilityDir)

from ContextualAnomalyInject import GenerateData
from QCAD import QCAD


from ContextualAnomalyInject import GenerateData
from QCAD import QCAD




# =============================================================================
# #Step2 Define a function to do sensitivity analysis
# =============================================================================


def SensitivityAnalysisOfK(FilePath,
                           ResultFilePath,
                           SaveFilePath,
                           AllCols,
                           AllColsWithTruth,
                           ContextCols,
                           BehaveCols,
                           NumCols,
                           anomaly_value,
                           sample_value,
                           min_k,
                           max_k,
                           step_k,
                           num_dataset):
    """
    

    Parameters
    ----------
    FilePath : string
        the path of raw dataset.
    ResultFilePath : string
        the path of resulting dataset.
    MB_dataset_path : string
        the path of MB used in LoPAD.
    MyColList : list
        the list of all feature names.
    AllColsWithTruth : list
        the list of all feature names and ground-truth.
    MyContextList : list
        the list of contextual feature names.
    MyBehaveList : list
       the list of behavioural feature names.
    NumCols : list
        the list of numerical contextual feature names.
    anomaly_value : int
        the number of anomalies.
    sample_value : int
        the number of anomalies.
    min_k : int
        the minimal number of neighbours.
    max_k : int
        the maximal number of neighbours.
    step_k : int
        stepwidth of increasing the number ofneighbours.
    num_dataset : int
        the number of trials.

    Returns
    -------
    None.

    """
    
    import pandas as pd
    myResult =  pd.DataFrame(columns = ["neighbour_value","pr_auc_value","roc_auc_value","p_at_n_value","duration1","duration3", "data_set"])
    #generate different datasets by using different random state
    for MyRandomState in range(42,42+num_dataset):
        
        FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()  #remove missing values
        
        MyDataSet = FinalDataSet[AllColsWithTruth]
        
        MyContextDataSet = MyDataSet[ContextCols]
        MyBehaveDataSet = MyDataSet[BehaveCols]
                
        #for each dataset, calculate score
        for k_value in range(min_k, max_k, step_k):
            
            neighbour_value = k_value 
                
            my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, TempDataSet = QCAD(MyDataSet,
                                                                                            AllColsWithTruth,
                                                                                            ContextCols,
                                                                                            BehaveCols, 
                                                                                            neighbour_value,
                                                                                            sample_value)
        
  
            myResult.loc[len(myResult)] = [neighbour_value, my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, MyRandomState]
    
    myResult2 = myResult.copy()
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
    
    del myResult2["pr_auc_value"]
    del myResult2["roc_auc_value"]
    del myResult2["p_at_n_value"]
    del myResult2["duration1"]
    del myResult2["duration3"]
    del myResult2["data_set"]
    
    myResult2 = myResult2.drop_duplicates()
    
    myResult2.to_csv(SaveFilePath, sep=',')
    
    ###############################################################################
    ##beautiful plots of performance metrics
    import seaborn as sns
    import pandas as pd
    
    sns.set(style="white")
    
    x = myResult2["neighbour_value"]
    mean_1 = myResult2["pr_auc_value_mean"]
    std_1 = myResult2["pr_auc_value_std"]
    
    mean_2 = myResult2["roc_auc_value_mean"]
    std_2 = myResult2["roc_auc_value_std"]
    
    mean_3 = myResult2["p_at_n_value_mean"]
    std_3 = myResult2["p_at_n_value_std"]
    
    plt.plot(x, mean_2, 'g--', label='ROC AUC', marker='.')
    plt.ylim((-0.1, 1.1)) 
    plt.xlabel("number of neighbours")
    plt.ylabel("performance metrics")
    
    plt.title(r'Sensitivity of $k$ on %i datasets' % num_dataset)
    plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='g', alpha=0.2)
    plt.plot(x, mean_1, 'b--', label='PR AUC', marker='*')
    plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
    plt.plot(x, mean_3, 'r--', label='Precision@n', marker='^')
    plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='r', alpha=0.2)
    
    plt.legend()
    plt.show()
    

# =============================================================================
# ##Step3 Example ->  Toxicity Dataset, Uncomment the following code 
# =============================================================================


# FilePath =  AbsRootDir+r'/QCAD/Data/RawData/Toxicity.csv'
# ResultFilePath = AbsRootDir+r'/QCAD/Data/TempFiles/ToxicityGene.csv'
# SaveFilePath = AbsRootDir+r'/QCAD/Data/TempFiles/Toxicity_QCADTest.csv'

# AllCols = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50"]

# AllColsWithTruth = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50",
#                     'ground_truth']

# ContextCols = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP"]

# BehaveCols = ['LC50']

# NumCols =  ['LC50']

# anomaly_value = 90

# sample_value = 90

# min_k = 10

# max_k = 908

# step_k = 20

# num_dataset = 10


# SensitivityAnalysisOfK(FilePath,
#                         ResultFilePath,
#                         SaveFilePath,
#                         AllCols,
#                         AllColsWithTruth,
#                         ContextCols,
#                         BehaveCols,
#                         NumCols, 
#                         anomaly_value,
#                         sample_value,
#                         min_k,
#                         max_k,
#                         step_k,
#                         num_dataset)
    



