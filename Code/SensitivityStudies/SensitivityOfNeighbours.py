#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 07:55:50 2021

@author: zlifr
"""

import warnings
warnings.filterwarnings("ignore")

###############################################################################
###############################################################################
##Step1 define a function to calculate anomaly score 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import percentile


###############################################################################
###############################################################################
##Step2 generate different dataset to use
from ContextualAnomalyInject import GenerateData
from ICAD_QRF import ICAD_QRF


###############################################################################
##DataSet1:  SynDataSet4
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/SynDataSet1.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1Gene.csv"

# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/SynTempData/SynDataSet1_ICADtest.csv" ##Too slow!


# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# AllColsWithTruth = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#                     'con_num_cat0', 'con_num_cat1',
#                     'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#                     'ground_truth']

# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3', 'con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# anomaly_value = 100

# sample_value = 100

# min_k = 10

# max_k = 1000

# step_k = 50

# num_dataset = 5


###############################################################################
##DataSet2:  ForestFires
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/forestFires.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFiresGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/forestFires_ICADtest.csv"

# AllCols = ['X', 'Y', 'month', 'day', 
#             'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# AllColsWithTruth = ['X', 'Y', 'month', 'day', 
#                     'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area',
#                     'ground_truth']

# ContextCols = ['X', 'Y', 'month', 'day']

# BehaveCols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# NumCols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# anomaly_value = 52

# sample_value = 52

# min_k = 10

# max_k = 510

# step_k = 20

# num_dataset = 5


###############################################################################
##DataSet3:  Energy
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/energy.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/energyGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/energy_ICADtest.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/energyGene.csv"

# AllCols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']

# AllColsWithTruth = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2',
#                     'ground_truth']

# ContextCols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# BehaveCols = ['Y1', 'Y2']

# NumCols =  ['Y1', 'Y2']

# anomaly_value = 70

# sample_value = 70

# min_k = 10

# max_k = 700

# step_k = 20

# num_dataset = 10



###############################################################################
##DataSet4:  heartFailure
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/heartFailure.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailureGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/heartFailure_ICADtest.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/heartFailureGene.csv"

# AllCols = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia",
#             "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time",
#             "DEATH_EVENT"]

# AllColsWithTruth = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia",
#                     "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time",
#                     "DEATH_EVENT",
#                     'ground_truth']

# ContextCols = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia"]

# BehaveCols = ["creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

# NumCols = ["creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

# anomaly_value = 30

# sample_value = 30

# min_k = 10

# max_k = 300

# step_k = 10

# num_dataset = 10

###############################################################################
##DataSet5:  hepatitis to do
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/hepatitis.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitisGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/hepatitis_ICADtest.csv"
# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/hepatitisGene.csv"


# AllCols = ['Category', 'Age', 'Sex', 
#             'ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# AllColsWithTruth = ['Category', 'Age', 'Sex', 
#                     'ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT',
#                     'ground_truth']

# ContextCols = ['Category', 'Age', 'Sex']

# BehaveCols = ['ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# NumCols =  ['ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# anomaly_value = 59

# sample_value = 59

# min_k = 10

# max_k = 610

# step_k = 20

# num_dataset = 5


###############################################################################
##DataSet6:  indianLiverPatient
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/indianLiverPatient.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatientGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/indianLiverPatient_ICADtest.csv"

# MB_dataset_path = r"/Users/zlifr/Desktop/HHBOS/Data3/MB/indianLiverPatientGene.csv"


# AllCols = ['Age', 'Gender', 'Selector',
#             'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# AllColsWithTruth = ['Age', 'Gender', 'Selector',
#                     'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#                     'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio',
#                     'ground_truth']

# ContextCols = ['Age', 'Gender', 'Selector']

# BehaveCols = ['Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#               'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# NumCols =  ['Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# anomaly_value = 58

# sample_value = 58

# min_k = 10

# max_k = 570

# step_k = 20

# num_dataset = 5


###############################################################################
##DataSet7:  QSRanking 
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/QSRanking.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRankingGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/QSRanking_ICADtest.csv"

# AllCols = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
#             'SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
#             'Academic Reputation', 'Employer Reputation', 'Faculty Student',
#             'Citations per Faculty', 'International Faculty',
#             'International Students']

# AllColsWithTruth = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
#                     'SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
#                     'Academic Reputation', 'Employer Reputation', 'Faculty Student',
#                     'Citations per Faculty', 'International Faculty',
#                     'International Students',
#                     'ground_truth']

# ContextCols = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country']

# BehaveCols = ['Academic Reputation', 'Employer Reputation', 'Faculty Student',
#               'Citations per Faculty', 'International Faculty',
#               'International Students']

# NumCols = ['Academic Reputation', 'Employer Reputation', 'Faculty Student',
#             'Citations per Faculty', 'International Faculty', 'International Students']

# anomaly_value = 48

# sample_value = 48

# min_k = 10

# max_k = 470

# step_k = 20

# num_dataset = 5


###############################################################################
##DataSet8:  synchronousMachine  
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/SynchronousMachine.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachineGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_ICADtest.csv"

# AllCols = ['Iy', 'PF', 'e', 'dIf', 'If']

# AllColsWithTruth = ['Iy', 'PF', 'e', 'dIf', 'If',
#                     'ground_truth']

# ContextCols = ['Iy', 'PF', 'e', 'dIf']

# BehaveCols = ['If']

# NumCols =  ['If']

# anomaly_value = 55

# sample_value = 55

# min_k = 10

# max_k = 550

# step_k = 10

# num_dataset = 10


###############################################################################
##DataSet9:  bodyfat
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Bodyfat.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/BodyfatGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/Bodyfat_ICADtest.csv"


# AllCols = ['Density', 'BodyFat', 
#             'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# AllColsWithTruth = ['Density', 'BodyFat', 
#                     'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
#                     'ground_truth']

# ContextCols = ['Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# BehaveCols = ['Density', 'BodyFat']

# NumCols = ['Density', 'BodyFat']

# anomaly_value = 30

# sample_value = 30

# min_k = 10

# max_k = 250

# step_k = 10

# num_dataset = 10

###############################################################################
##DataSet10:  boston
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/boston.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/bostonGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/boston_ICADtest.csv"

# AllCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

# AllColsWithTruth = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV',
#                     'ground_truth']

# ContextCols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']

# BehaveCols = ['MEDV']

# NumCols = ['MEDV']

# anomaly_value = 50

# sample_value = 50

# min_k = 10

# max_k = 500

# step_k = 20

# num_dataset = 10


###############################################################################
##DataSet11:  yachtHydrodynamics
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/YachtHydrodynamics.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamicsGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_ICADtest.csv"

# AllCols = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#             'Beam_draught_ratio', 'Length_beam_ratio','Froude_number', 
#             'resistance']

# AllColsWithTruth = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#                     'Beam_draught_ratio', 'Length_beam_ratio','Froude_number', 
#                     'resistance',
#                     'ground_truth']

# ContextCols = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#                 'Beam_draught_ratio', 'Length_beam_ratio','Froude_number']

# BehaveCols = ['resistance']

# NumCols = ['resistance']

# anomaly_value = 31

# sample_value = 31

# min_k = 30

# max_k = 300

# step_k = 5

# num_dataset = 10


###############################################################################
##DataSet12:  fish
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Fish.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/FishGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/Fish_ICADtest.csv"


# AllCols = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']

# AllColsWithTruth = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height','Width',
#                     'ground_truth']

# ContextCols = ['Species', 'Length1', 'Length2', 'Length3', 'Height','Width']

# BehaveCols = ['Weight']

# NumCols =  ['Weight']

# anomaly_value = 20

# sample_value = 20


# min_k = 5

# max_k = 150

# step_k = 5

# num_dataset = 10


###############################################################################
##DataSet13:  airfoil
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Airfoil.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/AirfoilGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_ICADtest.csv"


# AllCols = ['f', 'alpha', 'c', 'U_infinity', 'delta', 'SSPL']

# AllColsWithTruth = ['f', 'alpha', 'c', 'U_infinity', 'delta', 'SSPL',
#                     'ground_truth']

# ContextCols = ['f', 'alpha', 'c', 'U_infinity', 'delta']

# BehaveCols = ['SSPL']

# NumCols =  ['SSPL']

# anomaly_value = 150

# sample_value = 150


# min_k = 10

# max_k = 1500

# step_k = 20

# num_dataset = 5

###############################################################################
##DataSet14:  Concrete
###############################################################################
# FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Concrete.csv"
# ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/ConcreteGene.csv"
# SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_ICADtest.csv"

# AllCols = ["C1","C2","C3","C4","C5","C6","C7","Age","Strength"]

# AllColsWithTruth = ["C1","C2","C3","C4","C5","C6","C7","Age","Strength",
#                     'ground_truth']

# ContextCols = ["C1","C2","C3","C4","C5","C6","C7","Age",]

# BehaveCols = ['Strength']

# NumCols =  ['Strength']

# anomaly_value = 100

# sample_value = 100

# min_k = 10

# max_k = 1030

# step_k = 20

# num_dataset = 5


###############################################################################
##DataSet15:  Toxicity
###############################################################################
FilePath = r"/Users/zlifr/Desktop/HHBOS/Data/Toxicity.csv"
ResultFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/ToxicityGene.csv"
SaveFilePath = r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_ICADtest.csv"

AllCols = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50"]

AllColsWithTruth = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50",
                    'ground_truth']

ContextCols = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP"]

BehaveCols = ['LC50']

NumCols =  ['LC50']

anomaly_value = 90

sample_value = 90

min_k = 10

max_k = 908

step_k = 20

num_dataset = 10




###############################################################################
##def a function to do sensitivity analysis
###############################################################################

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
                
            my_pr_auc, my_roc_score, P_at_n_value, duration1, duration3, TempDataSet = ICAD_QRF(MyDataSet,
                                                                                                AllColsWithTruth,
                                                                                                ContextCols,
                                                                                                BehaveCols, 
                                                                                                neighbour_value,
                                                                                                anomaly_value,
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
# #execute the function
# =============================================================================
# SensitivityAnalysisOfK(FilePath,
#                        ResultFilePath,
#                        SaveFilePath,
#                        AllCols,
#                        AllColsWithTruth,
#                        ContextCols,
#                        BehaveCols,
#                        NumCols, 
#                        anomaly_value,
#                        sample_value,
#                        min_k,
#                        max_k,
#                        step_k,
#                        num_dataset)
    


#############################################################################################################################
#plot it by ourselfs
    
# import seaborn as sns

# myResult = pd.read_csv("/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_ICADtest.csv", sep=",")
# sns.set(style="white")

# x = myResult["neighbour_value"]
# mean_1 = myResult["pr_auc_value_mean"]
# std_1 = myResult["pr_auc_value_std"]

# mean_2 = myResult["roc_auc_value_mean"]
# std_2 = myResult["roc_auc_value_std"]

# mean_3 = myResult["p_at_n_value_mean"]
# std_3 = myResult["p_at_n_value_std"]

# plt.plot(x, mean_2, 'g--', label='ROC AUC', marker='.')
# plt.ylim((-0.1, 1.1)) 
# plt.xlabel("number of neighbours")
# plt.ylabel("performance metrics")

# plt.title(r'Sensitivity of $k$ on %i datasets' % num_dataset)
# plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='g', alpha=0.2)
# plt.plot(x, mean_1, 'b--', label='PR AUC', marker='*')
# plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
# plt.plot(x, mean_3, 'r--', label='Precision@n', marker='^')
# plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='r', alpha=0.2)

# plt.legend()
# plt.show()



###############################################################################     
##4.2.2 Display the results
###############################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

f, axes = plt.subplots(2, 3)

# =============================================================================
# Dataset1
# =============================================================================

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/TempData/Airfoil_ICADtest.csv",sep=",")

ax11 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="pr_auc_value_mean", palette = "hls",
                   marker="p", color='g', ax=axes[0,0], label="PRC ROC Mean & Std")
ax11.title.set_text('Air Foil')
ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 2})
ax11.get_legend().remove()
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)
ax11.xaxis.label.set_visible(False)
ax11.yaxis.label.set_text("Performance Metrics")
ax11.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["pr_auc_value_mean"] - RawDataSet["pr_auc_value_std"],
                 RawDataSet["pr_auc_value_mean"] + RawDataSet["pr_auc_value_std"], color='g', alpha=0.2)


ax12 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="roc_auc_value_mean", palette = "hls",
                   marker="s", color='r', ax=axes[0,0], label="ROC ROC Mean & Std")
ax12.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["roc_auc_value_mean"] - RawDataSet["roc_auc_value_std"],
                 RawDataSet["roc_auc_value_mean"] + RawDataSet["roc_auc_value_std"], color='r', alpha=0.2)
ax12.get_legend().remove()


ax13 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="p_at_n_value_mean", palette = "hls", marker="d",
                   color='b', ax=axes[0,0], label="P@n Mean & Std")
ax13.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["p_at_n_value_mean"] - RawDataSet["p_at_n_value_std"],
                 RawDataSet["p_at_n_value_mean"] + RawDataSet["p_at_n_value_std"],
                 color='b', alpha=0.2)
ax13.get_legend().remove()
# =============================================================================
# Dataset2
# =============================================================================
RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/TempData/energy_ICADtest.csv",sep=",")

ax21 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="pr_auc_value_mean", palette = "hls",
                   marker="p", color='g', ax=axes[0,1], label="PRC ROC Mean & Std")
ax21.title.set_text('Energy')
ax21.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 2})
ax21.get_legend().remove()
ax21.spines['right'].set_visible(False)
ax21.spines['top'].set_visible(False)
ax21.xaxis.label.set_visible(False)
ax21.yaxis.label.set_visible(False)
ax21.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["pr_auc_value_mean"] - RawDataSet["pr_auc_value_std"],
                 RawDataSet["pr_auc_value_mean"] + RawDataSet["pr_auc_value_std"], color='g', alpha=0.2)

ax22 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="roc_auc_value_mean", palette = "hls",
                   marker="s", color='r', ax=axes[0,1], label="ROC ROC Mean & Std")
ax22.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["roc_auc_value_mean"] - RawDataSet["roc_auc_value_std"],
                 RawDataSet["roc_auc_value_mean"] + RawDataSet["roc_auc_value_std"], color='r', alpha=0.2)
ax22.get_legend().remove()

ax23 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="p_at_n_value_mean", palette = "hls", marker="d",
                   color='b', ax=axes[0,1], label="P@n Mean & Std")
ax23.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["p_at_n_value_mean"] - RawDataSet["p_at_n_value_std"],
                 RawDataSet["p_at_n_value_mean"] + RawDataSet["p_at_n_value_std"],
                 color='b', alpha=0.2)
ax23.get_legend().remove()

# =============================================================================
# Dataset3
# =============================================================================
RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/TempData/Concrete_ICADtest.csv",sep=",")

ax31 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="pr_auc_value_mean", palette = "hls",
                   marker="p", color='g', ax=axes[0,2], label="PRC ROC Mean & Std")
ax31.title.set_text('Concrete')
ax31.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 6.3})
ax31.spines['right'].set_visible(False)
ax31.spines['top'].set_visible(False)
ax31.xaxis.label.set_visible(False)
ax31.yaxis.label.set_visible(False)

ax31.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["pr_auc_value_mean"] - RawDataSet["pr_auc_value_std"],
                 RawDataSet["pr_auc_value_mean"] + RawDataSet["pr_auc_value_std"], color='g', alpha=0.2)

ax32 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="roc_auc_value_mean", palette = "hls",
                   marker="s", color='r', ax=axes[0,2], label="ROC ROC Mean & Std")
ax32.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 6.3})
ax32.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["roc_auc_value_mean"] - RawDataSet["roc_auc_value_std"],
                 RawDataSet["roc_auc_value_mean"] + RawDataSet["roc_auc_value_std"], color='r', alpha=0.2)


ax33 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="p_at_n_value_mean", palette = "hls", marker="d",
                   color='b', ax=axes[0,2], label="P@n Mean & Std")
ax33.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["p_at_n_value_mean"] - RawDataSet["p_at_n_value_std"],
                 RawDataSet["p_at_n_value_mean"] + RawDataSet["p_at_n_value_std"],
                 color='b', alpha=0.2)
ax33.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 6.3})

# =============================================================================
# Dataset4
# =============================================================================

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/TempData/SynchronousMachine_ICADtest.csv",sep=",")

ax41 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="pr_auc_value_mean", palette = "hls",
                   marker="p", color='g', ax=axes[1,0], label="PRC ROC Mean & Std")
ax41.title.set_text('Synchronous Machine')
ax41.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 2})
ax41.get_legend().remove()
ax41.spines['right'].set_visible(False)
ax41.spines['top'].set_visible(False)
ax41.xaxis.label.set_text("Number of Neighbours")
ax41.yaxis.label.set_text("Performance Metrics")
ax41.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["pr_auc_value_mean"] - RawDataSet["pr_auc_value_std"],
                 RawDataSet["pr_auc_value_mean"] + RawDataSet["pr_auc_value_std"], color='g', alpha=0.2)


ax42 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="roc_auc_value_mean", palette = "hls",
                   marker="s", color='r', ax=axes[1,0], label="ROC ROC Mean & Std")
ax42.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["roc_auc_value_mean"] - RawDataSet["roc_auc_value_std"],
                 RawDataSet["roc_auc_value_mean"] + RawDataSet["roc_auc_value_std"], color='r', alpha=0.2)
ax42.get_legend().remove()


ax43 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="p_at_n_value_mean", palette = "hls", marker="d",
                   color='b', ax=axes[1,0], label="P@n Mean & Std")
ax43.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["p_at_n_value_mean"] - RawDataSet["p_at_n_value_std"],
                 RawDataSet["p_at_n_value_mean"] + RawDataSet["p_at_n_value_std"],
                 color='b', alpha=0.2)
ax43.get_legend().remove()

# =============================================================================
# Dataset5
# =============================================================================

RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/TempData/Toxicity_ICADtest.csv",sep=",")

ax51 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="pr_auc_value_mean", palette = "hls",
                   marker="p", color='g', ax=axes[1,1], label="PRC ROC Mean & Std")
ax51.title.set_text('Toxicity')
ax51.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 2})
ax51.get_legend().remove()
ax51.spines['right'].set_visible(False)
ax51.spines['top'].set_visible(False)
ax51.xaxis.label.set_text("Number of Neighbours")
ax51.yaxis.label.set_text("Performance Metrics")
ax51.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["pr_auc_value_mean"] - RawDataSet["pr_auc_value_std"],
                 RawDataSet["pr_auc_value_mean"] + RawDataSet["pr_auc_value_std"], color='g', alpha=0.2)


ax52 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="roc_auc_value_mean", palette = "hls",
                   marker="s", color='r', ax=axes[1,1], label="ROC ROC Mean & Std")
ax52.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["roc_auc_value_mean"] - RawDataSet["roc_auc_value_std"],
                 RawDataSet["roc_auc_value_mean"] + RawDataSet["roc_auc_value_std"], color='r', alpha=0.2)
ax52.get_legend().remove()


ax53 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="p_at_n_value_mean", palette = "hls", marker="d",
                   color='b', ax=axes[1,1], label="P@n Mean & Std")
ax53.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["p_at_n_value_mean"] - RawDataSet["p_at_n_value_std"],
                 RawDataSet["p_at_n_value_mean"] + RawDataSet["p_at_n_value_std"],
                 color='b', alpha=0.2)
ax53.get_legend().remove()


# =============================================================================
# Dataset6
# =============================================================================
RawDataSet = pd.read_csv(r"/Users/zlifr/Desktop/HHBOS/TempData/YachtHydrodynamics_ICADtest.csv",sep=",")

ax61 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="pr_auc_value_mean", palette = "hls",
                   marker="p", color='g', ax=axes[1,2], label="PRC ROC Mean & Std")
ax61.title.set_text('Yacht')
ax61.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 6.3})
ax61.spines['right'].set_visible(False)
ax61.spines['top'].set_visible(False)
ax61.xaxis.label.set_text("Number of Neighbours")
ax61.yaxis.label.set_visible(False)

ax61.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["pr_auc_value_mean"] - RawDataSet["pr_auc_value_std"],
                 RawDataSet["pr_auc_value_mean"] + RawDataSet["pr_auc_value_std"], color='g', alpha=0.2)

ax62 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="roc_auc_value_mean", palette = "hls",
                   marker="s", color='r', ax=axes[1,2], label="ROC ROC Mean & Std")
ax62.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 6.3})
ax62.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["roc_auc_value_mean"] - RawDataSet["roc_auc_value_std"],
                 RawDataSet["roc_auc_value_mean"] + RawDataSet["roc_auc_value_std"], color='r', alpha=0.2)


ax63 = sns.lineplot(data=RawDataSet, x="neighbour_value", y="p_at_n_value_mean", palette = "hls", marker="d",
                   color='b', ax=axes[1,2], label="P@n Mean & Std")
ax63.fill_between(RawDataSet["neighbour_value"],
                 RawDataSet["p_at_n_value_mean"] - RawDataSet["p_at_n_value_std"],
                 RawDataSet["p_at_n_value_mean"] + RawDataSet["p_at_n_value_std"],
                 color='b', alpha=0.2)
ax63.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 6.3})


#############################################################################################################################
##lab code to plot:
# myResult2.to_csv("/Users/zlifr/Desktop/HHBOS/Result/votingResultMean2.csv", sep=',')

##myResult = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Result/votingResultMean2.csv", sep=",")

# import seaborn as sns
# sns.lineplot(data=myResult, x="neighbour_value", y="my_roc_score")
# sns.lineplot(data=myResult, x="neighbour_value", y="my_pr_auc")
# sns.lineplot(data=myResult, x="neighbour_value", y="P_at_n_value")
# sns.lineplot(data=myResult, x="neighbour_value", y="duration1")
# sns.lineplot(data=myResult, x="neighbour_value", y="duration3")

# df = myResult[["neighbour_value","my_pr_auc", "my_roc_score", "P_at_n_value"]].melt('neighbour_value', var_name='cols',  value_name='vals')
# g = sns.factorplot(x="neighbour_value", y="vals", hue='cols', data=df)
#############################################################################################################################




