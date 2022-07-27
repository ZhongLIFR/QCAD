#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 22:33:52 2021

@author: zlifr
"""



"""
Here is the process to generate injected contextual outliers from \cite{song2007conditional}:

-Description1: \cite{Kuo2016 CIKM}
We do not follow this scheme for several reasons. First, swapping the attribute values may not always
obtain desired outliers. It is likely that most of the swaps could result in normal data. 
Second, as we observe many extreme outliers in the real-world datasets, swapping values between 
samples in a clean data is less likely to produce this extreme difference between yi and yj. 
Here we present another way to generate outliers and we explore different types of outliers 
where we give controls to where and how many outliers are injected or its degree of outlierness.


To inject q×N outliers into a dataset with N data samples, we randomly select q×N records z®i = ( ®xi
,yi)to be perturbed. Let yi be the target attribute for perturbation.Let x®i be the rest of attributes. 
For all selected records, a random number from (0, α) is added up to yi as y′i. Then we add new
sample z®′ = ( ®xi,y′i) into the original dataset and flag it as outlier. Note that original N data 
samples are flagged as non-outlier. In the experiments, we standardized the target attribute to range (18,
30) which are the min and max value of the behavioral attribute in Elnino dataset. Set α as 50 by default.

"""

##############################################################################
##############################################################################
##Step1. define a function to inject outliers and output result as a csv file
##############################################################################
##############################################################################
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

##############################################################################
##############################################################################
##Step2. apply above defined function to generate injected contextual outliers
##############################################################################
##############################################################################

import pandas as pd
import numpy as np
##File0######################################################################

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/abaloneOriginal.csv", sep=",") #it contains 4177 rows
# RawDataSet = RawDataSet.dropna()

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# RawDataSet["Sex"] = label_encoder.fit_transform(RawDataSet["Sex"])
# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/abalone.csv", sep=',')

# FilePath0 = r"/Users/zlifr/Desktop/HHBOS/Data/abalone.csv"
# AllCols0 = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
#             'Viscera weight', 'Shell weight', 'Rings']


# ContextCols0 = ['Sex', 'Length', 'Diameter', 'Height']

# BehaveCols0 = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

# NumCols0 = ['Whole weight', 'Shucked weight','Viscera weight', 'Shell weight', 'Rings']

# FinalDataSet0 = GenerateData(FilePath0, AllCols0, ContextCols0, BehaveCols0, NumCols0,  418)

# FinalDataSet0.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/abaloneGene.csv", sep=',')


##File1######################################################################

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/depressedOriginal.csv", sep=",") #it contains  rows
# RawDataSet = RawDataSet.dropna()
# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/depressed.csv", sep=',')

# FilePath1 = r"/Users/zlifr/Desktop/HHBOS/Data/depressed.csv"
# #columns:
# #Contextual: "Ville_id","sex","Age","Married","Number_children","education_level","total_members"
# #Behavioural: "gained_asset","durable_asset","save_asset","living_expenses","other_expenses",
# #             "incoming_agricultural","farm_expenses","lasting_investment","no_lasting_investmen"
# #Label: "depressed"

# #For simplicity, we first omit "Ville_id", "incoming_salary", "incoming_own_farm","incoming_business","incoming_no_business","labor_primary"
# AllCols1 = ["sex","Age","Married","Number_children","education_level","total_members",
#             "gained_asset","durable_asset","save_asset","living_expenses","other_expenses",
#             "incoming_agricultural","farm_expenses","lasting_investment","no_lasting_investmen",
#             "depressed"]


# ContextCols1 = ["sex","Age","Married","Number_children","education_level","total_members"]

# BehaveCols1 = ["gained_asset","durable_asset","save_asset","living_expenses","other_expenses",
#               "incoming_agricultural","farm_expenses","lasting_investment","no_lasting_investmen"]

# NumCols1 = ["Age","gained_asset","durable_asset","save_asset","living_expenses","other_expenses",
#             "incoming_agricultural","farm_expenses","lasting_investment","no_lasting_investmen"]

# FinalDataSet1 = GenerateData(FilePath1, AllCols1, ContextCols1, BehaveCols1, NumCols1,  72)

# FinalDataSet1.to_csv("/Users/zlifr/Desktop/HHBOS/Data2/depressedGene.csv", sep=',')

##File2######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/heartFailureOriginal.csv", sep=",") #it contains 299 rows
# RawDataSet = RawDataSet.dropna()
# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/heartFailure.csv", sep=',')

# FilePath2 = r"/Users/zlifr/Desktop/HHBOS/Data/heartFailure.csv"
# ##columns:
# ##Contextual: "age","sex","smoking","diabetes","high_blood_pressure","anaemia"
# ##Behavioural: "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"
# ##Label: "DEATH_EVENT"

# AllCols2 =["age","sex","smoking","diabetes","high_blood_pressure","anaemia",
#             "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time",
#             "DEATH_EVENT"]

# ContextCols2 = ["age","sex","smoking","diabetes","high_blood_pressure","anaemia"]

# BehaveCols2 = ["creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

# NumCols2 = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

# FinalDataSet2 = GenerateData(FilePath2, AllCols2, ContextCols2, BehaveCols2,NumCols2, 30)
# FinalDataSet2.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/heartFailureGene.csv", sep=',')

##File3######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/elninoOriginal.csv", sep=",") #it contains 178080 rows

# RawDataSet=RawDataSet.replace('.',np.nan).dropna(axis = 0, how = 'any') # it contains 93935 after dropping missing values

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/elnino.csv", sep=',')

# FilePath3 = r"/Users/zlifr/Desktop/HHBOS/Data/elnino.csv"

# AllCols3 = ['Year', 'Month', 'Day', 'Date', 'Latitude','Longitude',
#             'Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp']

# ContextCols3 = ['Year', 'Month', 'Day', 'Date', 'Latitude','Longitude']

# BehaveCols3 = ['Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp']

# NumCols3 = ['Zonal_Winds', 'Meridional_Winds', 'Humidity', 'Air_Temp', 'Sea_Surface_Temp']

# # FinalDataSet3 = GenerateDataCategory(FilePath3, AllCols3, ContextCols3, BehaveCols3, NumCols3, 940)
# ##We downsample the data to 20000 and thus inject 1000 outliers
# FinalDataSet3 = GenerateData(FilePath3, AllCols3, ContextCols3, BehaveCols3, NumCols3, 1000)
# # FinalDataSet3 = GenerateData(FilePath3, AllCols3, ContextCols3, BehaveCols3, NumCols3, 20000)
# FinalDataSet3.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/elninoGene.csv", sep=',')

#add a downsample version
# FinalDataSet3 = GenerateDataCategory(FilePath3, AllCols3, ContextCols3, BehaveCols3, 940)
# FinalDataSet3.to_csv("/Users/zlifr/Desktop/HHBOS/Data2/elninoGeneDownSample.csv", sep=',')

##File4######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/creditCard.csv", sep=",") #it contains 30000 rows

# RawDataSet = RawDataSet.dropna() # it contains 30000 after dropping missing values


# FilePath4 = r"/Users/zlifr/Desktop/HHBOS/Data/creditCard.csv"

# AllCols4 = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
#             'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 
#             'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# ContextCols4 = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# BehaveCols4 = ['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 
#                 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
# NumCols4 = ['LIMIT_BAL','AGE',
#             'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 
#             'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
# FinalDataSet4 = GenerateData(FilePath4, AllCols4, ContextCols4, BehaveCols4, NumCols4, 300)
# FinalDataSet4.to_csv("/Users/zlifr/Desktop/HHBOS/Data2/creditCardGene.csv", sep=',')

##File5######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/bodyfat.csv", sep=",") #it contains 252 rows

# RawDataSet = RawDataSet.dropna()

# FilePath5 = r"/Users/zlifr/Desktop/HHBOS/Data/bodyfat.csv"

# AllCols5 = ['Density', 'BodyFat', 'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# ContextCols5 = ['Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# BehaveCols5 = ['Density', 'BodyFat']

# NumCols5 = ['Density', 'BodyFat', 'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# FinalDataSet5 = GenerateData(FilePath5, AllCols5, ContextCols5, BehaveCols5, NumCols5, 30)

# FinalDataSet5.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/bodyfatGene.csv", sep=',')

##File6######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/boston.csv", sep=",") #it contains 506 rows

# RawDataSet = RawDataSet.dropna()

# FilePath6 = r"/Users/zlifr/Desktop/HHBOS/Data/boston.csv"

# AllCols6 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

# ContextCols6 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']

# BehaveCols6 = ['MEDV']

# NumCols6 = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

# FinalDataSet6 = GenerateData(FilePath6, AllCols6, ContextCols6, BehaveCols6, NumCols6, 50)

# FinalDataSet6.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/bostonGene.csv", sep=',')

##File7######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/indianLiverPatientOriginal.csv", sep=",") #it contains 583 rows

# RawDataSet = RawDataSet.dropna()

# cleanup_nums = {"Gender":     {"Female": 0, "Male": 1}}

# RawDataSet = RawDataSet.replace(cleanup_nums)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/indianLiverPatient.csv", sep=',')

# FilePath7 = r"/Users/zlifr/Desktop/HHBOS/Data/indianLiverPatient.csv"

# AllCols7 = ['Age', 'Gender', 'Selector',
#             'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# ContextCols7 = ['Age', 'Gender', 'Selector']

# BehaveCols7 = ['Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#                 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# NumCols7 = ['Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']

# FinalDataSet7 = GenerateData(FilePath7, AllCols7, ContextCols7, BehaveCols7, NumCols7, 58)

# FinalDataSet7.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/indianLiverPatientGene.csv", sep=',')

##File8######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/hepatitisOriginal.csv", sep=",") #it contains 615 rows

# RawDataSet = RawDataSet.dropna()

# cleanup_nums = {"Category":     {"0=Blood Donor": 0, "0s=suspect Blood Donor": 1, 
#                                  "1=Hepatitis": 2, "2=Fibrosis": 3,
#                                  "3=Cirrhosis": 4},
#                 "Sex":     {"f": 0, "m": 1}
#                 }
# RawDataSet = RawDataSet.replace(cleanup_nums)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/hepatitis.csv", sep=',')

# FilePath8 = r"/Users/zlifr/Desktop/HHBOS/Data/hepatitis.csv"

# AllCols8 = ['Category', 'Age', 'Sex', 
#             'ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# ContextCols8 = ['Category', 'Age', 'Sex']

# BehaveCols8 = ['ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# NumCols8 = ['Age','ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# FinalDataSet8 = GenerateData(FilePath8, AllCols8, ContextCols8, BehaveCols8, NumCols8, 59)

# FinalDataSet8.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/hepatitisGene.csv", sep=',')

##File9######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/forestFiresOriginal.csv", sep=",") #it contains 517 rows

# RawDataSet = RawDataSet.dropna()

# cleanup_nums = {"month":     {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, "jul": 6,
#                               "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11 },
#                 "day":     {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
#                 }

# RawDataSet = RawDataSet.replace(cleanup_nums)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/forestFires.csv", sep=',')

# FilePath9 = r"/Users/zlifr/Desktop/HHBOS/Data/forestFires.csv"

# AllCols9 = ['X', 'Y', 'month', 'day', 
#             'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# ContextCols9 = ['X', 'Y', 'month', 'day']

# BehaveCols9 = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# NumCols9 = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']

# FinalDataSet9 = GenerateData(FilePath9, AllCols9, ContextCols9, BehaveCols9, NumCols9, 52)

# FinalDataSet9.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/forestFiresGene.csv", sep=',')

##File10######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/yachtHydrodynamicsOriginal.csv") #it contains 308 rows

# RawDataSet = RawDataSet.dropna()

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/yachtHydrodynamics.csv", sep=',')

# FilePath10 = r"/Users/zlifr/Desktop/HHBOS/Data/yachtHydrodynamics.csv"

# AllCols10 = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#               'Beam_draught_ratio', 'Length_beam_ratio','Froude_number', 
#               'resistance']

# ContextCols10 = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#                   'Beam_draught_ratio', 'Length_beam_ratio','Froude_number']

# BehaveCols10 = ['resistance']

# NumCols10 = ['Longitudinal_position', 'Prismatic_coefficient','Length_displacement_ratio', 
#               'Beam_draught_ratio', 'Length_beam_ratio','Froude_number', 
#               'resistance']

# FinalDataSet10 = GenerateData(FilePath10, AllCols10, ContextCols10, BehaveCols10, NumCols10, 31)

# FinalDataSet10.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/yachtHydrodynamicsGene.csv", sep=',')

##File11######################################################################


# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/studentPerformanceOriginal.csv") #it contains xxx rows

# RawDataSet = RawDataSet.dropna()

# cleanup_nums = {"school":     {"GP": 0, "MS": 1},
#                 "sex":     {"F": 0, "M": 1},
#                 "address":     {"U": 0, "R": 1},
#                 "famsize":     {"LE3": 0, "GT3": 1},
#                 "Pstatus":     {"A": 0, "T": 1},
#                 "Mjob":     {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4} ,
#                 "Fjob":     {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4} ,
#                 "reason":     {"home": 0, "reputation": 1, "course": 2, "other": 3 },
#                 "guardian":     {"mother": 0, "father": 1, "other": 2},
#                 "schoolsup":     {"no": 0, "yes": 1},
#                 "famsup":     {"no": 0, "yes": 1},
#                 "paid":     {"no": 0, "yes": 1},
#                 "activities":     {"no": 0, "yes": 1},
#                 "nursery":     {"no": 0, "yes": 1},
#                 "higher":     {"no": 0, "yes": 1},
#                 "internet":     {"no": 0, "yes": 1},
#                 "romantic":     {"no": 0, "yes": 1}           
#                 }

# RawDataSet = RawDataSet.replace(cleanup_nums)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/studentPerformance.csv", sep=',')

# FilePath11 = r"/Users/zlifr/Desktop/HHBOS/Data/studentPerformance.csv"

# AllCols11 = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
#               'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
#               'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
#               'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
#               'Walc', 'health', 'absences', 
#               'G1', 'G2', 'G3']

# ContextCols11 = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
#                   'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
#                   'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
#                   'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
#                   'Walc', 'health', 'absences']

# BehaveCols11 = ['G1', 'G2', 'G3']

# ##whether we should min_max all variables?
# NumCols11 = ['G1', 'G2', 'G3']

# FinalDataSet11 = GenerateData(FilePath11, AllCols11, ContextCols11, BehaveCols11, NumCols11, 65)

# FinalDataSet11.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/studentPerformanceGene.csv", sep=',')

##File12######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/adultOriginal.csv") #it contains 48842 rows

# RawDataSet=RawDataSet.replace('?',np.nan).dropna(axis = 0, how = 'any') # it contains 45222 after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()

# RawDataSet["workclass"] = label_encoder.fit_transform(RawDataSet["workclass"])
# RawDataSet["education"] = label_encoder.fit_transform(RawDataSet["education"])
# RawDataSet["marital.status"] = label_encoder.fit_transform(RawDataSet["marital.status"])
# RawDataSet["occupation"] = label_encoder.fit_transform(RawDataSet["occupation"])
# RawDataSet["relationship"] = label_encoder.fit_transform(RawDataSet["relationship"])
# RawDataSet["race"] = label_encoder.fit_transform(RawDataSet["race"])
# RawDataSet["sex"] = label_encoder.fit_transform(RawDataSet["sex"])
# RawDataSet["native.country"] = label_encoder.fit_transform(RawDataSet["native.country"])
# RawDataSet["income"] = label_encoder.fit_transform(RawDataSet["income"])
# del RawDataSet["education"]

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/adult.csv", sep=',')

# FilePath12 = r"/Users/zlifr/Desktop/HHBOS/Data/adult.csv"

# AllCols12 = ['age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
#               'occupation', 'relationship', 'race', 'sex', 'capital.gain',
#               'capital.loss', 'hours.per.week', 'native.country', 
#               'income']

# ContextCols12 = ['age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
#                   'occupation', 'relationship', 'race', 'sex', 'capital.gain',
#                   'capital.loss', 'hours.per.week', 'native.country']

# BehaveCols12 = ['income']

# ##whether we should min_max all variables?
# NumCols12 = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# FinalDataSet12 = GenerateData(FilePath12, AllCols12, ContextCols12, BehaveCols12, NumCols12, 452)

# FinalDataSet12.to_csv("/Users/zlifr/Desktop/HHBOS/Data/adultGene.csv", sep=',')

##File13######################################################################
##we need to find some large dataset where the number of contextual features is small such as 1-3
##this dataset is a good example for misdiagnosis in healthcare

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/breastCancerOriginal.csv") #it contains 569 rows

# RawDataSet = RawDataSet.iloc[:, 1 :-1] ##remove first and last column

# RawDataSet = RawDataSet.dropna() # it contains xxxx after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()

# RawDataSet["diagnosis"] = label_encoder.fit_transform(RawDataSet["diagnosis"])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/breastCancer.csv", sep=',')

# FilePath13 = r"/Users/zlifr/Desktop/HHBOS/Data/breastCancer.csv"

# AllCols13 = ['radius_mean', 'texture_mean', 'perimeter_mean',
#               'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#               'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#               'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#               'fractal_dimension_se', 'radius_worst', 'texture_worst',
#               'perimeter_worst', 'area_worst', 'smoothness_worst',
#               'compactness_worst', 'concavity_worst', 'concave points_worst',
#               'symmetry_worst', 'fractal_dimension_worst',
#               'diagnosis']

# ContextCols13 = ['radius_mean', 'texture_mean', 'perimeter_mean',
#                   'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#                   'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#                   'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#                   'fractal_dimension_se', 'radius_worst', 'texture_worst',
#                   'perimeter_worst', 'area_worst', 'smoothness_worst',
#                   'compactness_worst', 'concavity_worst', 'concave points_worst',
#                   'symmetry_worst', 'fractal_dimension_worst' ]

# BehaveCols13 = ['diagnosis']

# ##whether we should min_max all variables?
# NumCols13 = ['radius_mean', 'texture_mean', 'perimeter_mean',
#               'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#               'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#               'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#               'fractal_dimension_se', 'radius_worst', 'texture_worst',
#               'perimeter_worst', 'area_worst', 'smoothness_worst',
#               'compactness_worst', 'concavity_worst', 'concave points_worst',
#               'symmetry_worst', 'fractal_dimension_worst' ]

# FinalDataSet13 = GenerateData(FilePath13, AllCols13, ContextCols13, BehaveCols13, NumCols13, 57)

# FinalDataSet13.to_csv("/Users/zlifr/Desktop/HHBOS/Data/breastCancerGene.csv", sep=',')

##File14######################################################################

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/drugConsumptionQuantified.csv") #it contains 1885 rows

# demo_data = RawDataSet.copy()

# age = ['18-24' if a <= -0.9 else 
#        '25-34' if a >= -0.5 and a < 0 else 
#        '35-44' if a > 0 and a < 1 else 
#        '45-54' if a > 1 and a < 1.5 else 
#        '55-64' if a > 1.5 and a < 2 else 
#        '65+' 
#        for a in demo_data['Age']]

# gender = ['Female' if g > 0 else "Male" for g in demo_data['Gender']]

# education = ['Left school before 16 years' if e <-2 else 
#              'Left school at 16 years' if e > -2 and e < -1.5 else 
#              'Left school at 17 years' if e > -1.5 and e < -1.4 else 
#              'Left school at 18 years' if e > -1.4 and e < -1 else 
#              'Some college or university, no certificate or degree' if e > -1 and e < -0.5 else 
#              'Professional certificate/ diploma' if e > -0.5 and e < 0 else 
#              'University degree' if e > 0 and e < 0.5 else 
#              'Masters degree' if e > 0.5 and e < 1.5 else 
#              'Doctorate degree' 
#              for e in demo_data['Education']]

# country = ['USA' if c < -0.5 else 
#            'New Zealand' if c > -0.5 and c < -0.4 else 
#            'Other' if c > -0.4 and c < -0.2 else 
#            'Australia' if c > -0.2 and c < 0 else 
#            'Ireland' if c > 0 and c < 0.23 else 
#            'Canada' if c > 0.23 and c < 0.9 else 
#            'UK' 
#            for c in demo_data['Country']]

# ethnicity = ['Black' if e < -1 else 
#              'Asian' if e > -1 and e < -0.4 else 
#              'White' if e > -0.4 and e < -0.25 else 
#              'Mixed-White/Black' if e >= -0.25 and e < 0.11 else 
#              'Mixed-White/Asian' if e > 0.12 and e < 1 else 
#              'Mixed-Black/Asian' if e > 1.9 else 
#              'Other' 
#              for e in demo_data['Ethnicity']]


# demo_data['Age'] = age
# demo_data['Gender'] = gender
# demo_data['Education'] = education
# demo_data['Country'] = country
# demo_data['Ethnicity'] = ethnicity

# demo_data.to_csv("/Users/zlifr/Desktop/HHBOS/Data/drugConsumptionOriginal.csv", sep=',')

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/drugConsumptionOriginal.csv") #it contains 1885 rows

# RawDataSet = RawDataSet.iloc[:, 2 :] ##remove first and second column

# RawDataSet = RawDataSet.dropna() # it contains 1885 after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()


# for col_index in range(0,5):
#     col_name = RawDataSet.columns.tolist()[col_index]
#     RawDataSet[col_name] = label_encoder.fit_transform(RawDataSet[col_name])

# for col_index in range(12,31):
#     col_name = RawDataSet.columns.tolist()[col_index]
#     RawDataSet[col_name] = label_encoder.fit_transform(RawDataSet[col_name])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/drugConsumption.csv", sep=',')

# FilePath14 = r"/Users/zlifr/Desktop/HHBOS/Data/drugConsumption.csv"

# AllCols14 = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 
#               'Nscore','Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 
#               'Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
#               'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
#               'Nicotine', 'Semer', 'VSA']

# ContextCols14 = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 
#                   'Nscore','Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS' ]

# BehaveCols14 = ['Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
#                 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
#                 'Nicotine', 'Semer', 'VSA']

# ##whether we should min_max all variables?
# NumCols14 = ['Nscore','Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS' ]

# FinalDataSet14 = GenerateDataCategory(FilePath14, AllCols14, ContextCols14, BehaveCols14, NumCols14, 190)

# FinalDataSet14.to_csv("/Users/zlifr/Desktop/HHBOS/Data2/drugConsumptionGene.csv", sep=',')

##File15######################################################################
# import pandas as pd
# import numpy as np

# from unidecode import unidecode

# RawDataSet1 = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/QS2018Clean.csv", sep=";") #it contains xxxx rows

# RawDataSet1 = RawDataSet1.dropna() # it contains xxxx after dropping missing values

# RawDataSet1["World2018"] = RawDataSet1.index+1

# RawDataSet1["National2018"] = RawDataSet1.groupby("Country")["World2018"].rank("dense", ascending=True)

# RawDataSet1["Institution Name"] = RawDataSet1["Institution Name"].str.strip() #remove strip at both ends
# RawDataSet1["Institution Name"] = RawDataSet1["Institution Name"].apply(unidecode) #switch french letters to english letters


# RawDataSet2 = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/QS2019Clean.csv", sep=";") #it contains xxxx rows

# RawDataSet2 = RawDataSet2.dropna() # it contains xxxx after dropping missing values

# RawDataSet2["World2019"] = RawDataSet2.index+1

# RawDataSet2["National2019"] = RawDataSet2.groupby("Country")["World2019"].rank("dense", ascending=True)

# RawDataSet2["Institution Name"] = RawDataSet2["Institution Name"].str.strip()#remove strip at both ends
# RawDataSet2["Institution Name"] = RawDataSet2["Institution Name"].apply(unidecode)#switch french letters to english letters

# QS1 = pd.merge(RawDataSet1[["Institution Name", "World2018", "National2018"]], 
#                RawDataSet2[["Institution Name", "World2019", "National2019"]], 
#                how='left', on=["Institution Name"])

# RawDataSet3 = QS1.copy()

# RawDataSet4 = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/QS2020Clean.csv", sep=";") #it contains xxxx rows

# RawDataSet4 = RawDataSet4.dropna() # it contains xxxx after dropping missing values

# RawDataSet4["World2020"] = RawDataSet4.index+1

# RawDataSet4["National2020"] = RawDataSet4.groupby("Country")["World2020"].rank("dense", ascending=True)

# RawDataSet4["Institution Name"] = RawDataSet4["Institution Name"].str.strip()#remove strip at both ends
# RawDataSet4["Institution Name"] = RawDataSet4["Institution Name"].apply(unidecode)#switch french letters to english letters

# RawDataSet4["Institution Name"] = RawDataSet4["Institution Name"].str.upper() #turn into uppercase


# QS2 = pd.merge(RawDataSet3, 
#                RawDataSet4, 
#                how='right', on=["Institution Name"])

# RawDataSet5 = QS2.copy()

# RawDataSet5=RawDataSet5.replace('-',np.nan).dropna(axis = 0, how = 'any')

# RawDataSet5 = RawDataSet5.dropna() 


# RawDataSet5[['Academic Reputation', 'Employer Reputation',
#              'Faculty Student', 'Citations per Faculty', 'International Faculty',
#              'International Students', 'Overall Score']]=\
#     RawDataSet5[['Academic Reputation', 'Employer Reputation',
#              'Faculty Student', 'Citations per Faculty', 'International Faculty',
#              'International Students', 'Overall Score']].astype(float)
    
# RawDataSet5.to_csv("/Users/zlifr/Desktop/HHBOS/Data/QSRankingOriginal.csv", sep=',')   

# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/QSRankingOriginal.csv") #it contains 475 rows

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()

# RawDataSet["Country"] = label_encoder.fit_transform(RawDataSet["Country"])
# RawDataSet["SIZE"] = label_encoder.fit_transform(RawDataSet["SIZE"])
# RawDataSet["FOCUS"] = label_encoder.fit_transform(RawDataSet["FOCUS"])
# RawDataSet["RESEARCH INTENSITY"] = label_encoder.fit_transform(RawDataSet["RESEARCH INTENSITY"])
# RawDataSet["STATUS"] = label_encoder.fit_transform(RawDataSet["STATUS"])

# del RawDataSet["Unnamed: 0"]
# del RawDataSet["Institution Name"]
# del RawDataSet["Overall Score"]

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/QSRanking.csv", sep=',')


# FilePath15 = r"/Users/zlifr/Desktop/HHBOS/Data/QSRanking.csv"

# AllCols15 = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
#               'SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
#               'Academic Reputation', 'Employer Reputation', 'Faculty Student',
#               'Citations per Faculty', 'International Faculty',
#               'International Students']

# ContextCols15 = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country']

# ## BehaveCols15 = ['SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
# ##                 'Academic Reputation', 'Employer Reputation', 'Faculty Student',
# ##                 'Citations per Faculty', 'International Faculty',
# ##                'International Students']

# BehaveCols15 = ['Academic Reputation', 'Employer Reputation', 'Faculty Student',
#                 'Citations per Faculty', 'International Faculty',
#                 'International Students']
# ##whether we should min_max all variables?
# NumCols15 = ['Academic Reputation', 'Employer Reputation', 'Faculty Student',
#               'Citations per Faculty', 'International Faculty','International Students' ]
# # NumCols15 = []

# # NumCols15 = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
# #               'SIZE', 'FOCUS', 'RESEARCH INTENSITY', 'AGE', 'STATUS',
# #               'Academic Reputation', 'Employer Reputation', 'Faculty Student',
# #               'Citations per Faculty', 'International Faculty',
# #               'International Students']
# # NumCols15 = ['World2018', 'National2018', 'World2019', 'National2019','World2020', 'National2020', 'Country',
# #               'Academic Reputation', 'Employer Reputation', 'Faculty Student',
# #               'Citations per Faculty', 'International Faculty', 'International Students']

# # FinalDataSet15 = GenerateDataCategory(FilePath15, AllCols15, ContextCols15, BehaveCols15, NumCols15, 48)
# FinalDataSet15 = GenerateData(FilePath15, AllCols15, ContextCols15, BehaveCols15, NumCols15, 48)

# FinalDataSet15.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/QSRankingGene.csv", sep=',')

##File16######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/mushroomsOriginal.csv") #it contains 8124 rows

# RawDataSet = RawDataSet.dropna() # it contains xxxx after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()


# for col_index in range(len(RawDataSet.columns.tolist())):
#     col_name = RawDataSet.columns.tolist()[col_index]
#     RawDataSet[col_name] = label_encoder.fit_transform(RawDataSet[col_name])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/mushrooms.csv", sep=',')

# FilePath16 = r"/Users/zlifr/Desktop/HHBOS/Data/mushrooms.csv"

# AllCols16 = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#               'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#               'stalk-surface-below-ring', 'stalk-color-above-ring',
#               'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#               'ring-type', 'spore-print-color', 'population', 'habitat',
#               'class']

# ContextCols16 = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#                   'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#                   'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#                   'stalk-surface-below-ring', 'stalk-color-above-ring',
#                   'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#                   'ring-type', 'spore-print-color', 'population', 'habitat']

# BehaveCols16 = ['class']

# ##whether we should min_max all variables?
# NumCols16 = []

# FinalDataSet16 = GenerateData(FilePath16, AllCols16, ContextCols16, BehaveCols16, NumCols16, 812)

# FinalDataSet16.to_csv("/Users/zlifr/Desktop/HHBOS/Data/mushroomsGene.csv", sep=',')

##File17######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/votingOriginal.csv") #it contains 435 rows

# del RawDataSet['export-administration-act-south-africa'] #remove rthis dataset because it contains too many missing values

# RawDataSet=RawDataSet.replace('?',np.nan).dropna(axis = 0, how = 'any') # it contains 281 after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()


# for col_index in range(len(RawDataSet.columns.tolist())):
#     col_name = RawDataSet.columns.tolist()[col_index]
#     RawDataSet[col_name] = label_encoder.fit_transform(RawDataSet[col_name])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/voting.csv", sep=',')

# FilePath17 = r"/Users/zlifr/Desktop/HHBOS/Data/voting.csv"

# AllCols17 = ['handicapped-infants', 'water-project-cost-sharing',
#               'adoption-of-the-budget-resolution', 'physician-fee-freeze',
#               'el-salvador-aid', 'religious-groups-in-schools',
#               'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile',
#               'immigration', 'synfuels-corporation-cutback', 'education-spending',
#               'superfund-right-to-sue', 'crime', 'duty-free-exports',
#               'Class Name']

# ContextCols17 = ['handicapped-infants', 'water-project-cost-sharing',
#                   'adoption-of-the-budget-resolution', 'physician-fee-freeze',
#                   'el-salvador-aid', 'religious-groups-in-schools',
#                   'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile',
#                   'immigration', 'synfuels-corporation-cutback', 'education-spending',
#                   'superfund-right-to-sue', 'crime', 'duty-free-exports']

# BehaveCols17 = ['Class Name']

# ##whether we should min_max all variables?
# NumCols17 = []

# FinalDataSet17 = GenerateData(FilePath17, AllCols17, ContextCols17, BehaveCols17, NumCols17, 28)

# FinalDataSet17.to_csv("/Users/zlifr/Desktop/HHBOS/Data/votingGene.csv", sep=',')

##File18######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/nurseryOriginal.csv") #it contains 12960 rows

# RawDataSet = RawDataSet.dropna() # it contains xxxx after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()


# for col_index in range(len(RawDataSet.columns.tolist())):
#     col_name = RawDataSet.columns.tolist()[col_index]
#     RawDataSet[col_name] = label_encoder.fit_transform(RawDataSet[col_name])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/nursery.csv", sep=',')

# FilePath18 = r"/Users/zlifr/Desktop/HHBOS/Data/nursery.csv"

# AllCols18 = ['Parents', 'Has_nurs', 'Form', 'Children', 'Housing', 'Finance','Social', 'Health', 
#               'Nursery']

# ContextCols18 = ['Parents', 'Has_nurs', 'Form', 'Children', 'Housing', 'Finance','Social', 'Health']

# BehaveCols18 = ['Nursery']

# #whether we should min_max all variables?
# NumCols18 = []

# FinalDataSet18 = GenerateData(FilePath18, AllCols18, ContextCols18, BehaveCols18, NumCols18, 130) ##we test with 650

# FinalDataSet18.to_csv("/Users/zlifr/Desktop/HHBOS/Data/nurseryGene.csv", sep=',')

##File19######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/fishOriginal.csv") #it contains 157 rows

# RawDataSet = RawDataSet.dropna() # it contains xxxx after dropping missing values

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()

# RawDataSet["Species"] = label_encoder.fit_transform(RawDataSet["Species"])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/fish.csv", sep=',')


# FilePath19 = r"/Users/zlifr/Desktop/HHBOS/Data/fish.csv"

# AllCols19 = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']

# ContextCols19 = ['Species', 'Length1', 'Length2', 'Length3', 'Height','Width']

# BehaveCols19 = ['Weight']

# NumCols19 = ['Weight']

# FinalDataSet19 = GenerateData(FilePath19, AllCols19, ContextCols19, BehaveCols19, NumCols19, 20)

# FinalDataSet19.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/fishGene.csv", sep=',')

##File20######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/airfoilSelfNoiseOriginal.csv") #it contains 1503 rows

# RawDataSet = RawDataSet.dropna() # it contains xxxx after dropping missing values

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/airfoil.csv", sep=',')

# FilePath20 = r"/Users/zlifr/Desktop/HHBOS/Data/airfoil.csv"

# AllCols20 = ['f', 'alpha', 'c', 'U_infinity', 'delta', 'SSPL']

# ContextCols20 = ['f', 'alpha', 'c', 'U_infinity', 'delta']

# BehaveCols20 = ['SSPL']

# NumCols20 = ['SSPL']

# FinalDataSet20 = GenerateData(FilePath20, AllCols20, ContextCols20, BehaveCols20, NumCols20, 150) 

# FinalDataSet20.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/airfoilGene.csv", sep=',')

##File21######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/gasEmission2015Original.csv") #it contains 7384 rows

# RawDataSet = RawDataSet.dropna() # it contains 7384 after dropping missing values

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/gasEmission.csv", sep=',')

# FilePath21 = r"/Users/zlifr/Desktop/HHBOS/Data/gasEmission.csv"

# AllCols21 = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO','NOX']

# ContextCols21 = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP']

# BehaveCols21 = ['TEY', 'CO','NOX']

# NumCols21 = ['TEY', 'CO','NOX']

# FinalDataSet21 = GenerateData(FilePath21, AllCols21, ContextCols21, BehaveCols21, NumCols21, 360) 

# FinalDataSet21.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/gasEmissionGene.csv", sep=',')

##File22######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/SeoulBikeOriginal.csv", encoding= 'unicode_escape') #it contains 8760 rows
# RawDataSet = RawDataSet.dropna() # it contains 8760 after dropping missing values

# del RawDataSet["Date"]
# RawDataSet.rename(columns={'Rented Bike Count': 'CountPerHour',
#                            'Hour': 'Hour',
#                            'Temperature(°C)': 'Temperature',
#                            'Humidity(%)': 'Humidity',
#                            'Wind speed (m/s)': 'WindSpeed',
#                            'Visibility (10m)': 'Visibility',
#                            'Dew point temperature(°C)': 'DewPointTemp',
#                            'Solar Radiation (MJ/m2)': 'Radiation',
#                            'Rainfall(mm)': 'Rainfall',
#                            'Snowfall (cm)': 'Snowfall',
#                            'Seasons': 'Seasons',
#                            'Holiday': 'Holiday',
#                            'Functioning Day': 'FunctioningDay',
#                            }, inplace=True)

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# RawDataSet["Seasons"] = label_encoder.fit_transform(RawDataSet["Seasons"])
# RawDataSet["Holiday"] = label_encoder.fit_transform(RawDataSet["Holiday"])
# RawDataSet["FunctioningDay"] = label_encoder.fit_transform(RawDataSet["FunctioningDay"])

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/SeoulBike.csv", sep=',')


# FilePath22 = r"/Users/zlifr/Desktop/HHBOS/Data/SeoulBike.csv"

# AllCols22 = ['CountPerHour', 'Hour', 'Temperature', 'Humidity', 'WindSpeed',
#               'Visibility', 'DewPointTemp', 'Radiation', 'Rainfall', 'Snowfall',
#               'Seasons', 'Holiday', 'FunctioningDay']

# ContextCols22 = ['Hour', 'Temperature', 'Humidity', 'WindSpeed',
#               'Visibility', 'DewPointTemp', 'Radiation', 'Rainfall', 'Snowfall',
#               'Seasons', 'Holiday', 'FunctioningDay']

# BehaveCols22 = ['CountPerHour']

# NumCols22 = ['CountPerHour']

# FinalDataSet22 = GenerateData(FilePath22, AllCols22, ContextCols22, BehaveCols22, NumCols22, 440) 

# FinalDataSet22.to_csv("/Users/zlifr/Desktop/HHBOS/Data2/SeoulBikeGene.csv", sep=',')

##File23######################################################################
# import pandas as pd
# import numpy as np
# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/MaintenanceOriginal.txt", sep='  ', header=None)
# RawDataSet.columns = ["LeverPosition", "ShipSpeed", "GTT", "GTn",
#                       "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48",
#                       "P1", "P2", "Pexh", "TIC", "mf",
#                       "CompressorDecay", "TurbineDecay"]
# RawDataSet = RawDataSet.dropna() # it contains 11934 after dropping missing values
# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/Maintenance.csv", sep=',')

# FilePath23 = r"/Users/zlifr/Desktop/HHBOS/Data/Maintenance.csv"

# AllCols23 = ["LeverPosition", "ShipSpeed", "GTT", "GTn",
#               "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48",
#               "P1", "P2", "Pexh", "TIC", "mf",
#               "CompressorDecay", "TurbineDecay"]

# ContextCols23 = ["LeverPosition", "GTT", "GTn",
#               "GGn", "Ts", "Tp", "T48", "T1", "T2", "P48",
#               "P1", "P2", "Pexh", "TIC", "mf"]

# BehaveCols23 = ['ShipSpeed',"CompressorDecay", "TurbineDecay"]

# NumCols23 = ['ShipSpeed',"CompressorDecay", "TurbineDecay"]

# FinalDataSet23 = GenerateData(FilePath23, AllCols23, ContextCols23, BehaveCols23, NumCols23, 600) 

# FinalDataSet23.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/MaintenanceGene.csv", sep=',')

##File24######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/synchronousMachineOriginal.csv", sep=';') #it contains 557 rows
# RawDataSet = RawDataSet.dropna() # it contains 557 rows after dropping missing values

# def format_change(x):
#     return x.replace(',', '.')
# RawDataSet = RawDataSet.applymap(format_change)
# RawDataSet = RawDataSet.astype(float)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/synchronousMachine.csv", sep=',')

# FilePath24 = r"/Users/zlifr/Desktop/HHBOS/Data/synchronousMachine.csv"

# AllCols24 = ['Iy', 'PF', 'e', 'dIf', 'If']

# ContextCols24 = ['Iy', 'PF', 'e', 'dIf']

# BehaveCols24 = ['If']

# NumCols24 = ['If']

# FinalDataSet24 = GenerateData(FilePath24, AllCols24, ContextCols24, BehaveCols24, NumCols24, 55) 

# FinalDataSet24.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/synchronousMachineGene.csv", sep=',')

##File25######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/ConcreteOriginal.csv") #it contains 557 rows
# RawDataSet.columns = ["C1","C2","C3","C4","C5","C6","C7","Age","Strength"]
# RawDataSet = RawDataSet.dropna() # it contains 557 rows after dropping missing values
# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/Concrete.csv", sep=',')


# FilePath25 = r"/Users/zlifr/Desktop/HHBOS/Data/Concrete.csv"

# AllCols25 = ["C1","C2","C3","C4","C5","C6","C7","Age","Strength"]

# ContextCols25 = ["C1","C2","C3","C4","C5","C6","C7","Age"]

# BehaveCols25 = ['Strength']

# NumCols25 = ['Strength']

# FinalDataSet25 = GenerateData(FilePath25, AllCols25, ContextCols25, BehaveCols25, NumCols25, 100) 

# FinalDataSet25.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/ConcreteGene.csv", sep=',')

##File26######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/energyOriginal.csv") #it contains 768 rows
# RawDataSet = RawDataSet.dropna() # it contains  rows after dropping missing values

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/Energy.csv", sep=',')

# FilePath26 = r"/Users/zlifr/Desktop/HHBOS/Data/Energy.csv"

# AllCols26 = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']

# ContextCols26 = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# BehaveCols26 = ['Y1', 'Y2']

# NumCols26 = ['Y1', 'Y2']

# FinalDataSet26 = GenerateData(FilePath26, AllCols26, ContextCols26, BehaveCols26, NumCols26, 70) 

# FinalDataSet26.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/EnergyGene.csv", sep=',')

##File27######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/toxicityOriginal.csv", sep=';', header = None) #it contains 908 rows
# RawDataSet = RawDataSet.dropna() # it contains  rows after dropping missing values
# RawDataSet.columns = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50"]

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/toxicity.csv", sep=',')

# FilePath27 = r"/Users/zlifr/Desktop/HHBOS/Data/toxicity.csv"

# AllCols27 = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","LC50"]

# ContextCols27 = ["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP"]

# BehaveCols27 = ['LC50']

# NumCols27 = ['LC50']

# FinalDataSet27 = GenerateData(FilePath27, AllCols27, ContextCols27, BehaveCols27, NumCols27, 90) 

# FinalDataSet27.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/toxicityGene.csv", sep=',')

##File28######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/powerOriginal.csv", sep=',') #it contains 9568 rows
# RawDataSet = RawDataSet.dropna() # it contains 9568 rows after dropping missing values

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/power.csv", sep=',')

# FilePath28 = r"/Users/zlifr/Desktop/HHBOS/Data/power.csv"

# AllCols28 = ['AT', 'V', 'AP', 'RH', 'PE']

# ContextCols28 = ['AT', 'V', 'AP', 'RH']

# BehaveCols28 = ['PE']

# NumCols28 = ['PE']

# FinalDataSet28 = GenerateData(FilePath28, AllCols28, ContextCols28, BehaveCols28, NumCols28, 500) 

# FinalDataSet28.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/powerGene.csv", sep=',')

##File29######################################################################
# import pandas as pd
# import numpy as np

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data/ParkinsonOriginal.csv", sep=',') #it contains 5875 rows
# RawDataSet = RawDataSet.dropna() # it contains 5875 rows after dropping missing values

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/Data/parkinson.csv", sep=',')

# FilePath29 = r"/Users/zlifr/Desktop/HHBOS/Data/parkinson.csv"

# AllCols29 = ['subject', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
#               'Jitter', 'Jitter_Abs', 'Jitter_RAP', 'Jitter_PPQ5', 'Jitter_DDP',
#               'Shimmer', 'Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5',
#               'Shimmer_APQ11', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

# ContextCols29 = ['subject', 'age', 'sex', 'test_time', 
#               'Jitter', 'Jitter_Abs', 'Jitter_RAP', 'Jitter_PPQ5', 'Jitter_DDP',
#               'Shimmer', 'Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5',
#               'Shimmer_APQ11', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

# BehaveCols29 = ['motor_UPDRS', 'total_UPDRS']

# NumCols29 = ['motor_UPDRS', 'total_UPDRS']

# FinalDataSet29 = GenerateData(FilePath29, AllCols29, ContextCols29, BehaveCols29, NumCols29, 600) 

# FinalDataSet29.to_csv("/Users/zlifr/Desktop/HHBOS/Data3/parkinsonGene.csv", sep=',')






##############################################################################
##Synthetic Data 1-5  ########################################################
##############################################################################
# import pandas as pd
# import numpy as np
# from SyntheticData import GenSynDataset


# RawDataSet = GenSynDataset(num_con =20 , num_con_cat = 5, num_behave = 5, sample_size = 1000, num_gaussian = 5)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset1.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset1.csv"

# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']


# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3', 'con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 10) ## Case 1
# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset1Gene.csv", sep=',')

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 20) ## Case 2
# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset2Gene.csv", sep=',')

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 50) ## Case 3
# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset3Gene.csv", sep=',')

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 100) ## Case 4
# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset4Gene.csv", sep=',')

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 200) ## Case 5
# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset5Gene.csv", sep=',')



##############################################################################
##Synthetic Data 6  ##########################################################
##############################################################################
# import pandas as pd
# import numpy as np
# from SyntheticData import GenSynDataset


# RawDataSet = GenSynDataset(num_con =20 , num_con_cat = 2, num_behave = 5, sample_size = 1000, num_gaussian = 5)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset3.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset3.csv"

# AllCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4',
#            'con_num5', 'con_num6', 'con_num7', 'con_num8', 'con_num9',
#            'con_num10', 'con_num11', 'con_num12', 'con_num13', 'con_num14',
#            'con_num15', 'con_num16', 'con_num17', 'con_num_cat0', 'con_num_cat1',
#            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# ContextCols = ['con_num0', 'con_num1', 'con_num2', 'con_num3', 'con_num4',
#                 'con_num5', 'con_num6', 'con_num7', 'con_num8', 'con_num9',
#                 'con_num10', 'con_num11', 'con_num12', 'con_num13', 'con_num14',
#                 'con_num15', 'con_num16', 'con_num17', 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']


# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 50) 


# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset3Gene.csv", sep=',')

##############################################################################
##Synthetic Data 7  ##########################################################
##############################################################################
# import pandas as pd
# import numpy as np
# from SyntheticData import GenSynDataset


# RawDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 10, sample_size = 1000, num_gaussian = 5)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset4.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset4.csv"

# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#             'behave_num5', 'behave_num6', 'behave_num7', 'behave_num8','behave_num9']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#               'behave_num5', 'behave_num6', 'behave_num7', 'behave_num8','behave_num9']


# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#             'behave_num5', 'behave_num6', 'behave_num7', 'behave_num8','behave_num9']

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 50) 


# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset4Gene.csv", sep=',')



##############################################################################
##Synthetic Data 8  ##########################################################
##############################################################################
# import pandas as pd
# import numpy as np
# from SyntheticData import GenSynDataset


# RawDataSet = GenSynDataset(num_con =5 , num_con_cat = 2, num_behave = 20, sample_size = 1000, num_gaussian = 5)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset4.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset4.csv"

# AllCols = ['con_num0', 'con_num1', 'con_num2',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#             'behave_num5', 'behave_num6', 'behave_num7', 'behave_num8','behave_num9',
#             'behave_num10', 'behave_num11', 'behave_num12', 'behave_num13','behave_num14',
#             'behave_num15', 'behave_num16', 'behave_num17', 'behave_num18','behave_num19']

# ContextCols = ['con_num0', 'con_num1', 'con_num2',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#               'behave_num5', 'behave_num6', 'behave_num7', 'behave_num8','behave_num9',
#               'behave_num10', 'behave_num11', 'behave_num12', 'behave_num13','behave_num14',
#               'behave_num15', 'behave_num16', 'behave_num17', 'behave_num18','behave_num19']


# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4',
#             'behave_num5', 'behave_num6', 'behave_num7', 'behave_num8','behave_num9',
#             'behave_num10', 'behave_num11', 'behave_num12', 'behave_num13','behave_num14',
#             'behave_num15', 'behave_num16', 'behave_num17', 'behave_num18','behave_num19']

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 50) 


# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset4Gene.csv", sep=',')

##############################################################################
##Synthetic Data 9  #########################################################
##############################################################################
# import pandas as pd
# import numpy as np
# from SyntheticData import GenSynDataset

# RawDataSet = GenSynDataset(num_con =20 , num_con_cat = 20, num_behave = 5, sample_size = 1000, num_gaussian = 5)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset5.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset5.csv"

# AllCols = ['con_num_cat0', 'con_num_cat1','con_num_cat2', 'con_num_cat3','con_num_cat4', 
#            'con_num_cat5', 'con_num_cat6','con_num_cat7', 'con_num_cat8','con_num_cat9', 
#            'con_num_cat10', 'con_num_cat11','con_num_cat12', 'con_num_cat13','con_num_cat14', 
#            'con_num_cat15', 'con_num_cat16','con_num_cat17', 'con_num_cat18','con_num_cat19',
#            'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# ContextCols = ['con_num_cat0', 'con_num_cat1','con_num_cat2', 'con_num_cat3','con_num_cat4', 
#                'con_num_cat5', 'con_num_cat6','con_num_cat7', 'con_num_cat8','con_num_cat9', 
#                'con_num_cat10', 'con_num_cat11','con_num_cat12', 'con_num_cat13','con_num_cat14', 
#                'con_num_cat15', 'con_num_cat16','con_num_cat17', 'con_num_cat18','con_num_cat19']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']


# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 50) 


# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset5Gene.csv", sep=',')


##############################################################################
##Synthetic Data 10  ##########################################################
##############################################################################
# import pandas as pd
# import numpy as np
# from SyntheticData import GenSynDataset


# RawDataSet = GenSynDataset(num_con =10 , num_con_cat = 2, num_behave = 5, sample_size = 20000, num_gaussian = 5)

# RawDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset2.csv", sep=',')

# FilePath = r"/Users/zlifr/Desktop/HHBOS/SynData/Dataset2.csv"

# AllCols = ['con_num0', 'con_num1', 'con_num2','con_num3','con_num4', 'con_num5','con_num6', 'con_num7',
#             'con_num_cat0', 'con_num_cat1',
#             'behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']


# ContextCols = ['con_num0', 'con_num1', 'con_num2','con_num3', 'con_num4', 'con_num5','con_num6', 'con_num7',
#                 'con_num_cat0', 'con_num_cat1']

# BehaveCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# NumCols = ['behave_num0', 'behave_num1', 'behave_num2', 'behave_num3','behave_num4']

# FinalDataSet = GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, 500) 


# FinalDataSet.to_csv("/Users/zlifr/Desktop/HHBOS/SynData/Dataset2Gene.csv", sep=',')
