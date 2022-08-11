#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:18:07 2021

@author: zlifr

##Reproduce ROCOD by Liang and Parthasarathy 2016

"""

##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub' 

# =============================================================================
# #Step I. implement ROCOD
# =============================================================================

import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import percentile
import numbers
import numpy as np


def ROCOD_Basic(RawDataSet, MyColList, MyContextList, MyBehaveList, distance_threshold, save_index, FilePath_DistanceMatrix):
    """
    
    When the ROCOD model is not learned, we use this function.
    
    Parameters
    ----------
    RawDataSet : dataframe
        dataframe containing raw dataset after preprocessing.
    MyColList : TYPE
        the list of all feature names.
    MyContextList : TYPE
        the list of contextual feature names.
    MyBehaveList : TYPE
       the list of behavioural feature names.
    distance_threshold : double
        threshold value used to find neighbours.
    save_index : 0 or 1
        indicate whether to save or load the distance matrix.
    FilePath_DistanceMatrix : string
        the file path to save or load the distance matrix.

    Returns
    -------
    my_pr_auc : double
        pr auc score.
    my_roc_score : double
        roc auc score.
    P_at_n_value : double
        p@n score.
    MyDataSet : dataframe
        dataframe containing raw dataset and anomaly scores.
        
    """
    
    RawDataSet = RawDataSet.dropna()  #remove missing values
    
    MyDataSet = RawDataSet[MyColList]
    MyContextDataSet = MyDataSet[MyContextList]
    MyBehaveDataSet = MyDataSet[MyBehaveList]
    sample_value = RawDataSet["ground_truth"][RawDataSet["ground_truth"] == 1].count()
    MyContextDataSet = MyContextDataSet.astype(str)
     
    #################################################################################
    ##Step 1: Doing global computing
    
    ##Step 1.1: Calculate Gower Distance
    if save_index == 0:
        import gower
        distance_matrix = gower.gower_matrix(MyContextDataSet) 
        # np.save("/Users/zlifr/Desktop/HHBOS/Data/tempfile.npy", distance_matrix) 
        np.save(FilePath_DistanceMatrix, distance_matrix)
        
    else:
        # distance_matrix = np.load("/Users/zlifr/Desktop/HHBOS/Data/tempfile.npy")
        distance_matrix = np.load(FilePath_DistanceMatrix)
        
        ##We should add a row and a column in the matrix
    
    ##Step 1.2: Construct Global regression
    from sklearn.linear_model import LinearRegression
    X_train = MyContextDataSet
    y_train_raw = MyBehaveDataSet
    
    Regression_Models = dict()
    # RSquare_Scores = dict()
    
    for col_name in  y_train_raw.columns:
         
        target_col = y_train_raw[col_name]
        
        reg_per_col = LinearRegression().fit(X_train, target_col)
        
        RSScore_per_col = reg_per_col.score(X_train, target_col) ##Return the coefficient of determination of the prediction.
        
        Regression_Models[col_name] = reg_per_col
        
        # RSquare_Scores[col_name] = RSScore_per_col
    
    ##Step 1.3: Find number of neighbors for each data point
    Number_Of_ContextMemebers = {}
    
    
    def NumOfNeighbours(query_point_index, distance_threshold):
        """
        Parameters
        ----------
        query_point_index : int
            the index of query instance.
        distance_threshold : double
            threshold value used to find neighbours.

        Returns
        -------
        query_filter_result : int
            the number of found neighbours.

        """
        # print("query_point_index:")
        # print(query_point_index)
        # print("query_point_index location:")
        # print(list(RawDataSet.index).index(query_point_index))
        
        query_point_index_location = (list(RawDataSet.index)).index(query_point_index)
        
        # query_result = distance_matrix[query_point_index]
        query_result = distance_matrix[query_point_index_location]
        
        query_filter_result = np.where(query_result <= distance_threshold)
        # query_filter_result = np.where(query_result <= 0.3)
        query_filter_result = list(query_filter_result[0])
        
        # query_filter_result.remove(query_point_index)
        query_filter_result.remove(query_point_index_location)
        
        return len(query_filter_result)
    
    # for query_index_num in range(RawDataSet.shape[0]):
    for query_index_num in RawDataSet.index:
        
        num_neighbors_result = NumOfNeighbours(query_index_num, distance_threshold)
        
        Number_Of_ContextMemebers[query_index_num] = num_neighbors_result
        
    Max_ContextualMembers = max(Number_Of_ContextMemebers.values())
    
    ##Step 1.4: Calculate mean for each column
    
    MyBehaveDataSet_Mean = list(MyBehaveDataSet.mean())
    MyBehaveDataSet_Mean = pd.DataFrame([MyBehaveDataSet_Mean], columns=MyBehaveDataSet.columns)
    
    
    #################################################################################
    ##Step 2: define a function to calculate raw anomaly score for each data point
            
    def AnomalyRawScorePerPoint(query_point_index, distance_threshold,
                                MyBehaveDataSet, MyContextDataSet,
                                Max_ContextualMembers, Regression_Models,
                                distance_matrix):
        """
        Parameters
        ----------
        query_point_index : int
            the index of query instance.
        distance_threshold : double
            threshold value used to find neighbours.
        MyBehaveDataSet : dataframe
            dataframe consisting of behavioural feature space of reference group and the query object.
        MyContextDataSet : dataframe
            dataframe consisting of contextual feature space of reference group and the query object..
        Max_ContextualMembers : int
            the number of found contextual neighbours.
        Regression_Models : dict
            a dictionary of learned regression models.
        distance_matrix : dataframe
            computed distance matrix.

        Returns
        -------
        Final_point_Bahave : dataframe
            anomaly scores in each feature (as target feature).

        """
    
        #################################################################################
        ##Step 2.1: For each point, find local pattern based on its neighbors
        
        query_point_index_location = (list(RawDataSet.index)).index(query_point_index)
        
        # query_result = distance_matrix[query_point_index]
        query_result = distance_matrix[query_point_index_location]
        
        query_filter_result = np.where(query_result <= distance_threshold)
        query_filter_result = list(query_filter_result[0])
        
        # query_filter_result.remove(query_point_index)
        query_filter_result.remove(query_point_index_location)
        
        Neighbours_Behave = MyBehaveDataSet[MyBehaveDataSet.index.isin(query_filter_result)]
        Local_point_Behave = list(Neighbours_Behave.mean())
        Local_point_Behave = pd.DataFrame([Local_point_Behave], columns=Neighbours_Behave.columns)
        
        # print(query_point_index)
        Query_point_Behave = MyBehaveDataSet[MyBehaveDataSet.index.isin([query_point_index])].values.tolist()
        # print(Query_point_Behave)
        Query_point_Behave = list(Query_point_Behave[0])
        Query_point_Behave = pd.DataFrame([Query_point_Behave], columns=Neighbours_Behave.columns)
        
        
        
        #################################################################################
        ##Step 2.2: For each point, find global pattern based on the whole dataset
               
        #For a query point, we predict its behaviour
        X_test = MyContextDataSet[MyContextDataSet.index.isin([query_point_index])]
        y_test_raw = MyBehaveDataSet[MyBehaveDataSet.index.isin([query_point_index])]
        
        Global_point_Behave = []
        
        for test_col_name in  y_test_raw.columns:
            
            Regression_Model_Per_col = Regression_Models[test_col_name]
            
            y_predicted_per_col = Regression_Model_Per_col.predict(X_test)[0]
            
            Global_point_Behave.append(y_predicted_per_col)
            
        Global_point_Behave = pd.DataFrame([Global_point_Behave], columns=y_test_raw.columns)
            
        #################################################################################
        ##Step 2.3: Combination of local pattern and global pattern
        
        ##Add a condition to test whether the number of neighbours is larger than the number of variables
        ##If not, we only use the gobal models
        import math
        
        if Max_ContextualMembers == 0:
            weight_local = math.sqrt(Number_Of_ContextMemebers[query_point_index]/0.01)
        else:          
            weight_local = math.sqrt(Number_Of_ContextMemebers[query_point_index]/Max_ContextualMembers)
        weight_global = 1 - weight_local
                        
                        
        Final_point_Bahave = weight_local*Local_point_Behave.iloc[0]  + weight_global*Global_point_Behave.iloc[0]
        
        Final_point_Bahave = pd.DataFrame([Final_point_Bahave], columns=Neighbours_Behave.columns)
    
        return Final_point_Bahave
    
    
    #################################################################################
    ##Step 5: Calculate weight for each column, this should be done after all calculations
    
    
    AllPoint_Bahave = pd.DataFrame(columns=MyBehaveDataSet.columns)
    
    
    # for data_point in range(RawDataSet.shape[0]): ##(this is for old version without train test split)
    for data_point in RawDataSet.index: 
    
        result_df = AnomalyRawScorePerPoint(query_point_index = data_point, distance_threshold = distance_threshold, 
                                            MyBehaveDataSet = MyBehaveDataSet, MyContextDataSet = MyContextDataSet,
                                            Max_ContextualMembers  = Max_ContextualMembers, Regression_Models = Regression_Models,
                                            distance_matrix = distance_matrix)
    
        AllPoint_Bahave = AllPoint_Bahave.append(result_df, ignore_index=True)
        
        # print(list(RawDataSet.index).index(data_point))
    # print(AllPoint_Bahave.shape[0])
    # print(MyBehaveDataSet.shape[0])
    AllPoint_Bahave.index = list(MyBehaveDataSet.index)       
    
    Weight_RSquare = []
    
    for col_name in MyBehaveDataSet.columns:
        
        RS_score_Denominator = 0
        RS_score_Numerator = 0
        RS_score = 0
        
        # for data_point in range(RawDataSet.shape[0]):
        for data_point in RawDataSet.index:
    
            RS_score_Numerator += (MyBehaveDataSet[col_name][data_point] - AllPoint_Bahave[col_name][data_point])**2
            RS_score_Denominator += (MyBehaveDataSet[col_name][data_point] - MyBehaveDataSet_Mean[col_name][0])**2
            
        # print(RS_score_Denominator)
        RS_score = max(0,1- RS_score_Numerator/RS_score_Denominator)
        
        # print(RS_score)
        
        Weight_RSquare.append(RS_score) 
       
    # Weight_RSquare = [1]*AllPoint_Bahave.shape[1]
    
    
    Weight_RSquare = pd.DataFrame([Weight_RSquare], columns=MyBehaveDataSet.columns) #Most weights are zero
    ##Problems occur here
    
    #################################################################################
    ##Step 6: Calculate difference between predicted and actual value for each cell
    AllPoint_Bahave["point_index"] = AllPoint_Bahave.index
    MyBehaveDataSet["point_index"] = MyBehaveDataSet.index
    
    
    Behave_Difference = AllPoint_Bahave.set_index('point_index').subtract(MyBehaveDataSet.set_index('point_index'), fill_value=0)
    def square(x):
        return x**2
    
    Behave_Difference = Behave_Difference.apply(square)
    
    MyBehaveDataSet.index = list(MyBehaveDataSet.index) 
    
    Final_Anomaly_Score = []
    
    # for data_point in range(RawDataSet.shape[0]): 
    for data_point in RawDataSet.index:
        
        # print(data_point)
        # print("difference")
        # print(Behave_Difference.loc[data_point])
        # print("weight")
        # print(Weight_RSquare.iloc[0])
        point_Anomaly_Score = np.sum(Behave_Difference.loc[data_point]*Weight_RSquare.iloc[0])
        # print(point_Anomaly_Score)
        
        Final_Anomaly_Score.append(point_Anomaly_Score)
        
        
    MyDataSet["Raw_Anomaly_Score"] =  Final_Anomaly_Score

                
        
    from scipy import stats
    
    percentiles = [stats.percentileofscore(Final_Anomaly_Score, i) for i in Final_Anomaly_Score]
    MyDataSet["anomaly_score"] = percentiles
    
    #################################################################################
    ##Step 7: Evaluate performance
    
    #evaluate anomaly score 1: roc auc score 
    from sklearn.metrics import roc_auc_score
    
    my_roc_score = roc_auc_score(MyDataSet["ground_truth"], MyDataSet["anomaly_score"])
    
    #evaluate anomaly score 2: Precision@n score, Recall@n score and F@n score
    TempDataSet = MyDataSet[["ground_truth","anomaly_score"]]
    
    P_TempDataSet = TempDataSet.sort_values(by=['anomaly_score'], ascending=[False]).head(sample_value)
    
    TP_value = (P_TempDataSet["ground_truth"]== 1).sum()
    
    P_at_n_value = TP_value/sample_value
        
    #evaluate anomaly score 3:prc auc score 
    from sklearn.metrics import precision_recall_curve, auc, roc_curve
    
    y = np.array(MyDataSet["ground_truth"])
    pred = np.array(MyDataSet["anomaly_score"])
    
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1) #to calculate roc auc
    precision, recall, thresholds = precision_recall_curve(y, pred, pos_label=1) #to calculate pr auc
    
    my_pr_auc = auc(recall, precision) #pr auc 
    
    return my_pr_auc, my_roc_score, P_at_n_value, MyDataSet
    


def ROCOD(RawDataSet, MyColList, MyContextList, MyBehaveList, distance_threshold, save_index, FilePath_DistanceMatrix, is_model_learned, TrainDataSet):
    """
    Two cases are considered:
        0: the model is not learned. Everything, especially the distance matrix, needs to be computed.
        1: the model has been learned. The distance matrix needs not to be recomputed when we perform multiple running tests.


    Parameters
    ----------
    RawDataSet : dataframe
        dataframe containing raw dataset after preprocessing.
    MyColList : TYPE
        the list of all feature names.
    MyContextList : TYPE
        the list of contextual feature names.
    MyBehaveList : TYPE
       the list of behavioural feature names.
    distance_threshold : double
        threshold value used to find neighbours.
    save_index : 0 or 1
        indicate whether to save or load the distance matrix.       
    FilePath_DistanceMatrix : string
        the file path to save or load the distance matrix.
    is_model_learned : o or 1
        indicate whether the model has been learned.
    TrainDataSet : dataframe
        dataframe containing raw dataset after preprocessing. A copy to avoid errors.

    Returns
    -------
    my_pr_auc : double
        pr auc score.
    my_roc_score : double
        roc auc score.
    P_at_n_value : double
        p@n score.
    duringtime1 : double
        training time. Not used.
    duringtime2 : double
        training + testing time. Not used

    """
    
    ##first test whther it is a train or test matrix
    if is_model_learned == 0:
       my_pr_auc, my_roc_score, P_at_n_value, MyDataSet =  ROCOD_Basic(RawDataSet, MyColList, MyContextList, MyBehaveList, distance_threshold, save_index, FilePath_DistanceMatrix)
       duringtime1 = 0
       duringtime2 = 0
       print(my_pr_auc, my_roc_score, P_at_n_value, duringtime1, duringtime1)       
       return my_pr_auc, my_roc_score, P_at_n_value, duringtime1, duringtime2
   
    ##Test whether it is a learned model
    elif is_model_learned == 1:
        
        """
        This is similar to the ROCOD_Basic() function
        
        """
        TestDataSet = RawDataSet.copy(deep=True)
        sample_value = TestDataSet["ground_truth"][TestDataSet["ground_truth"] == 1].count()
        
        import gower
        TrainTestGowerMatrix = gower.gower_matrix(TrainDataSet, TestDataSet) 
        Raw_Anomaly_Score_Vec = []
        
        for row_index  in RawDataSet.index:
            
            ##Step 1.1: Calculate Gower Distance
            ##Add a new row of gower distance
            row_index_location = (list(TestDataSet.index)).index(row_index)
            print(row_index_location)
            new_distance_row = list(TrainTestGowerMatrix[:,row_index_location])
           
            ##Load GowerDistanceMatrix and Add a new row and a new column
            distance_matrix = np.load(FilePath_DistanceMatrix)
            distance_matrix = np.vstack([distance_matrix, new_distance_row])
            new_distance_row.append(0)
            distance_matrix = np.vstack([distance_matrix.T, new_distance_row])
            
            
            RawDataSet = TrainDataSet.append(TestDataSet.iloc[[row_index_location]] ) 
            RawDataSet = RawDataSet.dropna()  #remove missing values
            MyDataSet = RawDataSet[MyColList]
            # print(MyDataSet.index)
            MyContextDataSet = MyDataSet[MyContextList]
            MyBehaveDataSet = MyDataSet[MyBehaveList]
            MyContextDataSet = MyContextDataSet.astype(str)
            
            ##Step 1.2: Construct Global regression
            from sklearn.linear_model import LinearRegression
            X_train = MyContextDataSet
            y_train_raw = MyBehaveDataSet
            
            Regression_Models = dict()


            for col_name in  y_train_raw.columns:
                 
                target_col = y_train_raw[col_name]
                
                reg_per_col = LinearRegression().fit(X_train, target_col)
                
                RSScore_per_col = reg_per_col.score(X_train, target_col) ##Return the coefficient of determination of the prediction.
                
                Regression_Models[col_name] = reg_per_col
                
                # RSquare_Scores[col_name] = RSScore_per_col
            
            ##Step 1.3: Find number of neighbors for each data point
            Number_Of_ContextMemebers = {}       
            
            def NumOfNeighbours(query_point_index, distance_threshold):
                 
                query_point_index_location = (list(RawDataSet.index)).index(query_point_index)
                
                # query_result = distance_matrix[query_point_index]
                query_result = distance_matrix[query_point_index_location]
                
                query_filter_result = np.where(query_result <= distance_threshold)
                # query_filter_result = np.where(query_result <= 0.3)
                query_filter_result = list(query_filter_result[0])
                
                # query_filter_result.remove(query_point_index)
                query_filter_result.remove(query_point_index_location)
                
                
                return len(query_filter_result)
        
            # for query_index_num in range(RawDataSet.shape[0]):
            for query_index_num in RawDataSet.index:
                
                num_neighbors_result = NumOfNeighbours(query_index_num, distance_threshold)
                
                Number_Of_ContextMemebers[query_index_num] = num_neighbors_result
                
            Max_ContextualMembers = max(Number_Of_ContextMemebers.values())
            
            ##Step 1.4: Calculate mean for each column
            
            MyBehaveDataSet_Mean = list(MyBehaveDataSet.mean())
            MyBehaveDataSet_Mean = pd.DataFrame([MyBehaveDataSet_Mean], columns=MyBehaveDataSet.columns)
                
            #################################################################################
            ##Step 2: define a function to calculate raw anomaly score for each data point
                    
            def AnomalyRawScorePerPoint(query_point_index, distance_threshold,
                                        MyBehaveDataSet, MyContextDataSet,
                                        Max_ContextualMembers, Regression_Models,
                                        distance_matrix):
                """
                As above
                """
            
                #################################################################################
                ##Step 2.1: For each point, find local pattern based on its neighbors
                
                query_point_index_location = (list(RawDataSet.index)).index(query_point_index)
                
                
                # query_result = distance_matrix[query_point_index]
                query_result = distance_matrix[query_point_index_location]
                
                query_filter_result = np.where(query_result <= distance_threshold)
                query_filter_result = list(query_filter_result[0])
                
                # query_filter_result.remove(query_point_index)
                query_filter_result.remove(query_point_index_location)
                
                Neighbours_Behave = MyBehaveDataSet[MyBehaveDataSet.index.isin(query_filter_result)]
                Local_point_Behave = list(Neighbours_Behave.mean())
                Local_point_Behave = pd.DataFrame([Local_point_Behave], columns=Neighbours_Behave.columns)
                
                # print(query_point_index)
                Query_point_Behave = MyBehaveDataSet[MyBehaveDataSet.index.isin([query_point_index])].values.tolist()
                # print(Query_point_Behave)
                Query_point_Behave = list(Query_point_Behave[0])
                Query_point_Behave = pd.DataFrame([Query_point_Behave], columns=Neighbours_Behave.columns)
                
                
                #################################################################################
                ##Step 2.2: For each point, find global pattern based on the whole dataset
                       
                #For a query point, we predict its behaviour
                X_test = MyContextDataSet[MyContextDataSet.index.isin([query_point_index])]
                y_test_raw = MyBehaveDataSet[MyBehaveDataSet.index.isin([query_point_index])]
                
                Global_point_Behave = []
                
                for test_col_name in  y_test_raw.columns:
                    
                    Regression_Model_Per_col = Regression_Models[test_col_name]
                    
                    y_predicted_per_col = Regression_Model_Per_col.predict(X_test)[0]
                    
                    Global_point_Behave.append(y_predicted_per_col)
                    
                Global_point_Behave = pd.DataFrame([Global_point_Behave], columns=y_test_raw.columns)
                    
                #################################################################################
                ##Step 2.3: Combination of local pattern and global pattern
                
                ##Add a condition to test whether the number of neighbours is large than the number variables
                ##If not, we only use the gobal models
                import math
                
                weight_local = math.sqrt(Number_Of_ContextMemebers[query_point_index]/Max_ContextualMembers)
                weight_global = 1 - weight_local
                                
                                
                Final_point_Bahave = weight_local*Local_point_Behave.iloc[0]  + weight_global*Global_point_Behave.iloc[0]
                
                Final_point_Bahave = pd.DataFrame([Final_point_Bahave], columns=Neighbours_Behave.columns)
            
                return Final_point_Bahave           
            
            #################################################################################
            ##Step 5: Calculate weight for each column, this should be done after all calculations
            
            
            AllPoint_Bahave = pd.DataFrame(columns=MyBehaveDataSet.columns)
            
            
            # for data_point in range(RawDataSet.shape[0]): ##(this is for old version without train test split)
            for data_point in RawDataSet.index: 
            
                result_df = AnomalyRawScorePerPoint(query_point_index = data_point, distance_threshold = distance_threshold, 
                                                    MyBehaveDataSet = MyBehaveDataSet, MyContextDataSet = MyContextDataSet,
                                                    Max_ContextualMembers  = Max_ContextualMembers, Regression_Models = Regression_Models,
                                                    distance_matrix = distance_matrix)
            
                AllPoint_Bahave = AllPoint_Bahave.append(result_df, ignore_index=True)
                
                # print(list(RawDataSet.index).index(data_point))
            # print(AllPoint_Bahave.shape[0])
            # print(MyBehaveDataSet.shape[0])
            AllPoint_Bahave.index = list(MyBehaveDataSet.index)       
            
            Weight_RSquare = []
            
            for col_name in MyBehaveDataSet.columns:
                
                RS_score_Denominator = 0
                RS_score_Numerator = 0
                RS_score = 0
                
                # for data_point in range(RawDataSet.shape[0]):
                for data_point in RawDataSet.index:
            
                    RS_score_Numerator += (MyBehaveDataSet[col_name][data_point] - AllPoint_Bahave[col_name][data_point])**2
                    RS_score_Denominator += (MyBehaveDataSet[col_name][data_point] - MyBehaveDataSet_Mean[col_name][0])**2
                    
                # print(RS_score_Denominator)
                RS_score = max(0,1- RS_score_Numerator/RS_score_Denominator)
                
                # print(RS_score)
                
                Weight_RSquare.append(RS_score) 
               
            # Weight_RSquare = [1]*AllPoint_Bahave.shape[1]
            
            
            Weight_RSquare = pd.DataFrame([Weight_RSquare], columns=MyBehaveDataSet.columns) #Most weights are zero
            ##Problems occur here
            
            #################################################################################
            ##Step 6: Calculate difference between predicted and actual value for each cell
            AllPoint_Bahave["point_index"] = AllPoint_Bahave.index
            MyBehaveDataSet["point_index"] = MyBehaveDataSet.index
            
            
            Behave_Difference = AllPoint_Bahave.set_index('point_index').subtract(MyBehaveDataSet.set_index('point_index'), fill_value=0)
            def square(x):
                return x**2
            
            Behave_Difference = Behave_Difference.apply(square)
            
            MyBehaveDataSet.index = list(MyBehaveDataSet.index) 
            
            Final_Anomaly_Score = []
            
            # for data_point in range(RawDataSet.shape[0]): 
            for data_point in RawDataSet.index:
                
                # print(data_point)
                # print("difference")
                # print(Behave_Difference.loc[data_point])
                # print("weight")
                # print(Weight_RSquare.iloc[0])
                point_Anomaly_Score = np.sum(Behave_Difference.loc[data_point]*Weight_RSquare.iloc[0])
                # print(point_Anomaly_Score)
                
                Final_Anomaly_Score.append(point_Anomaly_Score)
                
                
            MyDataSet["Raw_Anomaly_Score"] =  Final_Anomaly_Score

                
            from scipy import stats
            
            percentiles = [stats.percentileofscore(Final_Anomaly_Score, i) for i in Final_Anomaly_Score]
            # print(percentiles)
            MyDataSet["anomaly_score"] = percentiles
            
            index_location = (list(RawDataSet.index)).index(row_index)
            
            # print(row_index)
            # print(index_location)
            # print(RawDataSet[["ground_truth"]].iloc[[index_location]])
            test_anomaly_score = MyDataSet[["anomaly_score"]].iloc[[index_location]] #return a dataframe
            test_anomaly_score = test_anomaly_score.iloc[0]['anomaly_score']
            # print(MyDataSet[["anomaly_score"]].iloc[[index_location]])
            print(test_anomaly_score)
            Raw_Anomaly_Score_Vec.append(test_anomaly_score)
            
        TestDataSet["anomaly_score"] = Raw_Anomaly_Score_Vec
        #################################################################################
        ##Step 7: Evaluate performance
        
        #evaluate anomaly score 1: roc auc score 
        from sklearn.metrics import roc_auc_score
        
        my_roc_score = roc_auc_score(TestDataSet["ground_truth"], TestDataSet["anomaly_score"])
        
        #evaluate anomaly score 2: Precision@n score, Recall@n score and F@n score
        TempDataSet = TestDataSet[["ground_truth","anomaly_score"]]
        
        P_TempDataSet = TempDataSet.sort_values(by=['anomaly_score'], ascending=[False]).head(sample_value)
        
        TP_value = (P_TempDataSet["ground_truth"]== 1).sum()
        
        P_at_n_value = TP_value/sample_value
            
        #evaluate anomaly score 3:prc auc score 
        from sklearn.metrics import precision_recall_curve, auc, roc_curve
        
        y = np.array(TestDataSet["ground_truth"])
        pred = np.array(TestDataSet["anomaly_score"])
        
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1) #to calculate roc auc
        precision, recall, thresholds = precision_recall_curve(y, pred, pos_label=1) #to calculate pr auc
        
        my_pr_auc = auc(recall, precision) #pr auc 
        
        duringtime1 = 0
        duringtime2 = 0
        print(my_pr_auc, my_roc_score, P_at_n_value, duringtime1, duringtime1)
        
        return my_pr_auc, my_roc_score, P_at_n_value, duringtime1, duringtime2
        
        




# =============================================================================
# Step II. Testing function with real-world dataset
# =============================================================================
    
# ##########################################
# ## Step1: load dataset and set parameters
# ##########################################

# RawDataSetPath = AbsRootDir+r'/QCAD/Data/GenData/bodyfatGene.csv'
# RawDataSet = pd.read_csv(RawDataSetPath, sep=",")
# RawDataSet = RawDataSet.dropna()  #remove missing values

# MyColList = ['Density', 'BodyFat', 
#               'Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 
#               'Knee', 'Ankle', 'Biceps', 'Forearm',
#               'ground_truth']

# MyContextList = ['Age', 'Weight', 'Height', 'Neck', 'Chest','Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm']

# MyBehaveList = ['Density', 'BodyFat']

# MyDataSet = RawDataSet[MyColList]

# ##########################################
# ## Step2: call ROCOD function to get results
# ##########################################

# ROCOD(MyDataSet, MyColList, MyContextList, MyBehaveList, 0.9, 0, r'',0, MyDataSet) 









