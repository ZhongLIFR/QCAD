#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:26:57 2021

Reproduce LoPAD[Sha, et al.]

@author: zlifr, z.li@liacs.leidenuniv.nl

"""
##You must specify AbsRootDir by yourself !!!!!!!!
AbsRootDir = '/Users/zlifr/Documents/GitHub' 

# =============================================================================
# Step I. implement LoPAD
# =============================================================================

def LoPAD(RawDataSet, MB_dataset_path, sample_value):
    """

    Parameters
    ----------
    RawDataSet : dataframe
        dataframe containing raw dataset after preprocessing.
    MB_dataset_path : string
        file path containing the Markov Blankets of correspoding dataset.
    sample_value : int
        the number of anomalies.

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
    
    ##############################################################################
    ##Step 1.1. Load Dataset
    ##############################################################################
    import pandas as pd
    
    # X_train =  RawDataSet.iloc[:, 1:-2] ##not for average test
    
    X_train =  RawDataSet.iloc[:, 0:-1]
    # print(X_train.columns)
    
    
    ##############################################################################
    ##Step 1.2 Use fast-IAMB to learn Markov Blankets for a dataset
    ##############################################################################
    import pandas as pd
    ##This is obtained by runing R script since Python does not support fast-IMAB to get MB 
    ##So we just need to load it
    MB_dataset = pd.read_csv(MB_dataset_path, sep=";",
                             header= None,
                             names= range(len(X_train.columns)),
                             skip_blank_lines=False)
    
    ## If reporting error "ValueError: Length mismatch", please uncomment the following block
    ## block begin
    ##--------------------------------
    
    # print("MB_dataset.columns")
    # print(MB_dataset.columns)
    # print("X_train.columns")
    # print(X_train.columns)
    # print("MB_dataset.shape[0]")
    # print(MB_dataset.shape[0])
    # MB_dataset = MB_dataset.T
    
    ##--------------------------------
    ## block end
    
    MB_dataset = MB_dataset.set_index(X_train.columns)
    # print("good")
    MB_dataset = MB_dataset.T
    
    ##R will replace " " by "." when it reads csv file with a column name
    ##so we have to change it back to " "
    for col_name in MB_dataset.columns:
        MB_dataset[col_name] = MB_dataset[col_name].str.replace('.',' ')
    
    
    ##############################################################################
    ##Step 2. Use CART regression tree to predict expected values and compare it with actual values
    ##############################################################################
    
    ##This is for trainning dataset
    from sklearn import tree
    import numpy as np
    import pandas as pd
    model = tree.DecisionTreeRegressor()
    
    final_result = pd.DataFrame()
    ##To mitigate influences of outliers, we use bootstrap arregating 
    for rand_item in range(20):
        sample_df = RawDataSet.sample(frac=0.8, random_state=rand_item)
        X_train =  sample_df.iloc[:, 0:-1]
        #X_train =  sample_df.iloc[:, :-2]
        
        result_df_train = pd.DataFrame()
    
        ##For each target vairable and its MBs, we use CART to predict its value and calculate the corresponding anomaly score
        for item in X_train.columns:
            
            target_variable = item
            predictor_variables = list(MB_dataset[target_variable].values)
            predictor_variables = [x for x in predictor_variables if str(x) != 'nan']
            
            ##If its Markov Blanket is not  empty, we construct a predictor
            if len(predictor_variables) > 0:
                regressor = model.fit(X_train[predictor_variables], X_train[target_variable])
                
                predicted_value = list(regressor.predict(X_train[predictor_variables]))
                actual_value = list(X_train[target_variable].values)
                
                ##compute the abs difference between prediction and actual value
                abs_diff = [ abs(a_i - b_i) for a_i, b_i in zip(predicted_value, actual_value)]
                
                ##normalise the difference to Z-score
                mean_value = np.mean(abs_diff)
                std_value =  max(np.std(abs_diff),0.0001) #to avoid using zero as denominator
                norm_abs_diff = [ (i-mean_value)/ std_value for i in abs_diff]
                
                ##prune the normalised anomaly score
                max_norm_abs_diff = [ max(i, 0) for i in norm_abs_diff]
            
                result_df_train[target_variable] = max_norm_abs_diff 
            ##If its Markov Blanket is empty, we skip it by setting its anomaly score to 'nan'
            ##When we compute the average, we will ignore the 'nan' cell
            else:
                result_df_train[target_variable] = np.nan 
        
        ##Add up the anomaly scores of all feature as the total anomaly score 
        result_df_train["score_sum"] = result_df_train.iloc[:,].sum(axis=1)
    
        result_df_train = result_df_train.set_index(X_train.index)
    
        ##Merge the anomaly scores of sampled points in each iteration
        df1 = result_df_train[["score_sum"]]
        final_result = pd.merge(final_result, df1, left_index=True, right_index=True, how='outer')
    
    ##Use the mean anomaly scores of all bootstrapping iterations as the final anomaly score
    final_result["mean_score"] = final_result.mean(axis=1) #automatically ignore nan
    
    
    ##############################################################################
    ##Step 3. Calculate the performance metrics: PR AUC, ROC AUC and P@n
    ##############################################################################
    
    
    MyDataSet =  RawDataSet.copy()
    
    y_train_scores = final_result["mean_score"].values
    
    raw_anomaly_score = list(y_train_scores)
    
    from scipy import stats
    
    percentiles = [stats.percentileofscore(raw_anomaly_score, i) for i in raw_anomaly_score]
    
    MyDataSet["raw_anomaly_score"] = raw_anomaly_score  
    MyDataSet["anomaly_score"] = percentiles     
    
          
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
    
    duringtime1 = 0
    duringtime2 = 0
    ###############################################################################
    ##output
    print(my_pr_auc,my_roc_score,P_at_n_value)
    return my_pr_auc,my_roc_score,P_at_n_value,duringtime1,duringtime2


# =============================================================================
# Step II. testing LoPAD with real-world dataset
# =============================================================================

# ##########################################
# ## Step1: load dataset and set parameters
# ##########################################

# import pandas as pd
# RawDataSetPath = AbsRootDir+r'/QCAD/Data/GenData/bostonGene.csv'
# RawDataSet = pd.read_csv(RawDataSetPath, sep=",")
# MB_dataset_path = AbsRootDir+r'/QCAD/Data/GenData/MB/bostonGene.csv'
# sample_value = 50

# ##########################################
# ## Step2: call LoPAD function to get results
# ##########################################

# LoPAD(RawDataSet, MB_dataset_path, sample_value)

## If reporting error "ValueError: Length mismatch", please uncomment the block from lines 70 to 78