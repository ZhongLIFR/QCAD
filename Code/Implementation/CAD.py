#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:32:29 2021

@author: zlifr

Reproduce Conditional Anomaly Detection

"""
# =============================================================================
# Step I. implement COD
# =============================================================================

import pandas as pd
import math

import warnings
warnings.filterwarnings("ignore")


def COD(RawDataSet, MyContextList, MyBehaveList, num_gau_comp, alpha_log, is_model_learned, FilePath_MappingMatrix):
    from time import time
    t0 = time() # to record time 
    
    RawDataSet = RawDataSet.dropna()  #remove missing values
    
    print(RawDataSet.shape[1])
    print(RawDataSet.columns)
    print(MyContextList)
    print((len(MyContextList) + len(MyBehaveList)+2))
    
    if RawDataSet.shape[1] == (len(MyContextList) + len(MyBehaveList)+2):
        
        MyDataSet = RawDataSet.iloc[:,1:-1]##This is not for function
    else:
        MyDataSet = RawDataSet.iloc[:,0:-1] ##this is for function
    
    OriginalCols = MyDataSet.columns
    
    context_index = [i for i, e in enumerate(OriginalCols) if e in MyContextList] # context_index = [0,1,2,3,4,5,6,7]

    behave_index = [i for i, e in enumerate(OriginalCols) if e in MyBehaveList] # behave_index = [8,9]
    
    sample_value = RawDataSet["ground_truth"][RawDataSet["ground_truth"] == 1].count()


    ##############################################################################
    ##Step 2: Use EM to learn the CAD-full model, this step takes a lot of time to train
    ##We have six items to calculate in total to obtain the loglikelihood
    import numpy as np
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=num_gau_comp, random_state=0,  covariance_type='full').fit(MyDataSet)
    GMM_weight = gm.weights_
    GMM_mean = gm.means_
    GMM_var = gm.covariances_
    
    
    ##calculate the likelihood of a data point belonging to a multivariate normal distribution?
    from scipy.stats import multivariate_normal
    import numpy as np
    
    if is_model_learned != 1:
        
        ##Using a number to denote whether it is initialization
        is_frist_iteration = 1
    
        for num_ieteration in range(100):
            """ For a single iteration """
            ##Set default mapping matrix value   
            if is_frist_iteration ==1 :
                mapping_matrix = np.empty((num_gau_comp, num_gau_comp))
                mapping_matrix.fill(1/num_gau_comp)  
                is_frist_iteration = 0
                
                # print(mapping_matrix)
                
                ##Set default b value matrix value for first iteration, later it should be changed
                b_value_matrix = np.empty((MyDataSet.shape[0], num_gau_comp, num_gau_comp))
                b_value_matrix.fill(0)   
                expected_loglikelihood = 0
                expected_loglikelihood_old = expected_loglikelihood
                
            
            
            ##Calculate expectation of log_likelihood
            
            for point_index in range(MyDataSet.shape[0]):
                
                test_point = MyDataSet.iloc[point_index]
                test_point_context = test_point[MyContextList]
                test_point_behave = test_point[MyBehaveList]
                
                ##Calculate the likelihood of a given data point produced by all gaussian distributions for contextual features
                all_likelihood_gau = []
                for gau_comp in range(num_gau_comp):
                    mean_gau = GMM_mean[gau_comp]
                    mean_gau = mean_gau[context_index]
                    var_gau = GMM_var[gau_comp]
                    var_gau = var_gau[:,context_index]
                    var_gau = var_gau[context_index,:]
                    ##A singular covariance matrix may be generated during the fitting of an GMM to a dataset
                    ##If the covariance matrix is singular, then the multivariate_normal.pdf will report errors
                    ##Thus, we add "allow_singular=True" in this function although it will degrades the performance
                    likelihood_gau = multivariate_normal.pdf(test_point_context, mean=mean_gau, cov=var_gau, allow_singular=True)
                    all_likelihood_gau.append(likelihood_gau)
                    
                all_likehood_context =  np.array(all_likelihood_gau)*GMM_weight
                all_likehood_context = np.sum(all_likehood_context) ## 5th item
            
            
                likelihood_behave_all = []
                for gau_behave_index in range(num_gau_comp):
                    
                    ###Find the mean and covariance of a gaussian distribution for behavioral attributes
                    mean_behave = GMM_mean[gau_behave_index]
                    mean_behave = mean_behave[behave_index]
                    var_behave = GMM_var[gau_behave_index]
                    var_behave = var_behave[:,behave_index]
                    var_behave = var_behave[behave_index,:]
                    
                    ##Calculate the likelihood of a given data point produced by a certain gaussian distribution for behavioral features
                    likelihood_behave_all.append(multivariate_normal.pdf(test_point_behave, mean=mean_behave, cov=var_behave))
                    # print(likelihood_behave_all) 
                    
                likelihood_context_all = []
                for gau_context_index in range(num_gau_comp):
                    
                    ##Find the mean and covariance of a gaussian distribution for contextual attributes
                    mean_context = GMM_mean[gau_context_index]
                    mean_context = mean_context[context_index]
                    var_context = GMM_var[gau_context_index]
                    var_context = var_context[:,context_index]
                    var_context = var_context[context_index,:]
                    
                    ##Calculate the likelihood of a given data point produced by a certain gaussian distribution for contextual features
                    ##A singular covariance matrix may be generated during the fitting of an GMM to a dataset
                    ##If the covariance matrix is singular, then the multivariate_normal.pdf will report errors
                    ##Thus, we add "allow_singular=True" in this function although it will degrades the performance
                    likelihood_context_all.append(multivariate_normal.pdf(test_point_context, mean=mean_context, cov=var_context,allow_singular=True))
                    # print(likelihood_context_all) 
                    
                
                ##calculate the Denominator of b_value 
                b_value_deno = 0
                for gau_context_index in range(num_gau_comp):
                    
                    ##Get likelihood_behave from pre-computed list
                    likelihood_context = likelihood_context_all[gau_behave_index] ## 1st item
                            
                    for gau_behave_index in range(num_gau_comp):
                        
                        ##Get the probability of mapping from context to behaviour
                        proba_mapping = mapping_matrix[gau_context_index][gau_behave_index] ## 2nd item
             
                        ##Get likelihood_behave from pre-computed list
                        likelihood_behave = likelihood_behave_all[gau_behave_index] ## 3rd item
                                        
                        ##Calculate the weight of a certain gaussian distribution for contextual features
                        proba_gau = GMM_weight[gau_context_index] ## 4th item
                        
                        b_value_deno = b_value_deno + likelihood_context*proba_gau*likelihood_behave*proba_mapping
                        
                ##calculate b_value, update b_value_matrix, and calculate Expectation of Loglikelihood      
                for gau_context_index in range(num_gau_comp):
                    
                    ##Get likelihood_behave from pre-computed list
                    likelihood_context = likelihood_context_all[gau_behave_index] ## 3rd item
                     
                    for gau_behave_index in range(num_gau_comp):
                        
                        ##Get the probability of mapping from context to behaviour
                        proba_mapping = mapping_matrix[gau_context_index][gau_behave_index] ## 2nd item
             
                        ##Get likelihood_behave from pre-computed list
                        likelihood_behave = likelihood_behave_all[gau_behave_index] ## 1st item
                        
                        ##Calculate the weight of a certain gaussian distribution for contextual features
                        proba_gau = GMM_weight[gau_context_index] ## 4th item
                        
                        
                        b_value_nmr = likelihood_context*proba_gau*likelihood_behave*proba_mapping
                        
                        b_value_deno = max(alpha_log,0) #to avoid zero in the denominator
                        
                        b_value = b_value_nmr/b_value_deno ##6th item
                        
                        b_value_matrix[point_index][gau_context_index][gau_behave_index] = b_value
                        
                        log_likelihood_item = (math.log(likelihood_behave+alpha_log)+math.log(proba_mapping+alpha_log)+\
                                               math.log(likelihood_context+alpha_log)+ math.log(proba_gau) -\
                                               math.log(all_likehood_context+alpha_log))*b_value
                        
                        #print(log_likelihood_item)
                        expected_loglikelihood = expected_loglikelihood + log_likelihood_item
                                                            
            
                ##update mapping_matrix  
                for gau_context_index in range(num_gau_comp):
                    
                    b_update_demo = 0
                    for gau_behave_index in range(num_gau_comp):
                        for point_index in range(MyDataSet.shape[0]):
                            b_update_demo = b_update_demo + b_value_matrix[point_index][gau_context_index][gau_behave_index]
                        
                    for gau_behave_index in range(num_gau_comp):
                        
                        b_update_nmr = 0
                        for point_index in range(MyDataSet.shape[0]):
                            b_update_nmr = b_update_nmr + b_value_matrix[point_index][gau_context_index][gau_behave_index]
                        
                        b_update = b_update_nmr/(b_update_demo+alpha_log)
                        
                        
                        mapping_matrix[gau_behave_index][gau_context_index] = b_update
                        
                        
            expected_loglikelihood_new = expected_loglikelihood
            
            print("num_ieteration")
            print(num_ieteration)
            print("expected_loglikelihood_new")
            print(expected_loglikelihood_new)
            print("expected_loglikelihood_old")
            print(expected_loglikelihood_old)
                            
            # if (expected_loglikelihood_new - expected_loglikelihood_old) < 0.01 and num_ieteration > 10:
            if (expected_loglikelihood_new - expected_loglikelihood_old) < 0.01 and num_ieteration > 99:
                print("no more improvement for log likelihood")
                break  
            
            ##Update old threshold
            expected_loglikelihood_old = expected_loglikelihood_new

        ##Save learned model, i.e., the mapping matrix in CAD-full model
        np.save(FilePath_MappingMatrix, mapping_matrix)
    
                    
    ##############################################################################
    ##Step 3: Use learned CAD-full model to perform contextual anomaly detection
    ##In this step, we do not split the dataset for CV purpose
    mapping_matrix = np.load(FilePath_MappingMatrix)
    #print(mapping_matrix.shape[0])
    
    raw_anomaly_score_vec = []      
    ##Calculate anomaly score for each data point
    for point_index in range(MyDataSet.shape[0]):
        test_point = MyDataSet.iloc[point_index]
        test_point_context = test_point[MyContextList]
        test_point_behave = test_point[MyBehaveList]
        
        raw_anomaly_score = 0
        
        for gau_context_index in range(num_gau_comp):
            ##Calculate the weight of a certain gaussian distribution for contextual features
            proba_gau = GMM_weight[gau_context_index] ## 1st anomaly item
            
            likelihood_sum = 0 ## 2nd anomaly item
            for gau_behave_index in range(num_gau_comp):
                
                ###Find the mean and covariance of a gaussian distribution for behavioral attributes
                mean_behave = GMM_mean[gau_behave_index]
                mean_behave = mean_behave[behave_index]
                var_behave = GMM_var[gau_behave_index]
                var_behave = var_behave[:,behave_index]
                var_behave = var_behave[behave_index,:]
                
                ##Calculate the likelihood of a given data point produced by a certain gaussian distribution for behavioral features
                likelihood_behave = multivariate_normal.pdf(test_point_behave, mean=mean_behave, cov=var_behave) #2-1 anomaly item
                proba_mapping = mapping_matrix[gau_behave_index][gau_context_index]#2-2 anomaly item       
                likelihood_sum = likelihood_sum + likelihood_behave*proba_mapping
           
            raw_anomaly_score = raw_anomaly_score + proba_gau*likelihood_sum
        
        raw_anomaly_score_vec.append(raw_anomaly_score)
            
    raw_anomaly_score_vec = [-i for i in raw_anomaly_score_vec]
    
    from scipy import stats
        
    percentiles = [stats.percentileofscore(raw_anomaly_score_vec, i) for i in raw_anomaly_score_vec]
        
    MyDataSet["ground_truth"] = RawDataSet["ground_truth"]
    
    MyDataSet["raw_anomaly_score"] = raw_anomaly_score_vec    
    
    MyDataSet["anomaly_score"] = percentiles
     
    
    #################################################################################
    ##Step 4: Evaluate performance
    
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
    
    t1 = time() # to record time 
    
    # print("pr_auc, roc_auc, p@n")
    t1 = time() # to record time 
    duringtime1 = round(t1 - t0, ndigits=4)
    duringtime2 = round(t1 - t0, ndigits=4)
    
    print(my_pr_auc, my_roc_score, P_at_n_value, duringtime1, duringtime2)
    
    return my_pr_auc, my_roc_score, P_at_n_value, duringtime1, duringtime2


# =============================================================================
# Step II. testing COD with real world dataset
# =============================================================================

################################################
##Step 1: Load dataset and set parameters 
################################################

# RawDataSet = pd.read_csv("/Users/zlifr/Desktop/HHBOS/Data2/EnergyGene.csv", sep=",")
# MyContextList = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
# MyBehaveList = ['Y1', 'Y2']
# ##Number of gaussian components
# num_gau_comp = 5
# ##A number to avoid log(0) or devided by zero
# alpha_log = 0.001
# ##An indicator of whether the model has been already learned 
# is_model_learned = 1
# ##Filepath to save or load mapping matrix
# FilePath_MappingMatrix = r'/Users/zlifr/Desktop/HHBOS/TrainedModel/COD/mapping_matrix_EnergyGene.npy'

################################################
##Step 2: call COD to get results
################################################
# COD(RawDataSet, MyContextList, MyBehaveList, num_gau_comp, alpha_log, 0, r'')
    






