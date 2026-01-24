# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:33:24 2023

@author: USER

Script that performs HP tuning on the data. Script is able to perform
HP tuning on both RF and XGB models, different lead times, and different
spatial data clusters (county, country, lhz, all). For the final model
version, we run the HP tuning on all counties aggregated. 
Start of the script (until the HP PARAM TUNING section) is the
same as the ML_execution_HPC script, and reads/loads the input data in the same way.

"""
#####################################################################################################################
################################################### IMPORT PACKAGES  ###################################################
#####################################################################################################################
from time import sleep
from tqdm.auto import tqdm
import os
import random as python_random
import numpy as np
import datetime 

# plot packages
import pandas as pd
import seaborn as sns
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import mapclassify as mc
import matplotlib.pyplot as plt


# SKLEARN 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.metrics import PredictionErrorDisplay # only works with version 1.2 

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.feature_selection import SelectKBest

# XGBOOST
from xgboost import XGBRegressor
# others
import pydot
import scipy as sp
import time

from src.ML_functions import *


#######################################################################################################################
################################################### DESIGN VARIABLES  ####N###############################################
#######################################################################################################################

model_list= ['xgb']# 'xgb' or 'rf'
region_list= ['HOA']#['HOA']#['Kenya', 'Somalia', 'Ethiopia']
aggregations=['all']#['all','county','cluster']
experiment_list= ['NAME_20'] # last two numbers indicate train_test ratio 

design_variables = [(experiment, model_type, aggregation, region) for experiment in experiment_list for model_type in model_list for aggregation in aggregations for region in region_list]
leads=[0,1,2,3,4,8,12] # check:  12 lead creates quite some nan's in the training data 
search_method= 'mini' # Hyperparameter space search method  --> 'exhausive' or 'mini'
CV='TSS' # TSS or KFOLD
# extra options (not included in the final model version)
limited_features=False
with_WPG=False


for experiment, model_type, aggregation, region in design_variables:


 
    #######################################################################################################################
    ################################################### DATA PREPROCESSING  ####N###############################################
    #######################################################################################################################



    
    print ('Current experiment: ', experiment, 'Current model: ', model_type, 'Current aggregation: ', aggregation, 'Current region: ', region)
    time.sleep(1)
    traintest_ratio=int(experiment[-2:])/100
    ################################################### SET DATA FOLDERS ###################################################

    BASE= 'PATH/ML/' #'PATH/ML/'#'C:/Users/USER/Documents/ML/' #
    DATA_FOLDER= BASE+'input_collector/'

    ################################################### LOAD INPUT DATA ###################################################
    os.chdir(DATA_FOLDER)
    input_master=pd.read_csv('input_master.csv', index_col=0)
    input_master.index=pd.to_datetime(input_master.index) # convert index to datetime index
    # drop year column 
    input_master.drop('year', axis=1, inplace=True)
    
    # include WFP data (optional)
    if with_WPG==False:
        input_master=input_master[input_master.columns.drop(list(input_master.filter(regex='WPG')))]

    #######################################################################################################################
    ################################################### POOLING/ AGGREGATION LOOP #########################################
    #######################################################################################################################
    
    if region!='HOA':
        input_master=input_master[input_master['country']==region]
        input_master.drop('country', axis=1, inplace=True)

    ############## UNIT LOOP ##############
    if aggregation=='cluster':
        cluster_list=[ 'p','ap','other']
        print ('Clusters: ', cluster_list)
        # identify columns in input_master which are categorical
        
    elif aggregation=='all':
        units=['all']
        cluster_list= ['no_cluster'] 

    elif aggregation=='county':
        units=list(input_master['county'].unique())
        cluster_list=['no_cluster']

    
    for cluster in cluster_list:

        print ('Cluster: ', cluster)


        RESULT_FOLDER = BASE+'ML_results/'
        HP_RESULT_FOLDER= RESULT_FOLDER+'HP_results/'+aggregation+'_'+experiment+'_'+region+'_'+cluster+'_'+model_type+'/'


        if not os.path.exists(HP_RESULT_FOLDER):
            os.makedirs(HP_RESULT_FOLDER)
        else:
            # remove all content from folder
            files = os.listdir(HP_RESULT_FOLDER)
            for f in files:
                os.remove(os.path.join(HP_RESULT_FOLDER, f))


            

        for i, county in enumerate(units):
                print('County: ', county, 'in cluster:', cluster)

                if aggregation=='county': 
                    input_df2=input_master[input_master['county']==county]
                    input_df2.dropna(axis=1, how='all', inplace=True)# drop nan values when whole column is nan (CROP column)
                if aggregation=='cluster':
                    input_df2=input_master[input_master['lhz']==cluster]
                    input_df2.dropna(axis=1, how='all', inplace=True)
                    units=['cluster_%s'%cluster]
                if aggregation=='all':
                    input_df2=input_master.copy()
                    input_df2.dropna(axis=1, how='all', inplace=True)


                #######################################################################################################################
                ###################################################  DROP FEWS NAN VALUES #############################################
                ######################################################################################################################

                input_df2=input_df2[input_df2['FEWS_CS'].notna()] # no nans 


                #######################################################################################################################
                ###################################################  LEAD TIME LOOP   #########################################
                ######################################################################################################################

                for lead in leads:
                    print ('Current lead time: ', lead) 
                    
                    ############## Select Lead ##############
                    input_df3=input_df2[input_df2['lead']==lead]
                    
                    # sort on datetime index --> necessary to be able to execute train-test split by time... 
                    input_df3=input_df3.sort_index() # This shuffles the county order... Every unique datetime has a randomly shuffled order of counties
                    
                    ############## Store county df ##############
                    county_df=input_df3['county'] 

                    ############## Save stationarity forecast ##############
                    base1_preds=input_df3['base_forecast']

                    ############## one-hot encoding ########################
                    #if aggregation=='all':
                    #    input_df3=pd.get_dummies(input_df3, columns=['lhz'], prefix='', prefix_sep='') # use lhz as additional feature
                    
                    if region=='HOA':
                        input_df3=pd.get_dummies(input_df3, columns=['country'], prefix='', prefix_sep='') # use countries as additional feature

                    ############## drop cat columns ########################
                    cat_cols=input_df3.select_dtypes(include=['object', 'category']).columns 
                    input_df3.drop(cat_cols, axis=1, inplace=True)
                    print('dropping cat cols: ', cat_cols)


                    ################################################## Keep only certain features ###########################################
                    #if limited_features==True:
                        #base_features=['FEWS_CS','lead', 'base_forecast']
                        #selected_features= ['tp', 'wd_12','sm_root','sm_root_4','NDVI_anom', 'NDVI_anom_crop', 'NDVI_anom_range','NDVI_anom_4', 'NDVI_anom_12', 'WVG', 'MEI', 'WVG_4', 'MEI_12', 'Maize (white)_Pewi','Maize (white)_Pewi_4','acled_fatalities', 'DL_area','CPI_F']
                        #selected_features=base_features+selected_features
                        #input_df3=input_df3[selected_features]

                        # input_df3.drop(drop_features, axis=1, inplace=True)
                        
                        #climate_features= ['tp', 'wd_12','sm_root','sm_root_4','NDVI_anom', 'NDVI_anom_crop', 'NDVI_anom_range','NDVI_anom_4', 'NDVI_anom_12', 'WVG', 'MEI', 'WVG_4', 'MEI_12', 'FEWS_CS_lag1', 'FEWS_CS_lag2'']


                    
                    #######################################################################################################################
                    ###################################################  TRAIN-TEST SPLIT   ##############################################
                    ######################################################################################################################


                    ############################################### Create features- labels ###############################################
                    labels=pd.Series(input_df3['FEWS_CS'])
                    features=input_df3.drop(['FEWS_CS','lead', 'base_forecast'], axis=1) 
                    feature_list = list(features.columns) # Saving feature names for later use
                    
                    print(feature_list, len(feature_list))
                    ############################################### Training-test split ###############################################   

                    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = traintest_ratio, shuffle=False) #25% of the data used for testing (38 time steps)   random_state = 42. if random state is not fixed, performance is different each time the code is run.
                    # print the timestamps of the train_features, test_features, train_labels, test_labels dataframes 
                    print ('train_features: ', train_features.index[0], train_features.index[-1])
                    print ('test_features: ', test_features.index[0], test_features.index[-1])
                    print ('train_labels: ', train_labels.index[0], train_labels.index[-1])
                    print ('test_labels: ', test_labels.index[0], test_labels.index[-1])


                    #################################################### ACCESS AND CHECK KFOLD CONSTRUCTOR ####################################################
                    # construct a cv method 
                    if CV=='KFOLD':
                        cv_method = KFold(n_splits=5)
                    else:
                        cv_method = TimeSeriesSplit(n_splits=5)
                    # see how cv create folds based on train_features and test_features
                    for i, (train_index, test_index) in enumerate(cv_method.split(train_features)):
                            print(f"Fold {i}:")
                            # extract train values based on the train_index
                            train_values = train_features.iloc[train_index]
                            # extract test values based on the test_index
                            test_values = train_features.iloc[test_index]
                            print (f"all training dates= {list(train_values.index.astype('str').values)}")
                            print (f"all testing dates= {list(test_values.index.astype('str').values)}")
                            print(f"train values: {train_values}")
                            print(f"test values: {test_values}")



                    #######################################################################################################################
                    ################################################ HP PARAM TUNING  #################################################
                    ######################################################################################################################


                    #######################################################################################################################
                    ################################################ FOR XGB MODEL ######################################################
                    ######################################################################################################################

                    os.chdir(HP_RESULT_FOLDER)
                    if model_type=='xgb':
                        
                        # First create the base model to tune
                        xgboost = XGBRegressor()
                        


                        ###################################### MINI HP SEARCH #######################################  
                        if search_method=='mini':
                            print('HP tuning.... mini search')                                             
                            
                            grid = { 
                                # Learning rate shrinks the weights to make the boosting process more conservative
                                "learning_rate": [0.001,0.01,0.05,0.1,0.3],
                                # Maximum depth of the tree, increasing it increases the model complexity.
                                "max_depth": [4, 6, 10],
                                'n_estimators': [100,200,400,600,800],
                                }
                            
                            
                            # Excecute grid searchCV 
                            grid_xgb= GridSearchCV(estimator = xgboost,
                                                    param_grid = grid, 
                                                    scoring='neg_mean_squared_error',
                                                    cv = cv_method, 
                                                    verbose=4, 
                                                    n_jobs = -1, 
                                                    return_train_score=True, 
                                                    error_score='raise')
                        
                        ###################################### EXHAUSIVE HP SEARCH ####################################### 
                        else: 
                            print('HP tuning.... exhausive search')

                            grid = { 
                                "learning_rate": [0.001, 0.01, 0.05,0.3],# --> include 0.001! 0.3 seems obsolete, but it is the default... . Makes the model more robust by shrinking the weights on each step. Typical final values to be used: 0.01-0.2.  0.05 to 0.3 should work for different problems
                                "max_depth": [3,4, 6, 10],# Maximum depth of the tree, increasing it increases the model complexity. This should be between 3-10
                                "gamma": [0.0,0.1,0.3],# Gamma specifies the minimum loss reduction required to make a split. Default 0
                                "colsample_bytree": [0.5,0.7,1],# Percentage of columns to be randomly samples for each tree. 
                                "reg_alpha": [0.01,0, 1],# [0, 0.001, 0.005, 0.01, 0.05],  reg_alpha eprovides l1 regularization to the weight, higher values rsult in more conservative models. Penelizes more complex models.  https://www.geeksforgeeks.org/regularization-in-machine-learning/
                                "reg_lambda": [0.1, 1, 10],# reg_lambda provides l2 regularization to the weight, higher values result in more conservative models.  Though many data scientists don’t use it often, it should be explored to reduce overfitting. 
                                'n_estimators': [100,150,200,500],
                                'subsample': [0.5,0.7,1], #Denotes the fraction of observations to be random samples for each tree.Lower values make the algorithm more conservative and prevent overfitting, but too small values might lead to under-fitting.Typical values: 0.5-1
                                'min_child_weight': [1, 5, 10], # Higher values prevent a model from learning relations that might be highly specific to the particular sample selected for a tree.Too high values can lead to under-fitting; hence, it should be tuned using CV.
                                }   
                            #total numer of combinations: 4*4*3*3*3*3*4*3*3= 46656 
                            print('fitting 46656 models for L%s'%(lead))
                            grid_xgb= GridSearchCV(estimator = xgboost,
                                                    scoring='neg_mean_squared_error',
                                                    param_grid = grid, 
                                                    cv = cv_method, 
                                                    verbose=4, 
                                                    n_jobs = -1, 
                                                    return_train_score=True, 
                                                    error_score='raise')
                        
                        grid_result_xgb=grid_xgb.fit(train_features, train_labels)
                        grid_result_xgb_df = pd.DataFrame(grid_result_xgb.cv_results_)
                        grid_result_xgb_df=grid_result_xgb_df.sort_values(by='rank_test_score')
                        # make an extra column which is the training-test score difference
                        grid_result_xgb_df['generalization_score']=grid_result_xgb_df['mean_train_score']-grid_result_xgb_df['mean_test_score']
                        grid_result_xgb_df['gen_score_rank']=grid_result_xgb_df['generalization_score'].rank(ascending=True)
                        grid_result_xgb_df.to_excel('CV_results_XGB_L%s.xlsx'%(lead))





                    #######################################################################################################################
                    ################################################ FOR RF MODEL  ######################################################
                    #####################################################################################################################

                    if model_type=='rf':
                        
                        # Number of trees in random forest
                        n_estimators = [200,500,1000]
                        # Number of features to consider at every split
                        max_features = [None, 'sqrt']
                        # Maximum number of levels in tree
                        max_depth = [4,10,15]
                        max_depth.append(None)
                        # Minimum number of samples required to split a node
                        min_samples_split = [2,5]
                        # Minimum number of samples required at each leaf node
                        min_samples_leaf = [1,2,5]

                        # total combinations= 270 
                        
                        # Create the random grid
                        grid = {'n_estimators': n_estimators, #add criterion? 
                                    'max_features': max_features, # max features to consider for each split 
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf}



                        # First create the base model to tune
                        ini = RandomForestRegressor()
                        

                        ###################################### EXHAUSIVE HP SEARCH #######################################
                        if search_method=='exhausive':
                            # Exhaustive search of parameters
                            print('HP tuning.... exhausive search')
                            grid_rf= GridSearchCV(estimator = ini,
                                                    param_grid = grid, 
                                                    cv = cv_method, 
                                                    verbose=4, 
                                                    n_jobs = -1, 
                                                    return_train_score=True, 
                                                    error_score='raise')
                        
                        ###################################### MINI HP SEARCH #######################################

                        else: 
                            # Random search of parameters
                            print('HP tuning.... random search')
                            grid_rf= RandomizedSearchCV(estimator = ini, 
                                                    param_distributions = grid, 
                                                    n_iter = 1000, 
                                                    cv = 5, 
                                                    verbose=4, 
                                                    random_state=42, 
                                                    n_jobs = -1, 
                                                    return_train_score=True)
                            
                        grid_result_rf=grid_rf.fit(train_features, train_labels)
                        grid_result_rf_df = pd.DataFrame(grid_result_rf.cv_results_)
                        grid_result_rf_df=grid_result_rf_df.sort_values(by='rank_test_score')
                        # make an extra column which is the training-test score difference
                        grid_result_rf_df['generalization_score']=grid_result_rf_df['mean_train_score']-grid_result_rf_df['mean_test_score']
                        grid_result_rf_df['gen_score_rank']=grid_result_rf_df['generalization_score'].rank(ascending=True)
                        
                        



                        #######################################################################################################################
                        ################################################ SAVE RESULTS ######################################################
                        ######################################################################################################################
                        
                        grid_result_rf_df.to_excel('CV_results_%s_L%s_%s.xlsx'%(model_type,lead,cluster))
                    

                    print ('HP tuning done for lead %s and %s'%(lead, cluster))
