# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:33:24 2023

@author: Tim Busker 

This script executes the machine learning models on the High Performance Cluster (HPC).
It uses the input_data.csv file created by the input_collector.py and feature_engineering.py scripts. 
Current implementation allows for a random forest model or an XGBoost model, on multiple spatial levels
(county, country, livelihood zone, all). It was also tested whether the variation of FEWS IPC was
 a good way of pooling the counties. This was not the case. 


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

## FEATURE EXPLANATION
import shap

# WANDB 
import wandb

#os.chdir('C:/Users/USER/Documents/ML/scripts')
from ML_functions import *




################################################### DESIGN VARIABLES ###################################################
 

model_list= ['xgb'] # xgb or rf
region_list= ['HOA'] # HOA or ['Kenya', 'Somealia', 'Ethiopia'] 
aggregations=['cluster']# 'all','county','cluster', 'variation'
experiment_list= ['RUN_FINAL_PAPER_RF_20'] # Name of the experiment. Folders will be created with this name. last two numbers indicate train_test ratio 
leads=[0,1,2,3,4,8,12] # check:  12 lead creates quite some nan's in the training data 

# selecting the HP method (hyperparameteres from  both methods are found in the HP_tuning_HPC.py script). 
XGB_HP='Simple' # 'Simple' or 'Extensive'. 
selection_method='generalization' # 'absolute_best' or 'generalization'

# extra variables which define extra tests executed during the model's test phase, but not in the final model.
with_WPG=False # option to add the West-Pacific Gradient as a feature.
with_train_var=False  # flag to add the train variance as a feature
limited_features= False  # flag to limit the features to a selected set of features. 
limit_time=False # flag to limit the time period to 2011-2022 --> check for other FEWS version
# start and end date of the analysis in case limit_time is True
start_date='2011-03-01 00:00:00' 
end_date='2022-06-01 00:00:00' 
drop_on_stdev=False # flag to drop counties with little to no variation in FEWS (stdev<0.01)



# Create design variables. 
design_variables = [(experiment, model_type, aggregation, region) for experiment in experiment_list for model_type in model_list for aggregation in aggregations for region in region_list]



    #######################################################################################################################
    ################################################### PRE-PROCESSING ####N###############################################
    #######################################################################################################################

for experiment, model_type, aggregation, region in design_variables: # Loop over all design variables. Note that these variables are a list (e.g. model_list) and therefore the script allows for looping over multiple experiments. 
    
    
    print ('Current experiment: ', experiment, 'Current model: ', model_type, 'Current aggregation: ', aggregation, 'Current region: ', region)
    time.sleep(1)

    ################################################### SET DATA FOLDERS ###################################################
    BASE= 'PATH/ML/' #'PATH/ML/'#'C:/Users/USER/Documents/ML/' #
    DATA_FOLDER= BASE+'input_collector/'


    ################################################### SET TRAIN-TEST RATIO ###################################################
    traintest_ratio=int(experiment[-2:])/100 # train test ratio is the last two numbers of the experiment name.

    ################################################### LOAD INPUT DATA ###################################################
    os.chdir(DATA_FOLDER)
    input_master=pd.read_csv('input_master.csv', index_col=0)
    
    
    ################################################### DATA PREPROCESSING ###################################################

    input_master.index=pd.to_datetime(input_master.index) # convert index to datetime index

    # limit time period 
    if limit_time==True: 
        input_master=input_master[input_master.index>pd.Timestamp(start_date)]
        #input_master=input_master[input_master.index<pd.Timestamp('2022-06-01 00:00:00')]

    
    # drop year column and columns which column name contains 'WPG'
    input_master.drop('year', axis=1, inplace=True)

    # include WFP data (optional)
    if with_WPG==False:
        input_master=input_master[input_master.columns.drop(list(input_master.filter(regex='WPG')))]
    
    


    ################################################### DROP COUNTIES WITH LITTLE TO NO VARIATION  ###################################################
    stdev_dataframe=pd.DataFrame()

    # drop counties with little to no variation in FEWS (stdev<0.01)
    for zone in input_master.county.unique():
        zone_df=input_master[input_master['county']==zone]
        zone_df=zone_df[zone_df['FEWS_CS'].notna()]
        zone_df=zone_df[zone_df['lead']==0]
        zone_fews=zone_df['FEWS_CS'] # original FEWS TS 

        if drop_on_stdev==True:
            if zone_fews.std()<=0.01:
                print ('dropping county: ', zone, 'in country', zone_df['country'].iloc[0])
                input_master=input_master[input_master['county']!=zone]
        
        

        # calculate train set stdev per county. This can be used for pooling or for adding the stdev as a feature.

        first_date= pd.Timestamp('2009-07-01 00:00:00') #zone_fews.index[0]
        last_date= pd.Timestamp(end_date) # zone_fews.index[-1]
        date_list= pd.date_range(start=first_date, end=last_date, freq='MS')
        cut_off_time= (1-traintest_ratio)*len(date_list)
        cut_off_time= date_list[int(cut_off_time)]

        train_set_fews=zone_fews[zone_fews.index<cut_off_time]
        test_set_fews=zone_fews[zone_fews.index>=cut_off_time]
        data_points_train= len(train_set_fews)
        data_points_test= len(test_set_fews)
        data_points_all= len(zone_fews)
        
        stdev_dataframe=pd.concat([stdev_dataframe, pd.DataFrame({'county':zone, 'stdev_train':train_set_fews.std(ddof=0), 'data_points_train':data_points_train, 'stdev_test': test_set_fews.std(ddof=0), 'data_points_test':data_points_test, 'stdev_all':zone_fews.std(ddof=0), 'data_points_all':data_points_all}, index=[0])], axis=0)
    
    # sort dataframe by stdev
    stdev_dataframe.sort_values(by='stdev_train', inplace=True)

    # create three groups with low, medium and high variation based on train stdev 
    stdev_dataframe['group_train']=pd.qcut(stdev_dataframe['stdev_train'], 3, labels=['low', 'medium', 'high'])
    stdev_dataframe.to_excel('stdev_dataframeV3.xlsx')

    stdev_all= input_master.copy()
    stdev_all=stdev_all[stdev_all['FEWS_CS'].notna()]
    stdev_all_value=stdev_all[stdev_all['lead']==0]
    stdev_all_value=stdev_all_value['FEWS_CS']
    stdev_all_value=stdev_all_value.std(ddof=0)
    print('stdev of all counties together= ', stdev_all_value)

    

    if aggregation=='variation': # Pooling/aggregation based on variation in train set
        input_master['group_train']=input_master['county'].map(stdev_dataframe.set_index('county')['group_train'])

    if with_train_var==True:  # add the train set stdev as a feature to the input_master
        input_master['stdev_train']=input_master['county'].map(stdev_dataframe.set_index('county')['stdev_train'])

    

    #######################################################################################################################
    ################################################### POOLING/ AGGREGATION LOOP #########################################
    #######################################################################################################################
    
    if region!='HOA':
        input_master=input_master[input_master['country']==region]
        input_master.drop('country', axis=1, inplace=True)

    if aggregation=='cluster':
        cluster_list=[ 'p','ap','other']
        print ('Clusters: ', cluster_list)
        # identify columns in input_master which are categorical
        
    elif aggregation=='variation':
        cluster_list=['low', 'medium', 'high']
        print ('Clusters: ', cluster_list)
    
    elif aggregation=='all':
        units=['all']
        cluster_list= ['no_cluster'] 

    elif aggregation=='county':
        units=list(input_master['county'].unique())
        cluster_list=['no_cluster']

    
    for cluster in cluster_list:

        ################################################### Create storage dataframes ###################################################
        features_df_full=pd.DataFrame()
        preds_storage=pd.DataFrame()
        eval_stats=pd.DataFrame()

        shap_values_master= pd.DataFrame()
        shap_data_master= pd.DataFrame()
        shap_base_master= pd.DataFrame()
        train_master= pd.DataFrame()
        print ('Cluster: ', cluster)

        ################################################### SET RESULT FOLDERS ###################################################
        RESULT_FOLDER = BASE+'ML_results/'
        #HP_RESULT_FOLDER= RESULT_FOLDER+'HP_results/'+aggregation+'_'+experiment+'_'+region+'_'+cluster+'/'
        PLOT_FOLDER= RESULT_FOLDER+'plots/'+aggregation+'_'+experiment+'_'+region+'_'+cluster+'/'
        EVAL_PLOT_FOLDER= RESULT_FOLDER+'eval_plots/'+aggregation+'_'+experiment+'_'+region+'_'+cluster+'/'
        TREE_FOLDER= BASE+'trees/'+aggregation+'_'+experiment+'_'+region+'_'+cluster+'/'
        VECTOR_FOLDER=DATA_FOLDER+'Vector/'#'C:/Users/USER/Documents/Forecast_action_analysis/vector_data/'
        #WANDB_FOLDER= RESULT_FOLDER+'WANDB/'

        


        # check if the  required folders exist. If not, create them.
        if not os.path.exists(PLOT_FOLDER):
            os.makedirs(PLOT_FOLDER)
        else: 
            # remove all content from plot folder
            files = os.listdir(PLOT_FOLDER)
            for f in files:
                os.remove(os.path.join(PLOT_FOLDER, f))

        if not os.path.exists(EVAL_PLOT_FOLDER):
            os.makedirs(EVAL_PLOT_FOLDER)
        else:
            # remove all content from plot folder
            files = os.listdir(EVAL_PLOT_FOLDER)
            for f in files:
                os.remove(os.path.join(EVAL_PLOT_FOLDER, f))

        if not os.path.exists(TREE_FOLDER):
            os.makedirs(TREE_FOLDER)
        else:
            # remove all content from tree folder
            files = os.listdir(TREE_FOLDER)
            for f in files:
                os.remove(os.path.join(TREE_FOLDER, f))

        



        if aggregation=='cluster':
            input_df2=input_master[input_master['lhz']==cluster]
            input_df2.dropna(axis=1, how='all', inplace=True)
            units=['cluster_%s'%cluster]

        if aggregation=='variation':
            input_df2=input_master[input_master['group']==cluster]
            input_df2.dropna(axis=1, how='all', inplace=True)
            units=['cluster_%s'%cluster]




        #################################################################################################################
        ################################################### ADMIN UNITS OR POOLED CLUSTER LOOP ##########################
        #################################################################################################################

        
        for i, county in enumerate(units):
                print('County: ', county, 'in cluster:', cluster)
                
                if aggregation=='county': 
                    input_df2=input_master[input_master['county']==county]
                    input_df2.dropna(axis=1, how='all', inplace=True)# drop nan values when whole column is nan (CROP column)
                elif aggregation=='all':
                    input_df2=input_master.copy()
                    
                    

                    


                ############## Drop rows with FEWS nan values  ##############
                input_df2=input_df2[input_df2['FEWS_CS'].notna()] 


                #################################################################################################################
                ################################################### LEAD TIME LOOP## ############################################
                #################################################################################################################

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

                    ############## one-hot encoding #######################
                    #input_df3=pd.get_dummies(input_df3, columns=['county'], prefix='', prefix_sep='')

                    if aggregation=='all':
                        input_df3=pd.get_dummies(input_df3, columns=['lhz'], prefix='', prefix_sep='') # use lhz as additional feature
                    
                    if region=='HOA':
                        input_df3=pd.get_dummies(input_df3, columns=['country'], prefix='', prefix_sep='') # use countries as additional feature

                    ############## drop cat columns ########################
                    cat_cols=input_df3.select_dtypes(include=['object', 'category']).columns 
                    input_df3.drop(cat_cols, axis=1, inplace=True)
                    
                    

                    ############################################### Create features- labels ###############################################
                    labels=pd.Series(input_df3['FEWS_CS'])

                    ################################################## Keep only certain features (optional) ###########################################
                    
                    if limited_features==True:
                        print ('limited features disabled now')
                        exit()
                        #base_features=['FEWS_CS','lead', 'base_forecast']
                        #selected_features= ['tp', 'wd_12','sm_root','sm_root_4','NDVI_anom', 'NDVI_anom_crop', 'NDVI_anom_range','NDVI_anom_4', 'NDVI_anom_12', 'WVG', 'MEI', 'WVG_4', 'MEI_12', 'Maize (white)_Pewi','Maize (white)_Pewi_4','acled_fatalities', 'DL_area','CPI_F']
                        #selected_features=base_features+selected_features
                        #input_df3=input_df3[selected_features]

                        # make var drop_features, which contains all column names in input_df3 which do contain 'FEWS_CS'
                        #drop_features=[col for col in input_df3.columns if 'FEWS_CS' in col]
                        #input_df3.drop(drop_features, axis=1, inplace=True)
                        
                        #climate_features= ['tp', 'wd_12','sm_root','sm_root_4','NDVI_anom', 'NDVI_anom_crop', 'NDVI_anom_range','NDVI_anom_4', 'NDVI_anom_12', 'WVG', 'MEI', 'WVG_4', 'MEI_12', 'FEWS_CS_lag1', 'FEWS_CS_lag2'']



                    features=input_df3.drop(['lead', 'base_forecast','FEWS_CS'], axis=1) 
                    feature_list = list(features.columns) # Saving feature names for later use
                    
                    print(feature_list, len(feature_list))
                   
                    #################################################################################################################
                    ################################################### TRAIN-TEST SPLIT ###########################################
                    ################################################################################################################# 
                    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = traintest_ratio, shuffle=False)# 25% of the data used for testing (38 time steps)   random_state = 42. if random state is not fixed, performance is different each time the code is run.
                    # print the timestamps of the train_features, test_features, train_labels, test_labels dataframes 
                    print ('train_features: ', train_features.index[0], train_features.index[-1])
                    print ('test_features: ', test_features.index[0], test_features.index[-1])
                    print ('train_labels: ', train_labels.index[0], train_labels.index[-1])
                    print ('test_labels: ', test_labels.index[0], test_labels.index[-1])

                    print(train_features, test_features, train_labels, test_labels)


                    ############################################### Correlation analysis ###################################   
                    # correlations= features_l.drop(features_l.columns[-2:], axis=1).corr()

                    # mask = []
                    # for i in range(len(correlations.columns)):
                    #     mask_i = []
                    #     for j in range(len(correlations.columns)):
                    #         if i<j:
                    #             mask_i.append(True)
                    #         else: 
                    #             mask_i.append(False)
                    #     mask.append(mask_i)
                    # mask = np.array(mask)

                    # plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
                    # sns.set(font_scale=1.2)
                    # sns.heatmap(correlations,cmap='coolwarm',
                    #             center = 0, 
                    #             annot=True,
                    #             fmt='.1g',
                    #             annot_kws={"fontsize":8},
                    #             mask=mask)
                    
                    # plt.title('Correlation matrix for %s months lead, county= %s'%(lead, county))
                



                    
                    #################################################################################################################
                    ################################################### DETERMINE HYPERPARAMETERS ####################################
                    #################################################################################################################


                    ########################### Random Forest Model ###########################

                    # created on 27-07-2023 and indeed best for RF model. 
                    if lead==0:
                        RF_params={'n_estimators': 200, 'min_samples_split': 5,'min_samples_leaf': 1,'max_features': None, 'max_depth': 4,'random_state': 42}
                    if lead==1:
                        RF_params={'n_estimators': 500, 'min_samples_split': 2,'min_samples_leaf': 2,'max_features': None, 'max_depth': 4,'random_state': 42}
                    if lead==2:
                        RF_params={'n_estimators': 200, 'min_samples_split': 2,'min_samples_leaf': 5,'max_features': None, 'max_depth': 10,'random_state': 42}
                    if lead==3:
                        RF_params={'n_estimators': 500, 'min_samples_split': 5,'min_samples_leaf': 5,'max_features': 'sqrt', 'max_depth': 10,'random_state': 42}
                    if lead==4:
                        RF_params={'n_estimators': 500, 'min_samples_split': 5,'min_samples_leaf': 5,'max_features': 'sqrt', 'max_depth': 10,'random_state': 42}
                    if lead==8:
                        RF_params={'n_estimators': 500, 'min_samples_split': 2,'min_samples_leaf': 5,'max_features': 'sqrt', 'max_depth': 10,'random_state': 42}
                    if lead==12:
                        RF_params={'n_estimators': 200, 'min_samples_split': 2,'min_samples_leaf': 1,'max_features': 'sqrt', 'max_depth': 10,'random_state': 42}


                    ########################## XGBoost Model ###########################


                    ########## Simple HP parameters ##########

                    if XGB_HP=='Simple':

                        XGB_params={'n_estimators': 400, #100 | 400 # uppdated 26-07-23 --> checked and indeed best 
                                    'max_depth':  4, # 6 | 4
                                    'learning_rate':  0.01, #0.1 | 0.05 | 0.01
                                    'random_state': 42,  
                                    }
                    ########## Extensive HP parameters ##########

                    # Absolute best parameters for each lead time.
                    else:  
                        if selection_method=='absolute_best':
                            if lead==0:
                                XGB_params={'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 10, 'n_estimators': 150, 'reg_alpha': 0, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==1:
                                XGB_params={'colsample_bytree': 1, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 10, 'n_estimators': 500, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 0.5}
                            if lead==2:
                                XGB_params={'colsample_bytree': 1, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 5, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.5}
                            if lead==3:
                                XGB_params={'colsample_bytree': 0.7, 'gamma': 0.3, 'learning_rate': 0.01, 'max_depth': 10, 'min_child_weight': 5, 'n_estimators': 500, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 0.5}
                            if lead==4:
                                XGB_params={'colsample_bytree': 0.7, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 10, 'n_estimators': 150, 'reg_alpha': 0, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==8:
                                XGB_params={'colsample_bytree': 0.7, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 5, 'n_estimators': 500, 'reg_alpha': 0.01, 'reg_lambda': 10, 'subsample': 0.7}
                            if lead==12:
                                XGB_params={'colsample_bytree': 0.5, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}

                    # Best generalization parameters for each lead time.
                        else: 
                            if lead==0:
                                XGB_params={'colsample_bytree': 1, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 100, 'reg_alpha': 1, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==1:
                                XGB_params={'colsample_bytree': 1, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 500, 'reg_alpha': 1, 'reg_lambda': 0.1, 'subsample': 0.5}
                            if lead==2:
                                XGB_params={'colsample_bytree': 1, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 100, 'reg_alpha': 1, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==3:
                                XGB_params={'colsample_bytree': 0.5, 'gamma': 0.3, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 500, 'reg_alpha': 0, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==4:
                                XGB_params={'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 150, 'reg_alpha': 1, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==8:
                                XGB_params={'colsample_bytree': 1, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 500, 'reg_alpha': 1, 'reg_lambda': 10, 'subsample': 0.5}
                            if lead==12:
                                XGB_params={'colsample_bytree': 0.5, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}

                                


                    if model_type=='rf':                    
                        ini = RandomForestRegressor(n_estimators = RF_params['n_estimators'], 
                                                    min_samples_split=RF_params['min_samples_split'],
                                                    min_samples_leaf=RF_params['min_samples_leaf'],
                                                    max_features=RF_params['max_features'],
                                                    max_depth=RF_params['max_depth'],
                                                    #bootstrap=RF_params['bootstrap'],
                                                    random_state=RF_params['random_state'],
                                                    )
                    else: 
                        if XGB_HP=='Simple':
                            ini = XGBRegressor(n_estimators = XGB_params['n_estimators'],
                                                max_depth=XGB_params['max_depth'],
                                                learning_rate=XGB_params['learning_rate'],
                                                random_state=XGB_params['random_state'],

                                                )
                        else:
                            ini= XGBRegressor(colsample_bytree= XGB_params['colsample_bytree'],
                                                gamma=XGB_params['gamma'],
                                                learning_rate=XGB_params['learning_rate'],
                                                max_depth=XGB_params['max_depth'],
                                                min_child_weight=XGB_params['min_child_weight'],
                                                n_estimators=XGB_params['n_estimators'],
                                                reg_alpha=XGB_params['reg_alpha'],
                                                reg_lambda=XGB_params['reg_lambda'],
                                                subsample=XGB_params['subsample'],
                                                random_state=42,
                                                )
                        
            
                    #################################################################################################################
                    ################################################### Fit the Models #############################################
                    #################################################################################################################
                    ini.get_params()
    
                    print ('fitting rf or xgb model...')
                    
                    model=ini.fit(train_features, train_labels) # fit the model --> training part of the model 
                    
                    print ('rf/xgb fitting done')
                    #################################################################################################################
                    ################################################### Make predictions  #############################################
                    #################################################################################################################
                    
                    predictions = ini.predict(test_features) ## rf/xgb predictions for the months with lead ahead 
                    
                    

                    
                    #################################################################################################################
                    ################################################### Trees visualization #############################################
                    #################################################################################################################
                    print ('start viz trees...')

                    if model_type=='rf':
                        #Pull out one tree from the forest
                        os.chdir(TREE_FOLDER)
                        tree = ini.estimators_[5]# extracts 1 random tree from the forest
                        export_graphviz(tree, out_file = 'tree_L%s_%s.dot'%(lead,county), feature_names = feature_list, rounded = True, precision = 3)# Use dot file to create a graph
                        (graph, ) = pydot.graph_from_dot_file('tree_L%s_%s.dot'%(lead,county))# Write dot file to a png file
                        graph.write_png('tree_L%s_%s.png'%(lead,county)) 
                        # remove the dot file
                        os.remove('tree_L%s_%s.dot'%(lead,county))

                    print ('viz trees done')
                    
                    #################################################################################################################
                    ########################################## SCI-Kit Learn Feature importance ######################################
                    #################################################################################################################
                    
                    ###################### SCI-Kit Learn ##########################
                    
                    if model_type=='rf':
                        # Get feature importances -->  computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
                        importances = list(ini.feature_importances_)# List of tuples with variable and importance
                        std = np.std([tree.feature_importances_ for tree in ini.estimators_], axis=0) # CHECK: not used 
                        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
                    
                    else: 
                        importances= model.feature_importances_          
                        sorted_idx = np.argsort(importances)
                        features_sorted=np.array(feature_list)[sorted_idx]
                        # make a tuple with the feature importances and the feature names
                        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_sorted, importances[sorted_idx])]


                    # append featuer importances to df
                    # make a dataframe with 3 columns: importance, county, lead. Importances is a np.array. Attach the county and lead to the array and then make a df from it.
                    append= pd.DataFrame(importances, index=feature_list, columns=['importance'])
                    append['county']=county
                    append['lead']=lead
                    features_df_full=pd.concat([features_df_full, append], axis=0)

                    # sort feature importances and print most important feature
                    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
                    main_feature= feature_importances[0][0]
                    feature_imp= feature_importances[0][1]


                    ################# Plot feature importances #################
                    os.chdir(PLOT_FOLDER)

                    #  Variable importances
                    forest_importances = pd.Series(importances, index=feature_list)
                    # sort importances
                    forest_importances.sort_values(ascending=False, inplace=True)

                    # keep only top 15 features
                    forest_importances=forest_importances[:20]

                    fig, ax = plt.subplots()
                    forest_importances.plot.bar(ax=ax) # yerr=std
                    
                    
                    ax.set_title("Feature importances using MDI for %s, L%s"%(county,lead))
                    ax.set_ylabel("Mean decrease in impurity")
                    ax.set_ylim(0,0.5)
                    fig.tight_layout()
                    plt.savefig('Variable_Importances_%s_L%s.png'%(county,lead), dpi=300, bbox_inches='tight')
                    #plt.show()
                    plt.close()
                    
                    #################################################################################################################
                    ################################################ SHAP VALUES ####################################################
                    #################################################################################################################

                    print ('start shap...')
                    explainer = shap.Explainer(model) # build the explainer object 
                    shap_values= explainer(train_features) # SHAP VALUES 
                    
                    # store shap values so that they can be used later
                    # values 
                    county_df2=pd.DataFrame(county_df) # units from original input_df
                    shap_values_df=pd.DataFrame(shap_values.values, columns=shap_values.feature_names, index=train_features.index) 
                    shap_values_df['cluster']=cluster
                    shap_values_df['lead']=lead
                    shap_values_df['county']=county_df2['county'].iloc[:len(train_features)].values# extract counties from county df. 
                    shap_values_df['aggregation']=aggregation
                    shap_values_df['region']=region
                    # base values
                    base_values_df=pd.DataFrame(shap_values.base_values, columns=['base_values'], index=train_features.index)
                    base_values_df['cluster']=cluster
                    base_values_df['lead']=lead
                    base_values_df['county']=county_df2['county'].iloc[:len(train_features)].values# extract counties from county df. 
                    base_values_df['aggregation']=aggregation
                    base_values_df['region']=region
                    # data
                    shap_data_df=pd.DataFrame(shap_values.data, columns=shap_values.feature_names, index=train_features.index)
                    shap_data_df['cluster']=cluster
                    shap_data_df['lead']=lead
                    shap_data_df['county']=county_df2['county'].iloc[:len(train_features)].values # extract counties from county df. 
                    shap_data_df['aggregation']=aggregation
                    shap_data_df['region']=region

                    # add to master df's
                    shap_data_master=pd.concat([shap_data_master, shap_data_df], axis=0)
                    shap_base_master=pd.concat([shap_base_master, base_values_df], axis=0)
                    shap_values_master=pd.concat([shap_values_master, shap_values_df], axis=0)

                    train_master=pd.concat([train_master, train_features], axis=0)

                    
                    # create new explainer from the df's --> this should be part of the script where I read the shap values and make the SHAP plots 
                    # shap_values_master.drop('cluster', axis=1, inplace=True)
                    # shap_explanation = shap.Explanation(values=shap_values_master.values, base_values=shap_base_master.values, feature_names=shap_values_master.columns, data=shap_data_master.values)
                    # print ('shap done')
                    # exit()

                    ################# Plot SHAP values #################                   
                    # bar plot --> https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html
                    fig, ax = plt.subplots(figsize=(10, 7))
                    shap.plots.bar(shap_values, show=False)
                    plt.title('Bar plot for %s, L%s'%(county,lead))
                    plt.savefig('shap_bar_plot_L%s_%s.png'%(lead,county), bbox_inches='tight')
                    plt.close()

                    # beeswarm plot
                    fig, ax = plt.subplots(figsize=(10, 7))
                    shap.plots.beeswarm(shap_values, show=False)
                    plt.title('Beeswarm plot for %s, L%s'%(county,lead))
                    plt.savefig('shap_beeswarm_plot_L%s_%s.png'%(lead,county), bbox_inches='tight')
                    plt.close()


                    # summarize plot 
                    #shap.summary_plot(shap_values, features=features_l, feature_names=features_l.columns, plot_type='bar',title="Feature importance for %s, L%s"%(county,lead), show=False)
                    #plt.savefig(PLOT_FOLDER+'/shap_summary_plot_L%s_%s.png'%(lead,county), bbox_inches='tight')
                    #plt.close()

                    # dependence plot
                    #ranks=[0,1,2,3,4]# Rank 1,2,3,4
                    #shap_vals_dep_plot=explainer.shap_values(train_features)

                    #conflict_index= list(train_features.columns).index('conflict')
                    
                    # for rank in ranks:
                    
                    #     shap.dependence_plot('rank(%s)'%rank, shap_vals_dep_plot, train_features, show=False)
                    #     plt.title('Dependence plot for %s, L%s'%(county,lead))
                    #     plt.savefig('shap_dependence_plot_L%s_%s_rank%s.png'%(lead,county,rank), bbox_inches='tight')
                    #     plt.close()




                    #################################################################################################################
                    ################################################ CREATE LOCAL PREDICTION  DATAFRAME##############################
                    #################################################################################################################   
                    # This includes the persistence (base1_preds) and seasonality (base2_preds) baseline forecasts. 

                    ############### retrieve back where in the dataframe the different counties are #########
                    test_features['county']=county_df2['county'].iloc[-len(test_features):].values
                    train_features['county']=county_df2['county'].iloc[:len(train_features)].values

                    
                    ################## Arrays to Dataframes ##################
                    test_labels=pd.DataFrame(test_labels)
                    train_labels=pd.DataFrame(train_labels) 
                    labels2=pd.DataFrame(labels)
                    predictions= pd.DataFrame(predictions)

                    ################## Reset index to allow for positional selection ##################
                    county_df2.reset_index(inplace=True) 
                    test_features.reset_index(inplace=True)
                    train_features.reset_index(inplace=True) 
                    test_labels.reset_index(inplace=True) # 
                    train_labels.reset_index(inplace=True) 
                    labels2.reset_index(inplace=True) 

                    

                    
                    ############################################### Filter predictions per county  ###############################################
                    for c in list(county_df.unique()):  
                        
                        ############### COUNTY INDEX IDENTIFIERS ################
                        county_index=county_df2[county_df2['county']==c].index #whole time series identifiers
                        test_county_index=test_features[test_features['county']==c].index # test time series identifiers
                        train_county_index=train_features[train_features['county']==c].index # train time series identifiers
                        
                        ############## COUNTY LABELS ################   
                        labels_county=labels2.iloc[county_index] # select county labels  
                        test_labels_county=test_labels.iloc[test_county_index].set_index('index') # select county test labels
                        
                        ############## COUNTY FEATURES ################
                        features_county=features.iloc[county_index] # select county features
                        test_features_county=test_features.iloc[test_county_index].set_index('index') # select county test features
                        
                        ############## COUNTY PREDICTIONS ################ 
                        predictions_county=predictions.iloc[test_county_index] # select county predictions
                        
                        #lr_preds_county=lr_preds.iloc[test_county_index]# select county predictions
                        

                        ################################################### UNIT BASELINE MODELS ###################################################
                        
                        ############################################### Baseline 1: use base_forecast as prediction ###############################################
                        base1_preds=base1_preds.iloc[-len(test_labels):] # make base1_preds same length as test_labels
                        base1_preds_county=base1_preds.iloc[test_county_index] # select county base predictions

                        ############################################### Baseline 2: use train set seasonality as baseline ###############################################
                        # take training values of fews, for index with years < 2016. Calculate seasonality based on that (after 2015 there is a change in the months included in the fews datset). NOt done yet?
                        
                        train_labels_county=train_labels.iloc[train_county_index] # select county train labels
                        train_labels_county.set_index('index', inplace=True) # set index to datetime index

                        seasonality=train_labels_county.groupby(train_labels_county.index.month).mean()
                        seasonality=seasonality.reindex([1,2,3,4,5,6,7,8,9,10,11,12,1]) # this includes the months which are nan 
                        seasonality=seasonality.interpolate(method='linear', axis=0) # interpolate the values using interpolation
                        seasonality=seasonality.iloc[:-1] # make seasonality one row shorter (remove the last row!)
                        seasonality=seasonality.reindex(test_labels_county.index.month) # Assign these monthly values (from seasonality) to the test set, based on the month of the index
                        seasonality.index=test_labels_county.index # set the index of the seasonality dataframe to the index of the fews_base_original_test dataframe
                        
                        base2_preds_county= seasonality.copy()
                        base2_preds_county.columns=['base2_preds']



                        ############################################### Create dataframe with predictions ###############################################

                        predictions_data = pd.DataFrame(data = {'date': test_labels_county.index, 
                                                                'observed': test_labels_county.values.flatten(),
                                                                'prediction': predictions_county.values.flatten(),
                                                                #'lr':lr_preds_county.values.flatten(), 
                                                                'base1_preds':base1_preds_county.values.flatten(), 
                                                                'base2_preds':base2_preds_county.values.flatten(), 
                                                                'lead': lead, 
                                                                'county':c}) # Dataframe with predictions and dates
                        
                        predictions_data.set_index('date', inplace=True)

                        preds_storage=pd.concat([preds_storage, predictions_data], axis=0) # preds_storage defined in beginning of script


                        #################################################################################################################
                        ################################################ COUNTY-LEVEL PLOTS##############################################
                        #################################################################################################################

                        #load target obs
                        target_obs=labels_county.set_index('index').copy()

                    

                        ####################### Time-series plot  #######################
                        fig, ax = plt.subplots(figsize=(10, 5))
                        # Plot the actual values
                        plt.plot(target_obs.index,target_obs['FEWS_CS'], 'b-', label = 'Observed FEWS class')# Plot the predicted values
                        
                        # plot base predictions
                        plt.plot(predictions_data.index, predictions_data['base1_preds'], 'go', label = 'base1 prediction (future based on last observed value)')
                        plt.plot(predictions_data.index, predictions_data['base2_preds'], 'yo', label = 'base2 prediction (future based on train data seasonality)')
                        
                        # plot rf predictions
                        plt.plot(predictions_data.index, predictions_data['prediction'], 'ro', label = '%s prediction'%(model_type))

                        #plot lr predictions
                        #plt.plot(predictions_data.index, predictions_data['lr'], 'mo', label = 'Linear regression prediction')
                        
                        plt.xticks(rotation = 'vertical'); 
                        plt.xlabel('Date'); plt.ylabel('FEWS IPC class'); plt.title('FEWS observations vs %s predictions for L=%s, county= %s'%(model_type,lead,c))
                        plt.legend(loc='best')
                        plt.savefig('TS_%s_L%s.png'%(c,lead), dpi=300, bbox_inches='tight')
                        #plt.show() 
                        plt.close()

                    

                        #################### TS plot with features ####################
                        feature1, feature2=feature_importances[0][0], feature_importances[1][0] # 2 most important features
                        
                        # features lead --> not used now. These are the feature values shifted forward in time, used to train the model. 
                        feature_lead=features_county[[feature1,feature2]] # feature values 
                        feature_lead=feature_lead.loc[feature_lead.index>=target_obs.index[0]] # show features only since 2009

                        # features plot ---> used 
                        features_plot= features_county.copy()
                        #features_plot= features_plot[features_plot['lead']==0]
                        features_plot=features_plot[[feature1,feature2]]
                        features_plot=features_plot.loc[features_plot.index>=target_obs.index[0]]

                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax2=ax.twinx()
                        
                        # Plot the actual values
                        ax.plot(target_obs.index,target_obs['FEWS_CS'], 'b-', label = 'Observed FEWS class')# Plot the predicted values
                        # plot rf predictions 
                        ax.plot(predictions_data.index, predictions_data['prediction'], 'ro', label = 'random Forest prediction')


                        # plot lead features
                        #ax2.plot(feature_lead.index, feature_lead[feature1], 'o-', color='orange', label = feature1+'_lead')# Plot the predicted values
                        #ax2.plot(feature_lead.index, feature_lead[feature2], 'o-', color='black', label = feature2+'+lead')# Plot the predicted values
                        
                        # plot observed features
                        ax2.plot(features_plot.index, features_plot[feature1], 'o-', color='purple', label = feature1+'_obs')# Plot the predicted values
                        ax2.plot(features_plot.index, features_plot[feature2], 'o-', color='blue', label = feature2+'_obs')# Plot the predicted values
                        
                        plt.xticks(rotation = 'vertical'); 
                        plt.xlabel('Date'); plt.ylabel('Magnitude of features'); plt.title('Explanatory_plot for %s,L=%s'%(c,lead))
                        # axis label on ax axes
                        ax.set_ylabel('FEWS IPC class')
                        plt.legend()
                        plt.savefig('Explanatory_plot_%s_L%s.png'%(c,lead), dpi=300, bbox_inches='tight')
                        
                        #plt.show() 
                        plt.close()
                    


        #################################################################################################################
        ################################################ SAVE RESULTS ####################################################
        #################################################################################################################
        os.chdir(RESULT_FOLDER)
        preds_storage.to_excel('raw_model_output_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region, cluster))
        features_df_full.to_excel('feature_importances_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region, cluster))
        print ('saved results in %s'%(RESULT_FOLDER))

        shap_values_master.to_excel('shap_values_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region, cluster))   
        shap_base_master.to_excel('shap_base_values_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region, cluster))
        shap_data_master.to_excel('shap_data_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region, cluster))
        print ('saved shap values in %s'%(RESULT_FOLDER))

        train_master.to_excel('train_data_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region, cluster))
       
        #################################################################################################################
        ################################################ CALCULATE SCORES ON ADMIN UNIT LEVEL############################
        #################################################################################################################

        for unit in preds_storage['county'].unique():
            ############## Load unit preds ##############
            preds_ini=preds_storage[preds_storage['county']==unit]

            ############## Load lead preds ##############
            for lead in preds_ini['lead'].unique():

                preds=preds_ini[preds_ini['lead']==lead]
                
                truth=preds['observed'] 
                base1=preds['base1_preds']
                base2=preds['base2_preds']
                #lr=preds['lr']
                preds=preds['prediction']

                
            
                ############################# Mean Absolute Error (MAE) #############################
                mae = mean_absolute_error(truth, preds)
                mae_baseline= mean_absolute_error(truth, base1)
                mae_baseline2= mean_absolute_error(truth, base2)
                #mae_lr= mean_absolute_error(truth, lr)

                ############## RMSE ##############
                rmse = mean_squared_error(truth, preds, squared = False)
                rmse_baseline= mean_squared_error(truth, base1, squared = False)
                rmse_baseline2= mean_squared_error(truth, base2, squared = False)
                #rmse_lr= mean_squared_error(truth, lr, squared = False)

                ############# R2 #############
                r2=r2_score(truth, preds)
                r2_baseline=r2_score(truth, base1)
                r2_baseline2=r2_score(truth, base2)
                #r2_lr=r2_score(truth, lr)


                
                county_eval= pd.DataFrame(data = {'county': unit,
                                                'lead': lead,
                                                'mae': mae,
                                                'mae_baseline': mae_baseline,
                                                'mae_baseline2': mae_baseline2,
                                                #'mae_lr': mae_lr,
                                                'rmse': rmse,
                                                'rmse_baseline': rmse_baseline,
                                                'rmse_baseline2': rmse_baseline2,
                                                #'rmse_lr': rmse_lr,
                                                'r2': r2,
                                                'r2_baseline': r2_baseline,
                                                'r2_baseline2': r2_baseline2,
                                                #'r2_lr': r2_lr
                                                }, index=[0])
                
                

                
                
                eval_stats=pd.concat([eval_stats, county_eval], axis=0) # preds_storage defined in beginning of script

        # save eval stats
        eval_stats.to_excel('verif_unit_level_%s_%s_%s_%s.xlsx'%(aggregation,experiment, region,cluster))

