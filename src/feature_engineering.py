
"""
Created on Tue Jan 10 13:48:32 2023

@author: Tim Busker

- input = data_master.xlsx --> result from input_collector.py
- This script performs a feature engineering on the input data 
- Also adds lead times & creates the persistance forecast variable (base_ini)
- Output = input_master.csv --> input for ML model 
- Excel file with all features and leads, for all admin units, is saved to 'input_master.xlsx'

"""
# import necessary packages

import pandas as pd
import numpy as np
import os


############################################################################################################################################################################
############################################################################## SET FOLDERS  ##################################################################
############################################################################################################################################################################

#input data 
BASE_FOLDER= 'PATH/ML/input_collector/' #'C:\\Users\\USER\\Documents\\ML\\input_collector\\'



############################################################################################################################################################################
############################################################################## SET DESIGN VARIABLES  ##################################################################
############################################################################################################################################################################

data_master= pd.read_excel(BASE_FOLDER+'data_master.xlsx', index_col=0)
units= data_master.county.unique().tolist()
print ('UNITS IN DATA MASTER: ', units)

leads=[0,1,2,3,4,8,12] # check:  12 lead creates quite some nan's in the training data 

#####################################################################################################################
################################################### Load data master ###################################################
#####################################################################################################################


input_master=pd.DataFrame()

for i,county in enumerate(units): 
    print (county)
    print(i)
    
    #####################################################################################################################
    ################################################### Start feature engineering ###################################################
    #####################################################################################################################

    ################################### create empty lead dataframe ###################################
    features_l=pd.DataFrame() 

    features= data_master[data_master['county']==county]
    
    ############################################# ROLLING MEANS  #############################################
    ########## NDVI ##########
    # make a list of column names to not use feature engineering on
    do_nothing_cols=['FEWS_CS','tp','SPI_1','SPI_3','SPI_6','SPI_12','SPI_24', 'SPEI_1','SPEI_3','SPEI_6','SPEI_12','SPEI_24', 'SSMI_1','SSMI_3','SSMI_6','SSMI_12','SSMI_24', 'lhz', 'county', 'country']
    features_e= features[features.columns[~features.columns.isin(do_nothing_cols)]]

    for feature in features_e:
        features[feature+'_4']=features[feature].rolling(window=4).mean().shift(1)
        features[feature+'_12']=features[feature].rolling(window=12).mean().shift(1)
        # nan filling
        features[feature+'_4']=features[feature+'_4'].fillna((features[feature+'_4'].mean()))
        features[feature+'_12']=features[feature+'_12'].fillna((features[feature+'_12'].mean()))

    features.dropna(axis=1, how='all', inplace=True)



    #################################################### Extract MAM and OND seasons ####################################################
    features['month']=features.index.month
    features['year']=features.index.year

    # add OND and MAM rainy season to features
    features['OND']=((features['month']==10) | (features['month']==11) | (features['month']==12))
    features['MAM']=((features['month']==3) | (features['month']==4) | (features['month']==5))





    #################################################### feature engineering for FEWS ####################################################

    # create base_ini column, which is the FEWS_CS shifted by 1 month. used for stationarity prediction. 
    features['base_ini']=features['FEWS_CS'].shift(1)
    features['base_ini']=features['base_ini'].fillna(method='ffill') # fill nan values with previous value. 
    
    # FEWS_CS lags, and fill nan values with mean of the column
    features['FEWS_CS_lag1']=features['base_ini'].copy()
    features['FEWS_CS_lag1'].fillna((features['base_ini'].mean()), inplace=True)

    features['FEWS_CS_lag4']=features['base_ini'].shift(3) 
    features['FEWS_CS_lag4'].fillna((features['base_ini'].mean()), inplace=True)

    features['FEWS_CS_lag8']=features['base_ini'].shift(7)
    features['FEWS_CS_lag8'].fillna((features['base_ini'].mean()), inplace=True)
    


    # create new variables from rolling mean of existing features, where the preceding 4 months are used to calculate the mean. Do not include the current month 
    features['FEWS_CS_12']=features['base_ini'].rolling(window=12).mean()
    features['FEWS_CS_12'].fillna((features['base_ini'].mean()), inplace=True)
    

    #####################################################################################################################
    ################################################### Create lead times  ###################################################
    #####################################################################################################################


    #################################################### Add lead times to dataframe ####################################################

    # Shift features by the lead (1,4,8,12) months. This trains and tests the model with features from the past, creating a lag to predict the future.
    for lead in leads:
        if lead==0:
            features_l=features.copy()


        else: 
            features_l=features.copy()
            # shift the features by the lead. Does NOT include FEWS_CS column
            shift_columns=features_l.columns[~features_l.columns.isin(['FEWS_CS'])]
            # shift only the columns that are not FEWS_CS 
            features_l[shift_columns]=features_l[shift_columns].shift(lead)
            
            # remove the last X rows based on the lead
            features_l=features_l.drop(features_l.index[:lead])
        
        # column management 
        #features_l=features_l.loc[:, ~features_l.columns.str.contains('^county')]
        #features_l=features_l.loc[:, ~features_l.columns.str.contains('^country')]

    
        features_l['lead']=lead
        #features_l['county']=county
        ############################# Create base forecast, to be used later #############################
        features_l.rename(columns={'base_ini': 'base_forecast'}, inplace=True)
        # add all leads individually to the input dataframe
        input_master=pd.concat([input_master,features_l], axis=0)




#########################################################################################################################
#################################################### END OF FEATURE ENGINEERING  ####################################################
#########################################################################################################################

print ('INPUT DATAFRAME CONSTRUCTION SUCCESFULLY COMPLETED')          





#########################################################################################################################
#################################################### NAN FILLING (over all units)  ####################################################
#########################################################################################################################

# Some (not many) admin units do not have crop or rangeland, which makes the crop and range NDVI columns empty. This part fills the empty columns with the mean of the NDVI columns over the whole dataframe. 

crop_cols=[col for col in input_master.columns if 'crop' in col]
input_master[crop_cols]=input_master[crop_cols].fillna(input_master.NDVI_anom_crop.mean()) # fill nans of crop columns with mean of NDVI_crop column

range_cols=[col for col in input_master.columns if 'range' in col]
input_master[range_cols]=input_master[range_cols].fillna(input_master.NDVI_anom_range.mean()) # fill nans of range columns with mean of NDVI_range column

# check which input_master columns have at least 1 NAN values 
nan_cols=input_master.columns[input_master.isna().any()].tolist()

# nan are FEWS_CS, bsae_forecast, lhz_

if len(nan_cols)>2:
    print('NAN WARNING: MORE THAN 2 COLUMNS HAVE NAN VALUES', nan_cols)
else:
    print ('ONLY NANS ARE FEWS_CS AND BASE FORECAST, ACCEPTED & READY FOR ML MODEL. However, check the feature names to see if the engineering is correct')

print ('features: ', input_master.columns.to_list())

#########################################################################################################################
#################################################### EXCEL WRITING  ####################################################
#########################################################################################################################

input_master.to_csv(BASE_FOLDER+'input_master.csv')
