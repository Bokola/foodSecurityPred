
"""
Created on Mon Jan 23 15:33:24 2023

@author: Tim Busker 

This script takes the FEWS outlooks for near term (ML1) and medium term (ML2) and rasterizes 
them to the CHIRPS rainfall grid. Afterwards, the FEWS outlooks are aggregated to county level,
 and lead times are calculated. The results are saved as an excel file (fews_lead_df.xlsx) in 
 the FEWS folder. This file is then used in the plots_paper.py script to evaluate the outlooks against FEWS observations. 
"""

#####################################################################################################################
################################################### IMPORT PACKAGES  ###################################################
#####################################################################################################################
import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
from datetime import datetime
from scipy import stats
import seaborn as sns
import rioxarray
from scipy import stats
import cartopy
import cartopy.crs as ccrs    
import warnings
import sys
#warnings.simplefilter(action='ignore', category=FutureWarning)
from shapely import wkt
from rasterio.enums import Resampling
from rasterio import rio
#from function_def import rasterize_shp
#from function_def import *
from ML_functions import *

############################################################################################################################################################################
############################################################################## SET DIRECTORIES  ###############################################################################
############################################################################################################################################################################


BASE_FOLDER= 'PATH/ML/input_collector' #'C:/Users/USERNAME/Documents/ML/input_collector'
VECTOR_FOLDER=BASE_FOLDER+'/Vector/'
RAINFALL_FOLDER= BASE_FOLDER+'/CLIM_data/Rainfall' #'C:/Users/USERNAME/Documents/Forecast_action_analysis/CHIRPS025/HAD'
FC_FSO_FOLDER=BASE_FOLDER+'/FC_data'+'/FEWS'+'/FEWS_OUTLOOKS'
POP_FOLDER=BASE_FOLDER+'/Population'

############################################################################################################################################################################
############################################################################## Design variables  ###############################################################################
############################################################################################################################################################################
population_weighted=True

    
############################################################################################################################################################################
############################################################################## Load CHIRPS ##################################################################
############################################################################################################################################################################

os.chdir(RAINFALL_FOLDER+'/CHIRPS005_HAD') # CHIRPS 0.05 degree resolution
print ('loading rainfall...')
rainfall=xr.open_dataset('chirps-v2.0.2015.07.31_sub_HAD.nc').load() # one CHIRPS file 
print ('Rainfall loaded.')


# Save the lat and lon values of the rainfall dataset for lature use in rasterization of FEWS maps
lon_mask=rainfall.longitude.values 
lat_mask=rainfall.latitude.values





############################################################################################################################################################################
############################################################################## Name changes and IDP file removal  ##################################################################
############################################################################################################################################################################

# os.chdir(FC_FSO_FOLDER)
# hoa_files=[x for x in os.listdir() if 'horn_of_africa' in x]

# # Rename files 
# for hoa_file in hoa_files: 
#     file_dir=FC_FSO_FOLDER+'/'+hoa_file
#     os.chdir(file_dir)
#     for i in os.listdir():
#         os.rename(i, i.replace('HoA', 'EA'))

# # Rename folders 
# os.chdir(FC_FSO_FOLDER)
# for hoa_file in hoa_files:
#     os.rename(hoa_file, hoa_file.replace('horn_of_africa', 'east_africa'))

# delete IDP files 
# os.chdir(FC_FSO_FOLDER)
# for i in os.listdir():
#     os.chdir(FC_FSO_FOLDER+'/'+i) # enter the folder
#     IDP_file_list=[x for x in os.listdir() if 'IDP' in x]

#     for IDP_file in IDP_file_list:
#         os.remove(IDP_file)
#         print ('removing IDP file....%s'%(IDP_file))


############################################################################################################################################################################
########################################################## FEWS RASTERIZATION (SAVE AS NETCDF FILE) ########################################################################
############################################################################################################################################################################

data_types=['ML1', 'ML2'] 
for data_type in data_types: # loop over ML1 and ML2. This creates two NETCDF files with ML1 and ML2 FEWS maps.
    print (data_type)
    fews_xr= xr.Dataset()
    os.chdir(FC_FSO_FOLDER)


    if 'fews_xr_%s.nc'%(data_type) not in os.listdir(): 
        print('EXECUTING FEWS RASTERIZATION.....')

        for i in os.listdir():

            if 'east_africa' in i:
                date= datetime.strptime(i[-6:], '%Y%m')
                print (date)

                # enter the folder 
                os.chdir(FC_FSO_FOLDER+'/'+i)
                file_list=[x for x in os.listdir() if data_type in x]

                if len(file_list)>0:
                    ipc_sf= gpd.read_file(os.getcwd()+'/EA_%s_%s.shp'%(i[-6:], data_type))

        

                    ################################# ML1 or ML2 #########################################
                    fews_cs=ipc_sf[[data_type, 'geometry']].set_index(data_type)# 'ML1', 'ML2'

                    fews_cs_raster = regionmask.mask_geopandas(fews_cs,lon_mask,lat_mask) # works! just ahve a geo_df with ML1/ML2 and geometry 
                    fews_cs_raster.coords['time'] = date 
                    fews_cs_raster = fews_cs_raster.expand_dims(dim='time')

                    # make a dataset
                    fews_cs_raster = xr.Dataset({data_type: fews_cs_raster})
                    fews_x=fews_cs_raster.copy() 

                    # add fews_x to fews_xr 
                    fews_xr=xr.merge([fews_xr, fews_x])
                    
                    # delete > 10 values --> some 88 values present in the FEWS maps.. 
                    fews_xr=fews_xr.where(fews_xr<10)

        # save as NETCDF file 
        fews_xr.to_netcdf(FC_FSO_FOLDER+'/fews_xr_%s.nc'%(data_type))



############################################################################################################################################################################
############################################################################## Population data ############################################################################
############################################################################################################################################################################

# load ML1 and Ml2 preds 
# ML1 
fews_xr_ML1= xr.open_dataset(FC_FSO_FOLDER+'/fews_xr_ML1.nc').load()
# ML2 
fews_xr_ML2= xr.open_dataset(FC_FSO_FOLDER+'/fews_xr_ML2.nc').load()

# make a new xarray dataset with ML1 and ML2
fews_xr=xr.merge([fews_xr_ML1, fews_xr_ML2])



############################################################################################################################################################################
############################################################################## Population data ############################################################################
############################################################################################################################################################################
if population_weighted==True:
    os.chdir(POP_FOLDER)

    pop_hoa= xr.open_dataset(POP_FOLDER+'/pop_hoa.tif').load() # population data
    pop_hoa=pop_hoa.rename({'x':'lon', 'y':'lat'}) # rename x and y to longitude and latitude
    pop_hoa=pop_hoa.drop('spatial_ref') # drop spatial ref

    pop_hoa=pop_hoa.squeeze('band')
    pop_hoa=pop_hoa.drop('band')
    pop_hoa=pop_hoa.band_data # select band_data
    pop_EA_int=pop_hoa.interp_like(fews_xr, method='nearest') # interpolate to the same grid as the fews_xr dataset



############################################################################################################################################################################
########################################################## FEWS RASTER TO COUNTY EXCEL DATA ################################################################################
############################################################################################################################################################################

countries=['Somalia', 'Ethiopia', 'Kenya']
input_dataframe=pd.DataFrame()


for country in countries: #'Somalia','Ethiopia', 'Kenya'

    if country=='Kenya':
        units_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Kenya/County.shp')
        units_sf=units_sf.set_index('OBJECTID')
        units_sf.rename(columns={'COUNTY':'county'}, inplace=True)  
        


    if country=='Somalia':
        units_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Somalia/Som_Admbnda_Adm2_UNDP.shp')
        units_sf.rename(columns={'OBJECTID_1':'OBJECTID'}, inplace=True)
        units_sf=units_sf.set_index('OBJECTID')
        units_sf.rename(columns={'admin2Name':'county'}, inplace=True)  

    if country=='Ethiopia':
        units_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Ethiopia/eth_admbnda_adm2_csa_bofedb_2021.shp')
        units_sf.rename(columns={'ADM2_EN':'county'}, inplace=True) 
        units_sf.reset_index(inplace=True)
        units_sf.rename(columns={'index':'OBJECTID'}, inplace=True)
        units_sf=units_sf.set_index('OBJECTID')



    units_raster= regionmask.mask_geopandas(units_sf,lon_mask,lat_mask)  # using upscaled rainfall mask 
    units_sf_ri=units_sf.reset_index() # units_sf with object_id as column
    units=units_sf['county'].values
    print ('units=', units)
    
    for county in units:
        county_df=pd.DataFrame()
        
        ##### extract unit ID from unit shapefile ---> use to extract raster data for specific county
        unit_ID= units_sf_ri[units_sf_ri['county']==county].OBJECTID.values[0]

        print ('county=', county)


        fews_county_ml1_df=pd.DataFrame()
        fews_county_ml2_df=pd.DataFrame()

        os.chdir(FC_FSO_FOLDER)

        for i in range(len(fews_xr.time.values)):
            
            fews_raster=fews_xr.isel(time=i)
            date=pd.Timestamp(fews_raster.time.values)
            fews_raster_county=fews_raster.where(units_raster==unit_ID, np.nan)
            

                
            if population_weighted==False:
                fews_county_ml1=pd.Series({'FEWS_ML1':float(fews_raster_county.ML1.mean(skipna=True))}, name=date)
                fews_county_ml2=pd.Series({'FEWS_ML2':float(fews_raster_county.ML2.mean(skipna=True))}, name=date)         

            # population weights 
            if population_weighted==True:
                # create weights
                pop_county=pop_EA_int.where(units_raster==unit_ID, drop=True)
                pop_county= pop_county.fillna(0) # nan values are not allowed in weights

                # apply weights 
                fews_weighted=fews_raster_county.weighted(pop_county)
                fews_county_mean=fews_weighted.mean(('lon', 'lat'))

                fews_county_ml1=pd.Series({'FEWS_ML1':float(fews_county_mean.ML1)}, name=date)
                fews_county_ml2=pd.Series({'FEWS_ML2':float(fews_county_mean.ML2)}, name=date)



            # build ML1 and ML2 dataframe
            fews_county_ml1_df=fews_county_ml1_df._append(fews_county_ml1)
            fews_county_ml2_df=fews_county_ml2_df._append(fews_county_ml2)



        


        county_df=pd.concat([county_df,fews_county_ml1_df], axis=1)
        county_df=pd.concat([county_df,fews_county_ml2_df], axis=1)
        county_df['county']= county
        county_df['country']= country

        print('dataframe build for %s in %s'%(county,country))

        ######################################################### Append to input_df #########################################################
        input_dataframe=pd.concat([input_dataframe, county_df], axis=0)



############################################################################################################################################################################
########################################################## ML1 and ML2 OUTLOOKS -----> LEAD TIMES ################################################################################
############################################################################################################################################################################

# In the time of the test dataset, the FEWS CS estimations are released in feb, june and october. The FEWS outlooks are generated and published every month (ML 1 and ML2). We need to make a dataframe with the predictions from ML1 and ML2 outlooks, for the months feb, june and october. 
# The script below implements that, and creates the corresponding lead time values.  

fews_lead_df=input_dataframe.copy()
############################################################### Calculate lead times from the dataframe ###############################################################
# ML 1 --> month - lead tuples 
ML1_tuples= [(1, 1), (2, 4), (3, 3), (4, 2), (5, 1), (6, 4), (7, 3), (8, 2), (9, 1), (10, 4), (11, 3), (12, 2)]
# ML 2 --> month - lead tuples 
ML2_tuples= [(1, 5), (2, 8), (3, 7), (4, 6), (5, 5), (6, 8), (7, 7), (8, 6), (9, 5), (10, 8), (11, 7), (12, 6)] 

# ML1 --> ini month - target month tuples
ML1_target_month_tuples= [(1, 2), (2, 6), (3, 6), (4, 6), (5, 6), (6, 10), (7, 10), (8, 10), (9, 10), (10, 2), (11, 2), (12, 2)]
# ML2 --> ini month - target month tuples
ML2_target_month_tuples=[(1, 6), (2, 10), (3, 10), (4, 10), (5, 10), (6, 2), (7, 2), (8, 2), (9, 2), (10, 6), (11, 6), (12, 6)] 

# Assign the lead time to the dataframe according to the month in the tuples list
fews_lead_df['ML1_lead']= fews_lead_df.index.month.map(dict(ML1_tuples))
fews_lead_df['ML2_lead']= fews_lead_df.index.month.map(dict(ML2_tuples))



# ML1 dataframe 
fews_lead_df_ML1=fews_lead_df[['FEWS_ML1', 'ML1_lead', 'country', 'county']]
fews_lead_df_ML1['ini_date']= pd.to_datetime(fews_lead_df_ML1.index, format="%Y-%m-%d")
fews_lead_df_ML1['target_date']= fews_lead_df_ML1.apply(lambda x: x['ini_date'] + pd.DateOffset(months=x['ML1_lead']), axis=1)
fews_lead_df_ML1=fews_lead_df_ML1.set_index('target_date')
fews_lead_df_ML1.drop(columns=['ini_date'], inplace=True)

# ML2 dataframe 
fews_lead_df_ML2=fews_lead_df[['FEWS_ML2', 'ML2_lead', 'country', 'county']]
fews_lead_df_ML2['ini_date']= pd.to_datetime(fews_lead_df_ML2.index, format="%Y-%m-%d")
fews_lead_df_ML2['target_date']= fews_lead_df_ML2.apply(lambda x: x['ini_date'] + pd.DateOffset(months=x['ML2_lead']), axis=1)
fews_lead_df_ML2=fews_lead_df_ML2.set_index('target_date')
fews_lead_df_ML2.drop(columns=['ini_date'], inplace=True)

# rename FEWS_ML1 and FEWS_ML2 to FEWS_prediction
fews_lead_df_ML1.rename(columns={'FEWS_ML1':'FEWS_prediction'}, inplace=True)
fews_lead_df_ML2.rename(columns={'FEWS_ML2':'FEWS_prediction'}, inplace=True)
# rename ML2_lead and ML1_lead to lead
fews_lead_df_ML1.rename(columns={'ML1_lead':'lead'}, inplace=True)
fews_lead_df_ML2.rename(columns={'ML2_lead':'lead'}, inplace=True)


# concat ML1 and ML2 dataframe along the rows
fews_lead_df=pd.concat([fews_lead_df_ML1, fews_lead_df_ML2], axis=0)
fews_lead_df.reset_index(inplace=True)
# sort based on multiple columns. first sort on county, within the county on target date, and within target date on lead 
fews_lead_df=fews_lead_df.sort_values(by=['county', 'target_date', 'lead'])

############################################################################################################################################################################
########################################################## SAVE RESULTS  ################################################################################
############################################################################################################################################################################

# save as excel 
if population_weighted==True:
    fews_lead_df.to_excel(FC_FSO_FOLDER +'/fews_lead_df_pop_weighted.xlsx')
    print ('FEWS prediction dataframe finished and saved as fews_lead_df_pop_weighted.xlsx')
else:
    fews_lead_df.to_excel(FC_FSO_FOLDER +'/fews_lead_df.xlsx')
    print ('FEWS prediction dataframe finished and saved as fews_lead_df.xlsx')



