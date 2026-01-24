# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022

@author: Tim Busker 

This script contains the code required to assign livelihood zones to the counties in the Horn of Africa. The final result is a shapefile with the livelihood zones per county/admin unit. 

"""


########################################################################################################################################################
######################################################## Packages ######################################################################################
#######################################################################################################################################################

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import xarray as xr
#os.chdir('C:\\Users\\USERNAME\\Documents\\Forecast_action_analysis')
from ML_functions import *
import cartopy.crs as ccrs
import cartopy.feature as cf


import geopandas as gpd
import regionmask
import geoplot as gplt
import geoplot.crs as gcrs

################## Make raster reference layer --> HRES 005 CHIRPS data #####################
os.chdir('PATH/ML/input_collector/CLIM_data/Rainfall/CHIRPS005')
rainfall= xr.open_dataset(os.listdir()[10]).load()
lon=rainfall.x.values
lat=rainfall.y.values

########################################################################################################################################################
######################################################## SET DIRECTORIES ##############################################################################
#######################################################################################################################################################

BASE_FOLDER= 'PATH/ML/input_collector' #'C:/Users/USERNAME/Documents/ML/input_collector'
VECTOR_FOLDER=BASE_FOLDER+'/Vector/'#'C:/Users/USERNAME/Documents/Forecast_action_analysis/vector_data/'



########################################################################################################################################################
######################################################## LOAD SHAPEFILES ###############################################################################
#######################################################################################################################################################

### livelihoods are pastoral, agro-pastoral and other. Other mostly comprises of crop areas, but is also used for urban (1 in Kenya), and East Golis – Frankincense, Goats, and Fishing (Som).

#############################  load livelihood shapefiles ########################################
os.chdir('PATH/ML/input_collector/Vector/livelyhood_zones')
lh_som=gpd.read_file('SO_LHZ_2015.shp')#.set_index("FNID")
lh_som=lh_som[['CLASS', 'LZNAMEEN', 'geometry']]
lh_eth= gpd.read_file('ET_LHZ_2018.shp')
# drop rows where column LZTYPE is empty
lh_eth=lh_eth.dropna(subset=['LZTYPE'])


lh_eth=lh_eth[['CLASS', 'LZTYPE', 'geometry']]
lh_ken= gpd.read_file('KE_LHZ_2011.shp')
lh_ken=lh_ken[['CLASS', 'LZNAMEEN', 'geometry']]


########################################################################################################################################################
######################################################## ASSIGN LHZ NAMES ###############################################################################
#######################################################################################################################################################


#################### Somalia
# make column 'LZTYPE'. Assign 'pastoral' if column LZNAMEEN contains 'pastoral'. Assign 'agro_pastoral' if column LZNAMEEN contains 'agropastoral'. Assign 'other' to the other rows 
lh_som.loc[lh_som['LZNAMEEN'].str.contains('Pastoral'), 'LZ_P'] = 'pastoral'
lh_som.loc[lh_som['LZNAMEEN'].str.contains('Agropastoral'), 'LZ_AP'] = 'agro_pastoral'
lh_som.loc[lh_som['LZNAMEEN'].str.contains('Pastoral|Agropastoral')==False, 'LZ_other'] = 'other'

lh_som['LZ_tim']=lh_som['LZ_P'].fillna(lh_som['LZ_AP']).fillna(lh_som['LZ_other']) # merge LZ_P, LZ_AP and LZ_other to one column

lh_som=lh_som.drop(columns=['LZ_P', 'LZ_AP', 'LZ_other'])# drop columns LZ_P, LZ_AP and LZ_other

#################### Ethiopia
lh_eth.loc[lh_eth['LZTYPE'].str.contains('Pastoral'), 'LZ_P'] = 'pastoral'    
lh_eth.loc[lh_eth['LZTYPE'].str.contains('Agropastoral'), 'LZ_AP'] = 'agro_pastoral'
lh_eth.loc[lh_eth['LZTYPE'].str.contains('Pastoral|Agropastoral')==False, 'LZ_other'] = 'other'

lh_eth['LZ_tim']=lh_eth['LZ_P'].fillna(lh_eth['LZ_AP']).fillna(lh_eth['LZ_other']) # merge LZ_P, LZ_AP and LZ_other to one column

lh_eth=lh_eth.drop(columns=['LZ_P', 'LZ_AP', 'LZ_other'])# drop columns LZ_P, LZ_AP and LZ_other
lh_eth=lh_eth.drop(columns=['LZTYPE'])

#################### Kenya
lh_ken.loc[lh_ken['LZNAMEEN'].str.contains('Pastoral'), 'LZ_P'] = 'pastoral'    
lh_ken.loc[lh_ken['LZNAMEEN'].str.contains('Agropastoral'), 'LZ_AP'] = 'agro_pastoral'
lh_ken.loc[lh_ken['LZNAMEEN'].str.contains('Pastoral|Agropastoral')==False, 'LZ_other'] = 'other'

lh_ken['LZ_tim']=lh_ken['LZ_P'].fillna(lh_ken['LZ_AP']).fillna(lh_ken['LZ_other']) # merge LZ_P, LZ_AP and LZ_other to one column

lh_ken=lh_ken.drop(columns=['LZ_P', 'LZ_AP', 'LZ_other'])# drop columns LZ_P, LZ_AP and LZ_other

################## merge all gdf to one ##################

lh_all= pd.concat([lh_som,lh_eth,lh_ken], axis=0) # merge all gdf to one


########################################################################################################################################################
######################################################## CREATE SEPERATE GDF'S FOR EACH LHZ#############################################################
#######################################################################################################################################################

lh_p= lh_all.where(lh_all['LZ_tim']=='pastoral').dropna(how='all') # make gdf with only pastoral areas
lh_ap= lh_all.where(lh_all['LZ_tim']=='agro_pastoral').dropna(how='all') # make gdf with only agropastoral areas
lh_other= lh_all.where(lh_all['LZ_tim']=='other').dropna(how='all') # make gdf with only other areas

###################### make raster LHZ's #####################

######## PASTORAL ########
lh_p_mask=regionmask.mask_geopandas(lh_p,lon,lat)
lh_p_mask=lh_p_mask.to_dataset()
lh_p_mask=lh_p_mask.where(lh_p_mask.mask>=0, -9999)
lh_p_mask=lh_p_mask.where(lh_p_mask.mask==-9999, 1)        
lh_p_mask=lh_p_mask.where(lh_p_mask.mask==1,0)  
lh_p_mask=lh_p_mask.rename({'lon': 'longitude','lat': 'latitude'})

######## AGRO-PASTORAL ########
lh_ap_mask=regionmask.mask_geopandas(lh_ap,lon,lat)
lh_ap_mask=lh_ap_mask.to_dataset()
lh_ap_mask=lh_ap_mask.where(lh_ap_mask.mask>=0, -9999)
lh_ap_mask=lh_ap_mask.where(lh_ap_mask.mask==-9999, 1)        
lh_ap_mask=lh_ap_mask.where(lh_ap_mask.mask==1,0)  
lh_ap_mask=lh_ap_mask.rename({'lon': 'longitude','lat': 'latitude'})

######## OTHER ########
lh_other_mask=regionmask.mask_geopandas(lh_other,lon,lat)
lh_other_mask=lh_other_mask.to_dataset()
lh_other_mask=lh_other_mask.where(lh_other_mask.mask>=0, -9999)
lh_other_mask=lh_other_mask.where(lh_other_mask.mask==-9999, 1)        
lh_other_mask=lh_other_mask.where(lh_other_mask.mask==1,0)  
lh_other_mask=lh_other_mask.rename({'lon': 'longitude','lat': 'latitude'})


########################################################################################################################################################
######################################################## AGGREGATE LHZ TO COUNTY-LEVEL#################################################################
#######################################################################################################################################################

cluster_df= pd.DataFrame()


# KEN
units_ken=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Kenya/County.shp')
units_ken=units_ken.set_index('OBJECTID')
units_ken.rename(columns={'COUNTY':'county'}, inplace=True)  
units_ken=units_ken[['county','geometry']]
# SOM 
units_som=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Somalia/Som_Admbnda_Adm2_UNDP.shp')
units_som.rename(columns={'OBJECTID_1':'OBJECTID'}, inplace=True)
units_som=units_som.set_index('OBJECTID')
units_som.rename(columns={'admin2Name':'county'}, inplace=True)  
units_som=units_som[['county','geometry']]
# ETH 
units_eth=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Ethiopia/eth_admbnda_adm2_csa_bofedb_2021.shp')
units_eth.rename(columns={'ADM2_EN':'county'}, inplace=True) 
units_eth.reset_index(inplace=True)
units_eth.rename(columns={'index':'OBJECTID'}, inplace=True)
units_eth=units_eth.set_index('OBJECTID')
units_eth=units_eth[['county','geometry']]

units_all= pd.concat([units_ken,units_som,units_eth], axis=0) # merge all gdf to one
units_all=units_all.reset_index()
units_all=units_all.reindex(np.arange(0,len(units_all))) # reindex
units_all=units_all.drop(columns=['OBJECTID']) # drop column 'index'

units_all.to_file(VECTOR_FOLDER+'geo_boundaries/HOA/HOA.shp') # save as shapefile

raster=regionmask.mask_geopandas(units_all,lon,lat)
raster=raster.rename({'lon': 'longitude','lat': 'latitude'})


for units in units_all['county']:
    print (units)
    ID=units_all.where(units_all['county']==units).dropna(how='all').index.values[0]
    lh_p_county=lh_p_mask.where(raster==ID)
    lh_ap_county=lh_ap_mask.where(raster==ID)
    lh_other_county=lh_other_mask.where(raster==ID)
    
    p=float(lh_p_county.mean(dim=('latitude', 'longitude')).mask.values)
    ap=float(lh_ap_county.mean(dim=('latitude', 'longitude')).mask.values)
    other=float(lh_other_county.mean(dim=('latitude', 'longitude')).mask.values)
    cluster_df.loc[units,'p']=p
    cluster_df.loc[units,'ap']=ap
    cluster_df.loc[units,'other']=other



# make new column which represents the column with the highest value
cluster_df['max']=cluster_df.idxmax(axis=1)


# save as excel 
cluster_df.to_excel(BASE_FOLDER+'/livelihood_zones.xlsx')
lhz_tim=cluster_df[['max']]



########################################################################################################################################################
######################################################## SAVE TO SHAPEFILE#############################################################################
#######################################################################################################################################################
os.chdir('PATH/ML/input_collector/Vector/livelyhood_zones')

units_all=units_all.set_index('county')
# merge with lhz_tim on index 
lhz_gdf= pd.merge(units_all, lhz_tim, left_index=True, right_index=True)
lhz_gdf.to_file('lhz_busker.shp') # save as shapefile


