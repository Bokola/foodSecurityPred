# -*- coding: utf-8 -*-
"""10
Created on Fri Jan  7 11:33:21 2022

@author: Tim Busker 

This script contains all functions used in the ML analysis. 
"""

########################################################################################################################################################
######################################################## Packages ######################################################################################
#######################################################################################################################################################

import os
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
import geopandas as gpd
import regionmask
import xarray as xr
import urllib.request
import requests
import re
import glob
#from bs4 import BeautifulSoup
import datetime as dtmod
import re
from rasterio.enums import Resampling
import scipy.stats as stats
#path = 'C:/Users/USER/Documents/Forecast_action_analysis'

#os.chdir(path)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

########################################################################################################################################################
################################################### FUNCTION TO RASTERIZE SHAPEFILES ##################################################################
#######################################################################################################################################################

# Add these two helper functions to your ML_functions.py. 
# This standardizes how your scripts read and write settings.

import json
from pathlib import Path

def save_best_params(params, path):
    """Saves a dictionary of parameters to a JSON file."""
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"--- Hyperparameters saved to {path} ---")

def load_best_params(path, default_params):
    """Loads parameters from JSON; returns defaults if file is missing."""
    if Path(path).exists():
        with open(path, 'r') as f:
            print(f"--- Loading Hyperparameters from {path} ---")
            return json.load(f)
    print("--- HP file not found. Using default parameters. ---")
    return default_params
###########################################################################################

def rasterize_shp(input_shp, input_raster,resolution, upscaling_fac): # files should include file names and dir, res can be either 'as_input' or 'upscaled'. input raster can be of any resolution. 
    
    rainfall= input_raster.copy()#chirps-v2_ALL_YEARS_sub_HAD_NEW
    if resolution=='as_input': 
        #%% load CHIRPS and extract lon lats 
        
        lon_raster=rainfall.longitude.values
        lat_raster=rainfall.latitude.values
        
    if resolution=='upscaled': 
        #%% create mask 
        upscale_factor = upscaling_fac.copy()
        new_width = rainfall.rio.width * upscale_factor
        new_height = rainfall.rio.height * upscale_factor
        rainfall_mask=rainfall.rio.write_crs(4326, inplace=True)
        xds_upsampled = rainfall.rio.reproject(
            rainfall_mask.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )

        lon_raster=xds_upsampled.x.values
        lat_raster=xds_upsampled.y.values
        
    
           
    #%% make county raster with ID's 
    sf = input_shp
    
    sf_raster= regionmask.mask_geopandas(sf,lon_raster,lat_raster)        
    
    return (sf_raster)
    

########################################################################################################################################################
################################################### FUNCTION TO CALCULATE NUMBER OF RAINY DAYS #########################################################
#######################################################################################################################################################
def wet_days(rainfall_input, threshold, period):
    #%%%%% flag dry and wet days 
    dry_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 1) # isnull causes nan values to persist 
    dry_days=dry_days.where((dry_days['tp'] ==1) | dry_days.isnull(), 0)
    wet_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 0)
    wet_days=wet_days.where((wet_days['tp'] ==0) | wet_days.isnull(), 1)
    #%%%%% monthly sums 
    wet_days_number=wet_days.resample(time=period).sum()
    return(wet_days_number)
    
########################################################################################################################################################
################################################### FUNCTION TO CALCULATE DRY SPELL INDICATORS #########################################################
########################################################################################################################################################
def max_dry_spells(rainfall_input, threshold, period):
    #%%% dry days 
    dry_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 1) # isnull causes nan values to persist 
    dry_days=dry_days.where((dry_days['tp'] ==1) | dry_days.isnull(), 0)
    
    #%%% dry spells 
    ###### put beginning of season to 0! Leave all the other values representing the dry spell length --> creates inaccuracy of dry spell length with 0 day for dry spells that start at first day of the season. This is important as otherwise the fake dry days of the non-rainy season months (np.nan values) will create a very long dry spell in the rainy season. 
    dry_spell=dry_days.where((((dry_days['time'].dt.month !=3) | (dry_days['time'].dt.day !=1)) & ((dry_days['time'].dt.month !=10) | (dry_days['time'].dt.day !=1))), 0)
    ###### make a cumsum using only the dry days ##########
    # This function restarts cumsum after every 0, so at every wet day or beginning of the season. 
    cumulative = dry_spell['tp'].cumsum(dim='time')-dry_spell['tp'].cumsum(dim='time').where(dry_spell['tp'].values == 0).ffill(dim='time').fillna(0)
    dry_spell_length= cumulative.where(cumulative>=5, 0) ## only keep length of dry spells when >=5 days. Rest of the values to zero

    
    ####################################################################################### number of dry spells per season  ################################################################################
    # dry_spell_start=dry_spell_length.where(dry_spell_length==5, 0)
    # dry_spell_start=dry_spell_start.where(dry_spell_start!=5, 1) ## binary dry spell start  
    
    # dry_spell_number=dry_spell_start.resample(time='MS').sum() ## sum all dry spell starts 
    # dry_spell_number = dry_spell_number.where((((dry_spell_number['time'].dt.month >=3) & (dry_spell_number['time'].dt.month <=5)) | ((dry_spell_number['time'].dt.month >=10) & (dry_spell_number['time'].dt.month <=12))), np.nan)#this makes dry season months nan's
    
    # dry_spell_number=dry_spell_number.where(land_mask==1, np.nan) ## land mask
    # dry_spell_number=dry_spell_number.to_dataset()

    ####################################################################################### maximum (seasonal!!) dry spell length ################################################################################
    dry_spell_max= dry_spell_length.resample(time=period).max() ## max dry spell length per month, per season
    dry_spell_max=dry_spell_max.to_dataset() ## for processing later on 
    
    return (dry_spell_max)


########################################################################################################################################################
################################################### FUNCTION TO CALCULATE A PRECIPITATION MASK #########################################################
########################################################################################################################################################


#%%% Mask areas 
def P_mask(p_input,month, resample, p_thres):
    
    ## vars needed to sel months 
    months= (range(1,13))
    months_z= []
    for i in months: 
        j=f"{i:02d}"
        months_z.append(j)  
        
    def month_selector(month_select):
        return (month_select == month_of_interest)  ## can also be a range
                
    mask= p_input.resample(time=resample).sum()
    month_of_interest= int(months_z[month]) ## oct ## convert to int to select month, doesnt work with string
    mask = mask.sel(time=month_selector(mask['time.month']))
    mask=mask.mean(dim='time')
    mask= mask.where(mask.tp>p_thres, 0)
    mask=mask.where(mask.tp ==0, 1)
    
    return mask
   