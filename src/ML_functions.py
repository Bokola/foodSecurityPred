# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022
@author: Tim Busker 
"""

import os
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import regionmask
from rasterio.enums import Resampling

# Setup logging for the helper functions
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# HELPER FUNCTIONS: PARAMETER I/O
# ---------------------------------------------------------------------

def save_best_params(params, path):
    """Saves a dictionary of parameters to a JSON file."""
    # Convert any numpy types to python types for JSON compatibility
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (np.float32, np.float64)):
            clean_params[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            clean_params[k] = int(v)
        else:
            clean_params[k] = v
            
    with open(path, 'w') as f:
        json.dump(clean_params, f, indent=4)
    print(f"--- Hyperparameters saved to {path} ---")

def load_best_params(path, default_params):
    """
    Loads parameters from JSON; returns defaults if file is missing.
    Includes an extra check to ensure base_score isn't a bracketed string.
    """
    if Path(path).exists():
        try:
            with open(path, 'r') as f:
                params = json.load(f)
                print(f"--- Loading Hyperparameters from {path} ---")
                
                # SANITY CHECK: If base_score is a string like "[5E-1]", clean it
                if "base_score" in params and isinstance(params["base_score"], str):
                    clean_val = params["base_score"].replace("[", "").replace("]", "").strip()
                    params["base_score"] = float(clean_val)
                    
                return params
        except Exception as e:
            print(f"--- Error loading JSON: {e}. Using defaults. ---")
            return default_params
            
    print(f"--- HP file not found at {path}. Using default parameters. ---")
    return default_params

# ---------------------------------------------------------------------
# GEOGRAPHICAL & RASTER FUNCTIONS
# ---------------------------------------------------------------------

def rasterize_shp(input_shp, input_raster, resolution, upscaling_fac): 
    rainfall = input_raster.copy()
    if resolution == 'as_input': 
        lon_raster = rainfall.longitude.values
        lat_raster = rainfall.latitude.values
        
    if resolution == 'upscaled': 
        upscale_factor = upscaling_fac.copy()
        new_width = int(rainfall.rio.width * upscale_factor)
        new_height = int(rainfall.rio.height * upscale_factor)
        rainfall_mask = rainfall.rio.write_crs(4326, inplace=True)
        xds_upsampled = rainfall.rio.reproject(
            rainfall_mask.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )
        lon_raster = xds_upsampled.x.values
        lat_raster = xds_upsampled.y.values
        
    sf = input_shp
    sf_raster = regionmask.mask_geopandas(sf, lon_raster, lat_raster)        
    return sf_raster

def wet_days(rainfall_input, threshold, period):
    dry_days = rainfall_input.where((rainfall_input['tp'] > threshold) | rainfall_input.isnull(), 1)
    dry_days = dry_days.where((dry_days['tp'] == 1) | dry_days.isnull(), 0)
    wet_days = rainfall_input.where((rainfall_input['tp'] > threshold) | rainfall_input.isnull(), 0)
    wet_days = wet_days.where((wet_days['tp'] == 0) | wet_days.isnull(), 1)
    wet_days_number = wet_days.resample(time=period).sum()
    return wet_days_number

def max_dry_spells(rainfall_input, threshold, period):
    dry_days = rainfall_input.where((rainfall_input['tp'] > threshold) | rainfall_input.isnull(), 1)
    dry_days = dry_days.where((dry_days['tp'] == 1) | dry_days.isnull(), 0)
    
    dry_spell = dry_days.where((((dry_days['time'].dt.month != 3) | (dry_days['time'].dt.day != 1)) & 
                                ((dry_days['time'].dt.month != 10) | (dry_days['time'].dt.day != 1))), 0)
    
    cumulative = dry_spell['tp'].cumsum(dim='time') - \
                 dry_spell['tp'].cumsum(dim='time').where(dry_spell['tp'].values == 0).ffill(dim='time').fillna(0)
    
    dry_spell_length = cumulative.where(cumulative >= 5, 0)
    dry_spell_max = dry_spell_length.resample(time=period).max()
    return dry_spell_max.to_dataset()

def P_mask(p_input, month, resample, p_thres):
    def month_selector(month_select):
        return (month_select == int(month + 1)) 
                
    mask = p_input.resample(time=resample).sum()
    mask = mask.sel(time=month_selector(mask['time.month']))
    mask = mask.mean(dim='time')
    mask = mask.where(mask.tp > p_thres, 0)
    mask = mask.where(mask.tp == 0, 1)
    return mask