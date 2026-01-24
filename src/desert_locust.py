# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022

@author: Tim Busker 

This script contains the code necessary for the calculation of the desert locust dataset. It uses data from the FAO desert locust database and the country/county shapefiles. Final result are the area of desert locust swarms per month per county, saved in the desert_locust_dataset.xlsx file.
"""

############################################################################################################################################################################
############################################################################## INSERT PACKAGES  ##################################################################
############################################################################################################################################################################
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


from shapely import wkt
from rasterio.enums import Resampling
from rasterio import rio
import geoplot as gplt

############################################################################################################################################################################
############################################################################## SET FOLDERS  ##################################################################
############################################################################################################################################################################

SE_FOLDER='PATH/ML/input_collector/SE_data'
GEO_FOLDER='PATH/ML/input_collector/Vector/geo_boundaries'
LOCUST_FOLDER='PATH/ML/input_collector/Vector/locust'


############################################################################################################################################################################
############################################################################## LOAD LOCUST DATA  ##################################################################
############################################################################################################################################################################


os.chdir(SE_FOLDER)
# load csv file for SWARMS 
FAO_Locust_Swarms= pd.read_csv('FAO_Locust_Swarms.csv', sep=',')
# load column STARTDATE as datetime index 
FAO_Locust_Swarms['STARTDATE']=pd.to_datetime(FAO_Locust_Swarms['STARTDATE'])
# change format of date to YYYY-MM-DD
FAO_Locust_Swarms['STARTDATE']=FAO_Locust_Swarms['STARTDATE'].dt.strftime('%Y-%m-%d')
FAO_Locust_Swarms=FAO_Locust_Swarms.set_index('STARTDATE')

# make extra column with point geometry
latitudes=FAO_Locust_Swarms['Y']
longitudes=FAO_Locust_Swarms['X']

# make a gdf 
gdf_locust = gpd.GeoDataFrame(FAO_Locust_Swarms, geometry=gpd.points_from_xy(longitudes, latitudes))
# keep only the columns 'geometry', 'LOCNAME', 'AREAHA', 'COUNTRY_ID'
gdf_locust=gdf_locust[['geometry', 'LOCNAME', 'AREAHA', 'COUNTRYID']]

############################################################################################################################################################################
############################################################################## LOAD COUNTRY/COUNTY VECTORS  ##################################################################
############################################################################################################################################################################

os.chdir(GEO_FOLDER)

# load shapefile with countries
ETH=gpd.read_file('Ethiopia/eth_admbnda_adm2_csa_bofedb_2021.shp')
ETH.rename(columns={'ADM2_EN':'unit'}, inplace=True)
SOM=gpd.read_file('Somalia/Som_Admbnda_Adm2_UNDP.shp')
SOM.rename(columns={'admin2Name':'unit'}, inplace=True)

KEN=gpd.read_file('Kenya/County.shp')
KEN.rename(columns={'COUNTY':'unit'}, inplace=True)
#merge to one gdf 
gdf_region=gpd.GeoDataFrame(pd.concat([ETH, SOM, KEN], ignore_index=True), crs=4326)
gdf_region.plot()
plt.show() 

gdf_region=gdf_region[['unit', 'geometry']]


########################################################################################################################################################
######################################################## KEEP ONLY RECORDS WITHIN THE THREE COUNTRIES###################################################
#######################################################################################################################################################

# This part is unchecked as it is already done -> results are saved in gdf_locust.shp

# for each gdf_locust row, check if the point is within the gdf_region geometry
# if yes, append the unit name to the gdf_locust geodataframe
# gdf_locust['unit']=''
# for i in range(len(gdf_locust)):
#     for j in range(len(gdf_region)):
#         if gdf_locust['geometry'][i].within(gdf_region['geometry'][j]):
#             gdf_locust['unit'][i]=gdf_region['unit'][j]
#             print('point', i, 'is within', j)
#         else:
#             print('point', i, 'is not within', j)


# save as shapefile with a reference system
# gdf_locust.crs=4326

# gdf_locust.to_file('gdf_locust.shp')

########################################################################################################################################################
######################################################## CALCULATE AREA / MONTH / COUNTY ##############################################################
#######################################################################################################################################################

os.chdir(LOCUST_FOLDER)
# read the shapefile again
gdf_locust=gpd.read_file('gdf_locust.shp')

# STARTDATE as datetime index
gdf_locust['STARTDATE']=pd.to_datetime(gdf_locust['STARTDATE'])
# set as index
gdf_locust=gdf_locust.set_index('STARTDATE')
gdf_locust.sort_index(inplace=True)

# drop rows with nan in unit column
gdf_locust=gdf_locust.dropna(subset=['unit'])

# drop rows with AREAHA=0
gdf_locust=gdf_locust[gdf_locust['AREAHA']!=0]

# sum area per unit per month
gdf_locust=gdf_locust.groupby(['unit', pd.Grouper(freq='MS')])['AREAHA'].sum().reset_index()
# set index again
gdf_locust=gdf_locust.set_index('STARTDATE')
# sort index
gdf_locust.sort_index(inplace=True)

os.chdir(SE_FOLDER)

########################################################################################################################################################
######################################################## SAVE RESULTS #################################################################################
#######################################################################################################################################################

gdf_locust.to_excel('desert_locust_dataset.xlsx')

