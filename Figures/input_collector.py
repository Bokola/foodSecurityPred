# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:48:32 2023

@author: Tim Busker

- This script collects all the input data for the ML model on a specific administrative unit scale. 
- The outcome is one large excel file with all variables on monthly timesteps, with the counties stacked on top of each other 
- NAN values have been filled with a county-level mean. 
- Output excel file is saved to data_master.xlsx

- This file feeds into feature_engineering.py to start the feature engineering process, before the data is ready for the ML model.

"""



############################################################################################################################################################################
############################################################################## INSERT PACKAGES  ##################################################################
############################################################################################################################################################################
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
from datetime import datetime
from scipy import stats
import seaborn as sns
from scipy import stats
import sys
from rasterio.enums import Resampling
from rasterio import rio

# import functions from the ML_functions.py script
from ML_functions import *

############################################################################################################################################################################
############################################################################## SET DESIGN VARIABLES  ##################################################################
############################################################################################################################################################################
countries= ['Kenya','Somalia','Ethiopia'] 
input_dataframe= pd.DataFrame() 
indicator= 'Pewi' # or price --> this is the WFP price indicator used in the model 
population_weighted=True # if True, the WorldPOP population raster is used as weights to calculate mean IPC per county. 

############################################################################################################################################################################
############################################################################## SET FOLDERS  ##################################################################
############################################################################################################################################################################

#input data 
BASE_FOLDER= 'PATH/ML/input_collector' #'C:/Users/USER/Documents/ML/input_collector'
# Vectors
VECTOR_FOLDER=BASE_FOLDER+'/Vector/'#'C:/Users/USER/Documents/Forecast_action_analysis/vector_data/'
# Raster
RAINFALL_FOLDER= BASE_FOLDER+'/CLIM_data/Rainfall' #'C:/Users/USER/Documents/Forecast_action_analysis/CHIRPS025/HAD'
CHIRPS_HAD_FOLDER= RAINFALL_FOLDER+'/CHIRPS005_HAD' # CHIRPS 0.05 degree resolution 
NDVI_FOLDER=BASE_FOLDER+ '/CLIM_data/NDVI/NDVI_NOA_STAR_HAD_1981_2022' #C:/Users/USER/Documents/Forecast_action_analysis/impacts/NDVI/NDVI_NOA_STAR_HAD_1981_2022'
POP_FOLDER=BASE_FOLDER+'/Population'

# other folders 
SE_FOLDER= BASE_FOLDER+'/SE_data/' # social economic data
SST_FOLDER=BASE_FOLDER+'/CLIM_data/'+'/SST/' # sea surface temperature data
INDICES_FOLDER=BASE_FOLDER+'/CLIM_data/Indices/' # climate indices data
FC_FOLDER=BASE_FOLDER+'/FC_data/' # FEWSNET data


############################################################################################################################################################################
############################################################################## Load CHIRPS ###################################################################################
############################################################################################################################################################################

os.chdir(CHIRPS_HAD_FOLDER) 
print ('loading rainfall...')
rainfall=xr.open_dataset('chirps-v2_005_ALL.nc').load()
rainfall=rainfall.sortby('time')
print ('Rainfall loaded.')


######## Save lon lat values from rainfall to rasterize vectors later on ########
lon_mask=rainfall.longitude.values
lat_mask=rainfall.latitude.values


############################################################################################################################################################################
############################################################# RAINFALL INDICATORS ############################################################################################
############################################################################################################################################################################




######################################################## Process RAINFALL ####################################################        
P_HAD=rainfall.copy()

if 'spatial_ref' in P_HAD.coords:
    print ('spatial ref exists; will create 0 rainfall in SPI and therefore wrong spi ')
    exit()

######################################################## LAND MASK #################################################### 
# land mask
P_HAD=P_HAD.assign(total_rainfall=(P_HAD.tp.sum(dim=['time']))) ## only to mask areas that never get rainfall 
land_mask=P_HAD['total_rainfall'].where(P_HAD['total_rainfall'] ==0, 1)
P_HAD=P_HAD.where(land_mask==1, np.nan).drop('total_rainfall')

######################################################## WET DAYS (function from the ML_functions.py script )  #################################################### 
wd=wet_days(P_HAD, 1,'MS')
wd=wd.where(land_mask==1, np.nan) ## land mask again
wd=wd.rename(tp='wd')

######################################################## DRY SPELLS (function from the ML_functions.py script )  #################################################### 
ds=max_dry_spells(P_HAD, 1,'MS')
ds=ds.where(land_mask==1, np.nan) ## land mask again
ds=ds.rename(tp='ds') # rename tp to ds

####################################################### RESAMPLE TO MONTHLY ############################################
P_monthly=P_HAD.resample(time='MS').sum() 


############################################################################################################################################################################
############################################################# NDVI ##########################################################
############################################################################################################################################################################

os.chdir(NDVI_FOLDER)
print ('loading NDVI...')
NDVI=xr.open_dataset('NDVI_NOA_STAR_1981_2022.nc').load() 
NDVI_range=xr.open_dataset('NDVI_rangemask.nc').load().squeeze('band').drop('band').drop('spatial_ref') # Crop and rangeland masks as described in the paper 
NDVI_crop= xr.open_dataset('NDVI_cropmask.nc').load().squeeze('band').drop('band').drop('spatial_ref') 
NDVI=NDVI.where(NDVI.time.dt.year>=2000).dropna(how='all', dim='time') # Keep only 2000-2020
NDVI_crop=NDVI_crop.where(NDVI_crop.time.dt.year>=2000).dropna(how='all', dim='time') 
NDVI_range=NDVI_range.where(NDVI_range.time.dt.year>=2000).dropna(how='all', dim='time') 


########################################### NDVI ANOMALIES ##########################################################

def xr_anomaly(xr_dataset):
    climatology_mean = xr_dataset.groupby("time.month").mean("time")
    climatology_std = xr_dataset.groupby("time.month").std("time")

    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        xr_dataset.groupby("time.month"),
        climatology_mean,
        climatology_std,
    )
    return stand_anomalies

# stand anomalies for NDVI, NDVI_range, NDVI_crop
NDVI_anom=xr_anomaly(NDVI)
NDVI_anom=NDVI_anom.drop('month')
NDVI_anom_range=xr_anomaly(NDVI_range)
NDVI_anom_range=NDVI_anom_range.drop('month')
NDVI_anom_crop=xr_anomaly(NDVI_crop)
NDVI_anom_crop=NDVI_anom_crop.drop('month')

############################################################################################################################################################################
############################################################################## FEWS RASTERIZATION  ##################################################################
############################################################################################################################################################################
fews_xr= xr.Dataset()

os.chdir(FC_FOLDER+'FEWS//FEWS_OBS//')

if 'fews_xr.nc' not in os.listdir(FC_FOLDER): 
    print('EXECUTING FEWS RASTERIZATION.....')

    for i in os.listdir(): # loop over all FEWSNET folders
        date= datetime.strptime(i[-6:], '%Y%m') # extract date from folder name
        print (date) 

        ################################## convert ipc shapefile to raster #########################################
        # captures the CS and HA0 columns 

        ipc_sf = gpd.read_file(os.getcwd()+'/east-africa%s/EA_%s_CS.shp'%(i[-6:], i[-6:]))#.set_index("OBJECTID")

        if 'HA0' in ipc_sf.columns:
            with_HA=True
            print ('HA INCLUDED!')

        else:
            with_HA=False
        
        ################################# CS #########################################
        # extract current situation (CS) column and geometry column
        fews_cs=ipc_sf[['CS', 'geometry']].set_index('CS')# 'HA0', 'geometry', 'ADMIN1'
        fews_cs_raster = regionmask.mask_geopandas(fews_cs,lon_mask,lat_mask)
        fews_cs_raster.coords['time'] = date
        fews_cs_raster = fews_cs_raster.expand_dims(dim='time')

        # make a dataset
        fews_cs_raster = xr.Dataset({'CS': fews_cs_raster})
        fews_x=fews_cs_raster.copy() 

        ################################# HA #########################################
        # extract humanitaria assistance (HA) column and geometry column

        if with_HA==True:
            fews_ha=ipc_sf[['HA0', 'geometry']].set_index('HA0')# 'HA0', 'geometry', 'ADMIN1'
            fews_ha_raster = regionmask.mask_geopandas(fews_ha,lon_mask,lat_mask) 
            # attach time as extra dimension 
            fews_ha_raster.coords['time'] = date
            fews_ha_raster = fews_ha_raster.expand_dims(dim='time')
            fews_ha_raster = xr.Dataset({'HA': fews_ha_raster})

            # merge fews_cs_raster and fews_ha_raster dataset to make one xarray dataset
            fews_x=xr.merge([fews_cs_raster, fews_ha_raster])

        
        # add fews_x to fews_xr 
        fews_xr=xr.merge([fews_xr, fews_x])
        
        # delete > 10 values (FEWS maps sometimes have very high values, but just sporadically)
        fews_xr=fews_xr.where(fews_xr<10)

    # save as NETCDF file 
    fews_xr.to_netcdf(FC_FOLDER+'fews_xr.nc')

fews_xr= xr.open_dataset(FC_FOLDER+'fews_xr.nc')



############################################################# POPULATION ##########################################################
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
############################################################################## START LOOP OVER COUNTRIES ##################################################################
############################################################################################################################################################################


for country in countries: #'Somalia','Ethiopia', 'Kenya'

    ############################################################################################################################################################################
    ############################################################################## LOAD VECTORS  ##################################################################
    ############################################################################################################################################################################
    

    ####################################################### SHAPEFILE NAME CHANGES ####################################################
    # Necessary because the shapefile names are not consistent with the names used in the other datasets (ACLED, etc)

    # KENYA 
    # shapefile.loc[shapefile['COUNTY']=="Keiyo-Marakwet", 'COUNTY']='Elgeyo Marakwet'
    # shapefile.loc[shapefile['COUNTY']=="Murang'a", 'COUNTY']='Muranga'
    # shapefile.loc[shapefile['COUNTY']=='Tharaka', 'COUNTY']='Tharaka-Nithi'
    

    # ETHIOPIA
    # Rename Wolayita to  Welayta
	# Eastern, Western, Southern, Central, North Western, South Eastern, Mekelle to Eastern Tigray etc… 
	# / to - 
	# Kelem Wellega to Kellem Wollega
    # Assosa to Asosa

    # SOMALIA --> no name changes needed

    ################################################### load Kenya's units ##########################################################
    if country=='Kenya':
        units_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Kenya/County.shp')
        units_sf=units_sf.set_index('OBJECTID')
        units_sf.rename(columns={'COUNTY':'county'}, inplace=True)  
        
        
        admin_level= 'admin1' # admin1 or admin2
        goods=['Maize (white)', 'Fuel (diesel)'] # Sorghum tested. Not gave any value    

    if country=='Somalia':
        units_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Somalia/Som_Admbnda_Adm2_UNDP.shp')
        units_sf.rename(columns={'OBJECTID_1':'OBJECTID'}, inplace=True)
        units_sf=units_sf.set_index('OBJECTID')
        units_sf.rename(columns={'admin2Name':'county'}, inplace=True)  

        admin_level= 'admin2'
        goods=['Maize (white)', 'Fuel (diesel)'] # Sorghum tested but not yield any value  

    if country=='Ethiopia':
        units_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/Ethiopia/eth_admbnda_adm2_csa_bofedb_2021.shp')
        units_sf.rename(columns={'ADM2_EN':'county'}, inplace=True) 
        units_sf.reset_index(inplace=True)
        units_sf.rename(columns={'index':'OBJECTID'}, inplace=True)
        units_sf=units_sf.set_index('OBJECTID')

        admin_level= 'admin2'
        goods=['Maize (white)', 'Fuel (diesel)'] # Sorghum tested but not yield any value

    units_raster= regionmask.mask_geopandas(units_sf,lon_mask,lat_mask)  # using rainfall lat lon values saved in the beginning of the script 

    units_sf_ri=units_sf.reset_index() # units_sf with object_id as column
    
    ############################################################################################################################################################################
    ############################################################################## DEFINE ADMIN UNIT NAMES ##################################################################
    ############################################################################################################################################################################
    units=units_sf['county'].values
    print ('units=', units)
    
    
    ############################################################################################################################################################################
    ############################################################################## CREATE RASTER REGIONS   ##################################################################
    ############################################################################################################################################################################
    
    ######## RAINFALL MASK (ORIGINAL) #######
    longitude = P_monthly.longitude.values
    latitude = P_monthly.latitude.values
    RAINFALL_mask = regionmask.mask_geopandas(units_sf,longitude,latitude) # Clips the raster to the shapefile of the administrative unit 
    RAINFALL_mask=RAINFALL_mask.rename({'lon': 'longitude','lat': 'latitude'}) 

    ###### NDVI MASK #######
    lon_NDVI = NDVI.longitude.values
    lat_NDVI = NDVI.latitude.values
    NDVI_mask = regionmask.mask_geopandas(units_sf,lon_NDVI,lat_NDVI) # Clips the raster to the shapefile of the administrative unit
    NDVI_mask=NDVI_mask.rename({'lon': 'longitude','lat': 'latitude'})


    ############################################################################################################################################################################
    ############################################################################# SST ##########################################################################################
    ############################################################################################################################################################################  

    ########################## load SST index data ##########################
    # WVG processed sheet 
    WVG=pd.read_excel(SST_FOLDER+'SST_index.xlsx', sheet_name='WVG_processed', index_col=0, header=0)
    # shift WVG one row forward. THe WVG is MAM average, so it cannot be assumed that the WVG MAM average can already be known in March. We now assume it can be reasonably known in April.
    WVG=WVG.shift(1)
    # For WVG, ffill the missing values 
    WVG.fillna(method='ffill', inplace=True)

        
    # WPG processed sheet
    WPG=pd.read_excel(SST_FOLDER+'SST_index.xlsx', sheet_name='WPG_processed', index_col=0, header=0)
    # shift WPG one row forward 
    WPG=WPG.shift(1)
    # For WPG, ffill the missing values
    WPG.fillna(method='ffill', inplace=True)

    # MEI sheet 
    MEI=pd.read_excel(SST_FOLDER+'SST_index.xlsx', sheet_name='MEI', index_col=0, header=0)
    # remove rows with values <-10
    MEI=MEI[MEI['MEI']>-10]
    # NINA34 sheet
    NINA34=pd.read_excel(SST_FOLDER+'SST_index.xlsx', sheet_name='NINA34', index_col=0, header=0)
    # remove rows with values <-10
    NINA34=NINA34[NINA34['NINA34']>-10]
    # IOD sheet
    IOD=pd.read_excel(SST_FOLDER+'SST_index.xlsx', sheet_name='IOD', index_col=0, header=0)
    # convert index to datetime
    IOD.index=pd.to_datetime(IOD.index)
    

    ############################################################################################################################################################################
    ############################################################################### Conflicts ###################################################################################
    ############################################################################################################################################################################ 
    # https://acleddata.com/data/acled-api/
    #for codebook, see https://acleddata.com/acleddatanew/wp-content/uploads/2021/11/ACLED_Codebook_v1_January-2021.pdf

    
    os.chdir(SE_FOLDER)
    acled=pd.read_csv([i for i in os.listdir() if 'acled' in i][0], index_col=0) # acled data


    ################ ACLED NAME CONVERSIONS ################
    # Needed because the acled names are not consistent with the names used in the other datasets (shapefiles, etc)

    # ETHIOPIA
    # duplicate rows named 'North Shewa', but change name to 'North Shewa (AM)' 
    acled.loc[acled['admin2']=='North Shewa', 'admin2']='North Shewa (AM)'
    # rename Kemashi to Kamashi 
    acled.loc[acled['admin2']=='Kemashi', 'admin2']='Kamashi'
    # rename 'Kembata Tibaro' to Kembata Tembaro
    acled.loc[acled['admin2']=='Kembata Tibaro', 'admin2']='Kembata Tembaro'

    # Somalia and Kenya don't need name conversions 
    
    # Detect missing admin names, which are not in ACLED but are in the unit shapefile 
    map_units=units_sf.county.unique()
    acled_units=acled[acled['country']==country]
    acled_units=acled_units[admin_level].unique()
    units_not_in_acled=set(map_units)-set(acled_units)

    # check if any acled units are missing in the map units --> this should be due to name changes
    acled_units_not_in_map=set(acled_units)-set(map_units)
    if len(acled_units_not_in_map)>0:
        print ('some acled units are missing in the map units')
        print (acled_units_not_in_map+'sys will exit')
        exit()
    

    ################ ACLED PRE-PROCESSING ################
    # note that for KEN admin1 level is used. 

    # extract event date, admin1, and fatalities

    acled=acled[['event_date',admin_level,'fatalities']] # 'sub_event_type'
    acled['event_date']=pd.to_datetime(acled['event_date']) 
    acled.set_index('event_date',inplace=True)
    
    

    
    #remove Remote explosive/landmine/IED and Suicide bomb from sub_event_type
    #values_to_remove = ['Remote explosive/landmine/IED', 'Suicide bomb']
    #acled = acled[~acled['sub_event_type'].isin(values_to_remove)]
    
    # only keep Armed clash sub_event_types
    #acled=acled[acled['sub_event_type']=='Armed clash']

    # remove column sub_event_type
    #acled.drop(columns=['sub_event_type'], inplace=True)
    

    acled.rename(columns={admin_level:'county','fatalities':'acled_fatalities'}, inplace=True)
    acled.sort_index(inplace=True)

    ############################################################################################################################################################################
    ##################################################################################### Desert locust #######################################################################
    ############################################################################################################################################################################ 
    # https://locust-hub-hqfao.hub.arcgis.com/
    os.chdir(SE_FOLDER)
    locust=pd.read_excel('desert_locust_dataset.xlsx', index_col=0) # locust data


    ######################################################## CROP PRODUCTION  ####################################################
    # lizumi data. Not used now, but can be used in future --> https://doi.pangaea.de/10.1594/PANGAEA.909132


    ############################################################################################################################################################################
    ###################################################################################### INFLATION ###########################################################################
    ############################################################################################################################################################################

    os.chdir(SE_FOLDER)


    inflation_h_df=pd.DataFrame() # headline inflation 
    inflation_f_df=pd.DataFrame() # food inflation


    if country=='Ethiopia' or country=='Kenya':
        
        ######################################## headline inflation ###################################
        inflation_h=inflation=pd.read_excel([i for i in os.listdir() if 'Inflation_WB' in i][0],header=0, sheet_name='hcpi_m') # world bank data 
        inflation_h=inflation_h[inflation_h['Country']==country]
        # Drop first 5 columns
        inflation_h=inflation_h.iloc[:,5:]
        # Drop last 4 columns
        inflation_h=inflation_h.iloc[:,:-4]
        # convert the column name to index rows 
        inflation_h=inflation_h.T
        # convert the index (%Y%m) to datetime
        inflation_h.index=pd.to_datetime(inflation_h.index, format='%Y%m')
        # rename column to ETH
        inflation_h.rename(columns={inflation_h.columns[0]:'CPI_H_'+country}, inplace=True)

        #inflation_h = (inflation_h - inflation_h.min()) / (inflation_h.max() - inflation_h.min()) # normalization

        ######################################## food inflation ###################################
        inflation_f=pd.read_excel([i for i in os.listdir() if 'Inflation_WB' in i][0],header=0, sheet_name='fcpi_m') # world bank data
        inflation_f= inflation_f[inflation_f['Country']==country] # select country
        inflation_f=inflation_f.iloc[:,5:]# Drop first 5 columns
        inflation_f=inflation_f.iloc[:,:-5]# Drop last 5 columns
        inflation_f=inflation_f.T # convert the column name to index rows
        inflation_f.index=pd.to_datetime(inflation_f.index, format='%Y%m') # convert the index (%Y%m) to datetime
        inflation_f.rename(columns={inflation_f.columns[0]:'CPI_F_'+country}, inplace=True) # rename column to ETH

        #inflation_f = (inflation_f - inflation_f.min()) / (inflation_f.max() - inflation_f.min()) # normalization

    else:

        inflation_SOM=inflation=pd.read_excel('CPI_SOM_data_portal.xlsx',header=6) # https://somalia.opendataforafrica.org/
        # Delete the M in the strings in the first column 
        inflation_SOM.iloc[:,0]=inflation_SOM.iloc[:,0].str.replace('M','')
        # set first column as index
        inflation_SOM.set_index(inflation_SOM.columns[0], inplace=True)
        # convert index to datetime 
        inflation_SOM.index=pd.to_datetime(inflation_SOM.index, format='%Y%m')
        
        ######################################## headline inflation ###################################
        # select first column 
        inflation_h=inflation_SOM.iloc[:,0]
        # rename to country name
        inflation_h.rename('CPI_H_'+country, inplace=True)

        #inflation_h = (inflation_h - inflation_h.min()) / (inflation_h.max() - inflation_h.min()) # normalization

        ######################################## food inflation ###################################
        # select second column
        inflation_f=inflation_SOM.iloc[:,1]
        # rename to country name
        inflation_f.rename('CPI_F_'+country, inplace=True)

        # normalize the timeseries
        #inflation_f = (inflation_f - inflation_f.min()) / (inflation_f.max() - inflation_f.min())

        

    # concatenate to inflation_df
    inflation_h_df=pd.concat([inflation_h_df,inflation_h], axis=1)
    inflation_h_df.index=pd.to_datetime(inflation_h_df.index)

    inflation_f_df=pd.concat([inflation_f_df,inflation_f], axis=1)
    inflation_f_df.index=pd.to_datetime(inflation_f_df.index)



    ############################################################################################################################################################################
    ############################################################################### GDP per capita #############################################################################
    ############################################################################################################################################################################

    GDP=pd.read_excel('GDP_PER_CAPITA_IMF.xlsx',header=0)
    
    ########## Ethiopia ############
    # select gdp ethiopia (row 3)
    GDP_eth= GDP.iloc[1,:]
    # delete first row
    GDP_eth=GDP_eth.iloc[1:]
    GDP_eth.index=pd.to_datetime(GDP_eth.index, format='%Y')
    # resample to monthly and ffil 
    GDP_eth=GDP_eth.resample('MS').ffill()
    
    ########## Somalia ############
    # select gdp ethiopia (row 3)
    GDP_som= GDP.iloc[3,:]
    # delete first row
    GDP_som=GDP_som.iloc[1:]
    GDP_som.index=pd.to_datetime(GDP_som.index, format='%Y')
    # resample to monthly and ffil 
    GDP_som=GDP_som.resample('MS').ffill()
    # rename no data to nan
    GDP_som=GDP_som.replace('no data', np.nan)

    ########## Kenya ############
    GDP_ken= GDP.iloc[2,:]
    # delete first row
    GDP_ken=GDP_ken.iloc[1:]
    GDP_ken.index=pd.to_datetime(GDP_ken.index, format='%Y')
    # resample to monthly and ffil 
    GDP_ken=GDP_ken.resample('MS').ffill()

    ########## concatenate ############
    GDP_df=pd.concat([GDP_eth, GDP_som, GDP_ken], axis=1)
    # rename columns
    GDP_df.columns=['GDP_Ethiopia','GDP_Somalia','GDP_Kenya']

    
    ######################################################## FUTURE WORK?  ########################################################################### 

    # WFP price forecasts? --> https://dataviz.vam.wfp.org/economic_explorer/price-forecasts-alerts
    #https://www.fao.org/sustainable-development-goals/indicators/211/en/
    # FSNAU for Somalia -->  https://data.humdata.org/dataset/fsna-2019-2020-somalia ###########################################        

    #population ###################### 
    #https://data.humdata.org/dataset/kenya-population-statistics-2019
    # population= pd.read_excel('ken_admpop_2019.xlsx',sheet_name='ken_admpop_ADM1_2019',index_col=0, header=0)
    # total_pop= population[['ADM1_NAME','T_TL']].set_index('ADM1_NAME')





    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ############################################################################################################################################################################
    ############################################################################## START CREATION INPUT DATAFRAME --> exported to data_master###################################
    ############################################################################################################################################################################
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    for county in units:
        county_df=pd.DataFrame()
        
        ##### extract unit ID from unit shapefile ---> use to extract raster data for specific county
        unit_ID= units_sf_ri[units_sf_ri['county']==county].OBJECTID.values[0]

        print ('county=', county)
 
        ############################################################################################################################################################################
        ################################################################################## FEWS ####################################################################################
        ############################################################################################################################################################################
        
        fews_county_ha_df=pd.DataFrame()
        fews_county_cs_df=pd.DataFrame()

        os.chdir(FC_FOLDER+'FEWS//')
        fews_lz_master= pd.DataFrame()
        
        

        for i in range(len(fews_xr.time.values)):
            
            fews_raster=fews_xr.isel(time=i)
            date=pd.Timestamp(fews_raster.time.values)
            fews_raster_county=fews_raster.where(units_raster==unit_ID, drop=True)

            # population weights 
            if population_weighted==True:
                pop_county=pop_EA_int.where(units_raster==unit_ID, drop=True)
                pop_county= pop_county.fillna(0) # nan values are not allowed in weights
                
            if population_weighted==False:
                # build CS dataframe
                fews_county_cs=pd.Series({'FEWS_CS':float(fews_raster_county.CS.mean(skipna=True))}, name=date)
                # build HA dataframe --> HA monitoring starts from April 2012, onwards. 
                fews_county_ha=pd.Series({'FEWS_HA':float(fews_raster_county.HA.mean(skipna=True))}, name=date)
                


            elif population_weighted==True:             
                # apply weights 
                fews_weighted=fews_raster_county.weighted(pop_county)
                fews_county_mean=fews_weighted.mean(('lon', 'lat'))
                # build CS dataframe
                fews_county_cs=pd.Series({'FEWS_CS':float(fews_county_mean.CS)}, name=date)
                # build HA dataframe --> HA monitoring starts from April 2012, onwards.
                fews_county_ha=pd.Series({'FEWS_HA':float(fews_county_mean.HA)}, name=date)



                                
            fews_county_cs_df=fews_county_cs_df._append(fews_county_cs)
            fews_county_ha_df=fews_county_ha_df._append(fews_county_ha)





        ######################################################### Append to county_df #########################################################       
        county_df=pd.concat([county_df,fews_county_cs_df], axis=1)
        county_df=pd.concat([county_df,fews_county_ha_df], axis=1)
        
        
        ############################################################################################################################################################################
        ################################################################################## Rainfall ####################################################################################
        ############################################################################################################################################################################
        
        
        ######################################################## CHIRPS rainfall indicators per county ###########################################  
        
        ##################% rainfall 
        rainfall_county= P_monthly.where(RAINFALL_mask==unit_ID, np.nan)
        tp_county_mean=rainfall_county.mean(dim=('latitude', 'longitude'))
        tp_county_mean=tp_county_mean.to_dataframe()
        
        
        ##################% wet days 
        wd_county= wd.where(RAINFALL_mask==unit_ID, np.nan) 
        wd_county_mean=wd_county.mean(dim=('latitude', 'longitude'))    
        wd_county_mean=wd_county_mean.to_dataframe()
        
        ##################% Dry spells 
        ds_county= ds.where(RAINFALL_mask==unit_ID, np.nan) 
        ds_county_mean=ds_county.mean(dim=('latitude', 'longitude'))    
        ds_county_mean=ds_county_mean.to_dataframe()

        rainfall_df=pd.merge(tp_county_mean, wd_county_mean, left_index=True, right_index=True, how='inner')
        rainfall_df=pd.merge(rainfall_df, ds_county_mean, left_index=True, right_index=True, how='inner')
        
        county_df=pd.concat([county_df,rainfall_df], axis=1)  

        
        ############################################################################################################################################################################
        ################################################################################## Climate indices ####################################################################################
        ############################################################################################################################################################################
        

        ################################################## IDENTIFY NAME DIFFERENCES IN COUNTIES #######################################
        # required counties 
        # county_sf=gpd.read_file(VECTOR_FOLDER+'geo_boundaries/HOA/HOA.shp')
        # counties= county_sf.county.to_list()

        # indices counties
        # indices_folder=INDICES_FOLDER+'spi/'
        # SPI=pd.read_excel(indices_folder+'spi_1.xlsx',index_col=0, header=0)
        # indices_counties= SPI.columns.to_list()
        # missing_in_indices=[i for i in counties if i not in indices_counties]

        indices_names=['Keiyo-Marakwet', 'Tharaka', "Murang'a"]
        correct_names=['Elgeyo Marakwet', 'Tharaka-Nithi', 'Muranga']

        



        os.chdir(INDICES_FOLDER)
        accumulation= [1,3,6,12,24]

        ############################## SPI ################################
        os.chdir(INDICES_FOLDER+'spi/')
        SPI_dataframe=pd.DataFrame() #see slack marthe for pixel level spi 
        for i in accumulation:
            SPI=pd.read_excel('spi_%s.xlsx'%(str(i)), index_col=0, header=0)
            # rename index according to correct_names
            SPI.rename(columns=dict(zip(indices_names, correct_names)), inplace=True)

            SPI_county=SPI[county]
            SPI_county=SPI_county.to_frame('SPI_'+str(i))# to frame with specific name 
            SPI_county.index=SPI_county.index+pd.offsets.MonthBegin(-1)# set to start of the month 
            SPI_dataframe=pd.concat([SPI_dataframe,SPI_county], axis=1)

        
        ############################## SPEI ################################
        os.chdir(INDICES_FOLDER+'spei/')
        SPEI_dataframe=pd.DataFrame() #see slack marthe for pixel level spi 
        for i in accumulation:
            SPEI=pd.read_excel('spei_%s.xlsx'%(str(i)), index_col=0, header=0)
            # rename index according to correct_names
            SPEI.rename(columns=dict(zip(indices_names, correct_names)), inplace=True)

            SPEI_county=SPEI[county]
            SPEI_county=SPEI_county.to_frame('SPEI_'+str(i))# to frame with specific name 
            SPEI_county.index=SPEI_county.index+pd.offsets.MonthBegin(-1)# set to start of the month 
            SPEI_dataframe=pd.concat([SPEI_dataframe,SPEI_county], axis=1)
        
        ############################## SSMI ################################

        os.chdir(INDICES_FOLDER+'ssmi/')
        SSMI_dataframe=pd.DataFrame() #see slack marthe for pixel level spi 
        for i in accumulation:
            SSMI=pd.read_excel('ssmi_%s.xlsx'%(str(i)), index_col=0, header=0)
            # rename index according to correct_names
            SSMI.rename(columns=dict(zip(indices_names, correct_names)), inplace=True)
            SSMI_county=SSMI[county]
            SSMI_county=SSMI_county.to_frame('SSMI_'+str(i))# to frame with specific name 
            SSMI_county.index=SSMI_county.index+pd.offsets.MonthBegin(-1)# set to start of the month 
            SSMI_dataframe=pd.concat([SSMI_dataframe,SSMI_county], axis=1)        
        
        ############################# Create indices dataframe ##############################
        
        indices_df=pd.concat([SPI_dataframe, SPEI_dataframe, SSMI_dataframe], axis=1)


        

        ############################# Add to county df ##############################
        county_df=pd.concat([county_df,indices_df], axis=1)

        ############################################################################################################################################################################
        ################################################################################## NDVI ####################################################################################
        ############################################################################################################################################################################
        

        NDVI_DF=pd.DataFrame()

        # NDVI anomalies 
        # NDVI 
        NDVI_anom_county= NDVI_anom.where(NDVI_mask==unit_ID, np.nan)
        NDVI_anom_county_mean=NDVI_anom_county.mean(dim=('latitude', 'longitude'))
        NDVI_DF['NDVI_anom']=NDVI_anom_county_mean.NDVI.to_dataframe()

        # NDVI range
        NDVI_anom_county_range= NDVI_anom_range.where(NDVI_mask==unit_ID, np.nan)
        NDVI_anom_county_mean_range=NDVI_anom_county_range.mean(dim=('latitude', 'longitude'))
        NDVI_DF['NDVI_anom_range']=NDVI_anom_county_mean_range.NDVI.to_dataframe()

        # NDVI crop
        NDVI_anom_county_crop= NDVI_anom_crop.where(NDVI_mask==unit_ID, np.nan)
        NDVI_anom_county_mean_crop=NDVI_anom_county_crop.mean(dim=('latitude', 'longitude'))
        NDVI_DF['NDVI_anom_crop']=NDVI_anom_county_mean_crop.NDVI.to_dataframe()
        
        NDVI_DF.reset_index(inplace=True)
        NDVI_DF.rename(columns={'time': 'date'}, inplace=True)     
        NDVI_DF.set_index('date', inplace=True)


        county_df=pd.concat([county_df, NDVI_DF], axis=1)  

        ############################################################################################################################################################################
        ################################################################################## SST ####################################################################################
        ############################################################################################################################################################################
        
        ########################## SST data ##########################
        county_df=county_df.merge(WVG, how='left', left_index=True, right_index=True)# merge WVG to county_df
        county_df=county_df.merge(WPG, how='left', left_index=True, right_index=True)# merge WPG to county_df
        county_df=county_df.merge(MEI, how='left', left_index=True, right_index=True)# merge MEI to county_df
        county_df=county_df.merge(NINA34, how='left', left_index=True, right_index=True)# merge NINA34 to county_df
        county_df=county_df.merge(IOD, how='left', left_index=True, right_index=True)# merge IOD to county_df

        ############################################################################################################################################################################
        ################################################################################## PRICES ####################################################################################
        ############################################################################################################################################################################
        
        ######################################################## WFP VAM MARKET PRICES  ########################################### 
        
        os.chdir(SE_FOLDER+'/'+country)
        for good in goods: 
            os.chdir(SE_FOLDER+'/'+country + '/' + good)
            good_prices=pd.read_excel('market_data_%s_%s_%s.xlsx'%(county, country, good), index_col=0, header=0)

            # convert Price and Pewi columns to floats
            # first, replace commas with dots
            good_prices['Price']=good_prices['Price'].str.replace(',','.')
            good_prices['Price']=good_prices['Price'].astype(float)

            # check if units are always the same 
            if len(good_prices['Unit'].unique())>1:
                print ('MULTIPLE UNIT TYPES FOR %s'%(good))
                print ('unit types are... %s'%(good_prices['Unit'].unique()))
                print ('conversion in progress...')
                units_record= good_prices['Unit']

                if '90 KG' in units_record.values: 
                    # devide the Price column by 90 for the rows with Unit=='90 KG'
                    good_prices.loc[good_prices['Unit']=='90 KG', 'Price']=good_prices.loc[good_prices['Unit']=='90 KG', 'Price']/90 
                    # change the unit to KG
                    good_prices.loc[good_prices['Unit']=='90 KG', 'Unit']='KG'
                
                if '100 KG' in units_record.values:
                    # devide the Price column by 100 for the rows with Unit=='100 KG'
                    good_prices.loc[good_prices['Unit']=='100 KG', 'Price']=good_prices.loc[good_prices['Unit']=='100 KG', 'Price']/100
                    # change the unit to KG
                    good_prices.loc[good_prices['Unit']=='100 KG', 'Unit']='KG'

                    
                
                if len(good_prices['Unit'].unique())>1:
                    print ('ERROR, STILL !! MULTIPLE UNIT TYPES FOR %s'%(good))
                    print ('unit types are... %s'%(good_prices['Unit'].unique()))
                    print ('sys will exit')
                    exit()


            
            good_prices= good_prices[indicator]
            good_prices.rename('%s_%s'%(good, indicator), inplace=True)
            good_prices=good_prices.groupby(good_prices.index).mean() #average over datetime duplicates (retail vs wholesale)

            county_df=county_df.merge(good_prices, how='left', left_index=True, right_index=True)


        
        ###########################################################################################################################################################################
        ####################################################################################### Conflict data   ####################################################################
        ############################################################################################################################################################################
        #Counties not in acled: 
        #Ethiopia -->  ['Basketo', 'Yem Special', 'Majang', 'North Shewa (OR)'] 
        #Kenya --> 
        #Som --> 
        # These counties are filled with 0's. 

        acled_county=acled[acled['county']==county] # select county
        acled_county.insert(2,'acled_count', 1) # add column with value 1 for each conflict
        acled_county=acled_county.drop('county', axis=1)#drop county column
        
        ############# Monthly sums ###############
        acled_county=acled_county.resample('MS').sum() # returns conflicts counted per month --> already set months with NAN to 0 , which is what we want
        

        ############# Make sure each date is present --> fill with 0 otherwise ############### 
        start_date='2000-01-01'
        end_date=str(fews_xr.time.values[-1:])[2:12]
        date_list= pd.date_range(start=start_date, end=end_date, freq='MS') 
        acled_county=acled_county.reindex(date_list, fill_value=0) 

        county_df=county_df.merge(acled_county, how='left', left_index=True, right_index=True)


        ###########################################################################################################################################################################
        ########################################################################################### Inflation   #####################################################################
        ############################################################################################################################################################################
        inflation_f_append=inflation_f_df['CPI_F_'+country] # attach inflation from specific country 
        inflation_f_append.rename('CPI_F', inplace=True) # rename because in that way easier to read out in the model 
        inflation_h_append=inflation_h_df['CPI_H_'+country] 
        inflation_h_append.rename('CPI_H', inplace=True) # rename because in that way easier to read out in the model 

        county_df=county_df.merge(inflation_f_append, how='left', left_index=True, right_index=True)# merge food inflation to county_df
        county_df=county_df.merge(inflation_h_append, how='left', left_index=True, right_index=True)# merge headline inflation to county_df


        ###########################################################################################################################################################################
        ###########################################################################################  GDP   ##########################################################################
        ############################################################################################################################################################################
        GDP_append=GDP_df['GDP_'+country] # attach GDP from specific country
        GDP_append.rename('GDP', inplace=True) # rename because in that way easier to read out in the model
        county_df=county_df.merge(GDP_append, how='left', left_index=True, right_index=True)# merge GDP to county_df
        
        ###########################################################################################################################################################################
        ######################################################################################## DESERT LOCUST (% of county area)   ###################################################################
        ############################################################################################################################################################################
        locust_unit=locust[locust['unit']==county]
        locust_unit=locust_unit.drop('unit', axis=1)       

        start_date='2000-01-01'
        end_date=str(fews_xr.time.values[-1:])[2:12]
        date_list= pd.date_range(start=start_date, end=end_date, freq='MS') 

        locust_unit=locust_unit.reindex(date_list, fill_value=0) # fill na with 0
        locust_unit.rename(columns={'AREAHA':'DL_area'}, inplace=True) # rename column to locust




        # calculate county area
        county_sf=units_sf_ri[units_sf_ri['county']==county]
        county_sf= county_sf.to_crs('EPSG:3857') # to projected coordinate system (m) 
        county_sf['area_calculated']=county_sf['geometry'].area # calculate area (m2)
        county_sf['area_calculated']=county_sf['area_calculated']/10000 # convert to hectares
        area=county_sf['area_calculated'].values[0] # select area

        # percentage of county affected by locust
        locust_unit['DL_area']=(locust_unit['DL_area']/area)*100

        print('county is %s and area is %s ha'%(county, area))
        # locust_unit[locust_unit['DL_area']>0.0]      

        county_df=county_df.merge(locust_unit, how='left', left_index=True, right_index=True)# merge locust to county_df
        


        ############################################################################################################################################################################
        ####################################################################################### TIME CUT OFF ##########################################################################
        ############################################################################################################################################################################      
        # Start dataframe from 2000-01-01
        county_df=county_df[county_df.index>='2000-01-01']
        
        ############################################################################################################################################################################
        ############################################################# NAN processing method: interpolate in-between values, fill other values with mean#############################
        ############################################################################################################################################################################
        
        columns_to_impute= [column for column in county_df.columns if column not in ['FEWS_CS','county','aridity', 'lhz']]
        
        # for values in between other values: linear interpolation
        # for values which are not in between other values: fill with county mean 
        
        for column in columns_to_impute:
            # linear interpolation
            county_df[column].interpolate(method='linear', inplace=True)
            # fill other nan's with county mean
            county_df[column].fillna((county_df[column].mean()), inplace=True) 

        ############################################################################################################################################################################
        ############################################################################## ATTACH COUNTY CHARACTERISTICS  ##############################################################
        ############################################################################################################################################################################
        lhz=pd.read_excel(BASE_FOLDER+'/livelihood_zones.xlsx', index_col=0) # livelihood zones
        lhz_county=lhz[lhz.index==county]['max'].values[0]
        county_df.insert(len(county_df.columns), 'lhz', lhz_county)

        
        county_df['county']= county
        county_df['country']= country

        ############################################################################################################################################################################
        ############################################################################# APPEND COUNTY DF'S TO INPUT_DATAFRAME ########################################################
        ############################################################################################################################################################################
        print('dataframe build for %s in %s'%(county,country))

        input_dataframe=pd.concat([input_dataframe, county_df], axis=0)



    ############################################################################################################################################################################
    ############################################################# CREATE EXCEL FILE  ##########################################################
    ############################################################################################################################################################################

    print ('INPUT COLLECTION FINISHED for %s'%(country))

input_dataframe.to_excel(BASE_FOLDER+'/data_master.xlsx', freeze_panes=(1,1)) # FINAL EXCEL FILE 

print ('INPUT COLLECTION FINISHED!')


