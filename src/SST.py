

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:33:24 2023

@author: Tim Busker 

Script that extracts the SST indices from the NOAA websites, and saves them to an existing excel file SST_index.xlsx. In this excel file, the SST's from the WVG are already present (extracted from FUNK et al. 2023)
"""


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


wd=r'C:\Users\USER\Documents\ML\input_collector\CLIM_data\SST'
os.chdir(wd)


################################################################################################################################
###################################################### NOAA SST INDICES ########################################################
################################################################################################################################

################### Function to read NOAA DATA ##############################

# define a function to read the meiv2.txt file
def read_NOAA(df,variable):
    # make a list of the months
    months=[1,2,3,4,5,6,7,8,9,10,11,12]
    # make a list of the years
    years=df[0].values
    # make a list of the dates
    dates=[str(year)+'-'+str(month) for year in years for month in months]
    # convert the list of dates to a datetime index
    dates=pd.to_datetime(dates, format='%Y-%m')

    # get the values from the dataframe 
    values=df.iloc[:,1:].values.flatten()

    # make a dataframe with the datetime index and the values
    x=pd.DataFrame(values, index=dates, columns=[variable])
    return x

################################################### Multivariate ENSO Index Version 2 (MEI.v2)########################################################
# https://www.psl.noaa.gov/enso/mei
# This is multivariate, combining sea level temperatures and atmospheric pressures. 
df=pd.read_csv('meiv2.txt', sep='\s+', header=None, skiprows=1)
df=read_NOAA(df,'MEI')

# add the dataframe to a new tab in SST_index.xlsx excel file 
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='MEI')

####################################################  Nino Anom 3.4 Index  using ersstv5 from CPC ########################################################
# https://psl.noaa.gov/data/climateindices/list/for info
df=pd.read_csv('nina34.txt', sep='\s+', header=None, skiprows=1)
df=read_NOAA(df,'NINA34')
# add the dataframe to a new tab in SST_index.xlsx excel file
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='NINA34')

    
####################################################  IOD index  ########################################################
# source: https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data

# download data from this url link, save it to local disk and read it into a pandas dataframe. Skip last 7 rows 
df=pd.read_csv('https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data', sep='\s+', header=None, skiprows=1, skipfooter=7, engine='python')
df=read_NOAA(df,'IOD')

# write df to csv file --> needed because otherwise dates <1990 are not read correctly
df.to_csv('IOD.csv')
df=pd.read_csv('IOD.csv', index_col=0)
# write 'IOD.csv' to a new tab in SST_index.xlsx excel file
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='IOD')


###################################################  WVG (FUNK ET AL) ########################################################

# read the SST_index.xlsx excel file, tab WVG
df=pd.read_excel('SST_index.xlsx', sheet_name='WVG',skiprows=10)
df.drop('Unnamed: 2', axis=1, inplace=True)

# make a list of monthly datetimes using the year column. month column doesnt exist
dates=[str(year)+'-'+str(month) for year in df['year'] for month in range(1,13)]

# convert the list of dates to a datetime index
dates=pd.to_datetime(dates, format='%Y-%m')

# repeat every value in the WVG column 12 times to match the length of the datetime index
values=np.repeat(df['MAM WVG Values'].values,12)
df=pd.DataFrame(values, index=dates, columns=['WVG'])

# set all months except march-april-may to NaN
df.loc[(df.index.month!=3)&(df.index.month!=4) & (df.index.month!=5),'WVG']=np.nan

# add the dataframe to a new tab in SST_index.xlsx excel file
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='WVG_processed')

################################################## WPG (NOT USED IN PAPER) ########################################

# read the SST_index.xlsx excel file, tab WVG
df=pd.read_excel('SST_index.xlsx', sheet_name='WVG',skiprows=17)


# rename first column to year
df.rename(columns={'Unnamed: 0':'year'}, inplace=True)

# make a list of monthly datetimes using the year column. month column doesnt exist
dates=[str(year)+'-'+str(month) for year in df['year'] for month in range(1,13)]

# convert the list of dates to a datetime index
dates=pd.to_datetime(dates, format='%Y-%m')

# repeat every value in the WVG column 12 times to match the length of the datetime index
values=np.repeat(df['OND WPG Values'].values,12)
df=pd.DataFrame(values, index=dates, columns=['WPG'])

# set all months except october-november-december to NaN
df.loc[(df.index.month!=10)&(df.index.month!=11) & (df.index.month!=12),'WPG']=np.nan

# add the dataframe to a new tab in SST_index.xlsx excel file
with pd.ExcelWriter('SST_index.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='WPG_processed')

