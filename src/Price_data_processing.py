
"""
Created on Mon Jan 23 15:33:24 2023

@author: Tim Busker 

This script calculates the distance between the centroids of the counties and the markets around it per country. The result is a dataframe with the 3 closest markets and their distances (market_summary.xlsx). 
This information is used to create a file (market_data.xlsx) with the market data for each county, where data gaps in the WFP market prices for that county are filled with the closest county that has data recorded in that timestep. 


"""
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
from geopy import geocoders
import geopy.distance
from geopy.geocoders import Nominatim # openstreetmap geocoder
import geoplot as gplt
import geoplot.crs as gcrs
import time
geolocator = Nominatim(user_agent="hornofafrica")

#####################################################################################################################
################################################### SET DIRECTORIES #################################################
#####################################################################################################################

#input data 
BASE_FOLDER='PATH/ML/input_collector'# 'PATH/ML/input_collector' #'C:/Users/USER/Documents/ML/input_collector'
# Vectors
VECTOR_FOLDER=BASE_FOLDER+'/Vector/'#'C:/Users/USER/Documents/Forecast_action_analysis/vector_data/'

SE_FOLDER= BASE_FOLDER+'/SE_data/'


indicator= 'Pewi' # Price or Pewi


#####################################################################################################################
################################################### START COUNTRY LOOP ##############################################
#####################################################################################################################

for country in ['Kenya', 'Somalia','Ethiopia']: #

    if country=='Somalia':
        goods=['Maize (white)', 'Fuel (diesel)','Sorghum'] # Fuel prices are really available! From 2014-2023 from many markets 


    if country=='Ethiopia':
        goods=['Maize (white)','Fuel (diesel)', 'Sorghum'] # fuel prices from addis ababa only (2014-2020). After 2020, fuel prices from different markets 
        
    if country=='Kenya':
        goods=['Maize (white)','Fuel (diesel)','Sorghum'] # fuel prices from nairobi only (2014-2020). Food prices until start 2020... 

    COUNTRY_FOLDER=SE_FOLDER+country+'/'
    

    if not os.path.exists(COUNTRY_FOLDER):
        os.makedirs(COUNTRY_FOLDER)

    

    for good in goods:
        GOOD_FOLDER=COUNTRY_FOLDER+good
        if not os.path.exists(GOOD_FOLDER):
            os.makedirs(GOOD_FOLDER)

        # delete files 
        for file in os.listdir(GOOD_FOLDER):
            os.remove(GOOD_FOLDER+'/'+file)
        
        #####################################################################################################################
        ################################################### LOAD PEWI DATA (WHOLESALE AND RETAIL)############################
        #####################################################################################################################

        os.chdir(COUNTRY_FOLDER)
        tabular_data_retail= pd.read_csv('WFP_%s_FoodPrices_new_tool_retail.csv'%(country), sep=';')  # averaging over duplicates retail-wholesale happens in input_collector.py
        if country!='Somalia':
            tabular_data_wholesale= pd.read_csv('WFP_%s_FoodPrices_new_tool_wholesale.csv'%(country), sep=';')
            tabular_data= pd.concat([tabular_data_wholesale,tabular_data_retail], axis=0)
        else: 
            tabular_data=tabular_data_retail.copy()

        tabular_data.rename(columns={'Market Name':'market'}, inplace=True)
        

        if country=='Ethiopia':
            # rename Sorghum (red) and Sorghum (white) to Sorghum

            tabular_data['Commodity']=tabular_data['Commodity'].replace('Sorghum (red)','Sorghum')
            tabular_data['Commodity']=tabular_data['Commodity'].replace('Sorghum (white)','Sorghum')

        if country=='Somalia': 
            # rename Sorghum (red) and Sorghum (white) to Sorghum
            tabular_data['Commodity']=tabular_data['Commodity'].replace('Sorghum (red)','Sorghum')
            tabular_data['Commodity']=tabular_data['Commodity'].replace('Sorghum (white)','Sorghum') 

        # only obs 
        tabular_data=tabular_data[tabular_data['Data Type']!='Forecast']
        
        # select good 
        tabular_data=tabular_data[tabular_data['Commodity']==good]

        # tabular_data['Price Date'] as datetime
        tabular_data['Price Date']=pd.to_datetime(tabular_data['Price Date'])    
        tabular_data=tabular_data.rename(columns={'Price Date':'date'})# rename Price Date to date

        # convert date (e.g. 2000-01-15) to first day of the month (e.g. 2000-01-01)
        tabular_data['date']=tabular_data['date'].dt.to_period('M').dt.to_timestamp()
        
        
        if country == 'Ethiopia':
            # Rename Nazareth to Adama
            tabular_data['market']=tabular_data['market'].replace('Nazareth','Adama')
        
        # Drop rows which are nan in price column
        tabular_data=tabular_data.dropna(subset=[indicator])

        print('Calculation with %s markets for %s'%(len(tabular_data['market'].unique()), country))

        # create dictonary with market names and their coordinates
        market_names=tabular_data['market'].unique() 


        
        #####################################################################################################################
        ################################################### GEOLOCATE MARKETS ###############################################
        #####################################################################################################################

        # insert ethiopia manually in front of each market name --> makes it easier to geocode
        market_names=[country+' '+market for market in market_names]

        market_coordinates= pd.DataFrame(columns=['market','latitude','longitude'])

        for market in market_names:
            print (market)
            time.sleep(1) # needed because max capacity of nominatim is 1 request per second -> https://operations.osmfoundation.org/policies/nominatim/#requirements
            location = geolocator.geocode(market)
            if location is not None:
                market_coordinates=pd.concat([market_coordinates,pd.DataFrame({'market':market,'latitude':location.latitude,'longitude':location.longitude}, index=[0])], axis=0)
            else:
                print('location not found for '+market)
                print ('deleted from market list')
                market_names=[x for x in market_names if x != market]




        markets_gdf=gpd.GeoDataFrame(market_coordinates, geometry=gpd.points_from_xy(market_coordinates.longitude, market_coordinates.latitude))    


        if country == 'Ethiopia':
            basemap= gpd.read_file(VECTOR_FOLDER+'geo_boundaries/'+country+'/eth_admbnda_adm2_csa_bofedb_2021.shp')
            basemap.rename(columns={'ADM2_EN':'county'}, inplace=True)

        if country == 'Somalia':
            basemap= gpd.read_file(VECTOR_FOLDER+'geo_boundaries/'+country+'/Som_Admbnda_Adm2_UNDP.shp')
            basemap.rename(columns={'admin2Name':'county'}, inplace=True)

        if country == 'Kenya':
            basemap= gpd.read_file(VECTOR_FOLDER+'geo_boundaries/'+country+'/County.shp')
            basemap.rename(columns={'COUNTY':'county'}, inplace=True)
        basemap['geometry_point']=basemap['geometry'].centroid

        

        market_summary=pd.DataFrame(columns=['unit','closest_market','second_closest_market', 'third_closest_market','distance1', 'distance2', 'distance3'])
        missing_dates_df=pd.DataFrame(columns=['unit','missing_dates'])
        
        #####################################################################################################################
        ##################################### FIND CLOSEST MARKET WITH DATA #################################################
        #####################################################################################################################

        for unit in basemap['county'].unique():
            os.chdir(COUNTRY_FOLDER)
            print (unit)
            market_distances=pd.DataFrame(columns=['market','distance'])
            unit_loc= basemap[basemap['county']==unit]['geometry_point']
            for market in market_names:
                # market loc 
                market_loc=markets_gdf[markets_gdf['market']==market]['geometry']
                
                # market lat lon 
                market_loc_lat=str(market_loc.y.values[0])
                market_loc_lon=str(market_loc.x.values[0])
                # unit lat lon
                unit_loc_lat=str(unit_loc.y.values[0])
                unit_loc_lon=str(unit_loc.x.values[0])

                # create coordinate string for geopy
                market_loc_str=market_loc_lat+','+market_loc_lon
                unit_loc_str=unit_loc_lat+','+unit_loc_lon

                # distance between market and unit
                distance=geopy.distance.geodesic(market_loc_str,unit_loc_str).km

                market_distances.loc[len(market_distances)]=[market,distance]
            
            market_distances['market']=market_distances['market'].str.replace('%s '%(country),'')

            # find closest market
            market_distances=market_distances.sort_values(by='distance')
            nearest_market=market_distances.iloc[0]
            
            # check if nearest market has data for all timesteps since 2009 
            tabular_data_market=tabular_data[tabular_data['market']==nearest_market['market']]
            # drop rows with nan values in price column
            
            provided_dates = tabular_data_market['date'].tolist()
            
            provided_dates=[str(i)[:10] for i in provided_dates]

            start_date= '2009-01-01'
            end_date=tabular_data['date'].max()
            end_date=str(end_date)[:10]
            all_dates = pd.date_range(start=start_date, end=end_date, freq='MS').strftime("%Y-%m-%d").tolist()
            missing_dates= [i for i in all_dates if i not in provided_dates]
            
            
            
            for i in market_distances.market[1:]:
                if len(missing_dates)>0:
                    additional_market=tabular_data[tabular_data['market']==i]

                    # new market dates
                    dates_new_market=additional_market['date'].tolist()
                    dates_new_market=[str(i)[:10] for i in dates_new_market]

                    # dates necessary to fill in
                    extra_dates= [i for i in missing_dates if i in dates_new_market]

                    # input the data from this market (additional_market) for the extra dates into the tabular_data_market dataframe 
                    if len(extra_dates)>0:
                        additional_market=additional_market[additional_market['date'].isin(pd.to_datetime(extra_dates))]
                        tabular_data_market=pd.concat([tabular_data_market,additional_market], axis=0)
                        tabular_data_market= tabular_data_market.sort_values(by='date')
                        


                        # check if all dates are filled in
                        provided_dates = tabular_data_market['date'].tolist()
                        provided_dates=[str(i)[:10] for i in provided_dates]
                        missing_dates= [i for i in all_dates if i not in provided_dates]

                        
                        print ('still missing dates: %s'%(missing_dates))   

                        
            if len(missing_dates)>0:
                print ('Not all dates available from current market data for %s'%(unit))
                print ('DATES NOT COLLECTED: %s'%(missing_dates))

            
            os.chdir(GOOD_FOLDER)
            tabular_data_market.set_index('date', inplace=True) 

            # convert price str to floats 
            # replace all , with . in price column 
            tabular_data_market[indicator]=tabular_data_market[indicator].str.replace(',','.')
            tabular_data_market[indicator]=tabular_data_market[indicator].astype(float)
            
            
            
            
            
            #####################################################################################################################
            ################################################### SAVE MARKET DATA  #################################################
            #####################################################################################################################
            tabular_data_market.to_excel('market_data_%s_%s_%s.xlsx'%(unit,country,good))

            
            #####################################################################################################################
            ################################################### SAVE MARKET SUMMARY #############################################
            #####################################################################################################################


            if len(market_distances)<3: # Fuel prices for kenya are only available from Nairobi
                market_summary.loc[len(market_summary)]=[unit,nearest_market['market'],np.nan,np.nan,nearest_market['distance'],np.nan,np.nan]

            else: 
                market_summary.loc[len(market_summary)]=[unit,nearest_market['market'],market_distances.iloc[1]['market'],market_distances.iloc[2]['market'],nearest_market['distance'],market_distances.iloc[1]['distance'],market_distances.iloc[2]['distance']]
        
        missing_dates_df.loc[len(missing_dates_df)]=[unit,missing_dates]

        # market summary. This dataframe contains the closest market for each admin unit and good, and the distance to that market.
        market_summary.to_excel('market_summary_%s_%s.xlsx'%(country,good))
        
        # missing dates. This dataframe contains the admin units for which not all dates are available from the current market data.
        missing_dates_df.to_excel('missing_dates_%s_%s.xlsx'%(country,good))


