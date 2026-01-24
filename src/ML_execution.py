
"""
Optimized for Google Vertex AI
Original Author: Tim Busker

This script executes the machine learning models on the High Performance Cluster (HPC).
It uses the input_data.csv file created by the input_collector.py and feature_engineering.py scripts. 
Current implementation allows for a random forest model or an XGBoost model, on multiple spatial levels
(county, country, livelihood zone, all). It was also tested whether the variation of FEWS IPC was
 a good way of pooling the counties. This was not the case. 
"""


#####################################################################################################################
################################################### IMPORT PACKAGES  ###################################################
#####################################################################################################################
import os
import time
import datetime
import random as python_random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from tqdm.auto import tqdm

# SKLEARN 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# XGBOOST & SHAP
from xgboost import XGBRegressor
import shap

# LOGGING
import wandb

# Cloud-native pathing: Import from the package structure
# Ensure save_best_params and load_best_params are in your src/ML_functions.py
from src.ML_functions import * ################################################### ENVIRONMENT SETUP ###################################################

# Vertex AI sets AIP_CHECKPOINT_DIR to your GCS bucket path
BASE_DIR = Path(os.getenv("AIP_CHECKPOINT_DIR", os.getcwd()))
DATA_FOLDER = BASE_DIR / 'input_collector'
RESULTS_DIR = BASE_DIR / 'ML_results'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Ensure directories exist (Pathlib handles GCS fuse mounts correctly)
for folder in [RESULTS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

################################################### DESIGN VARIABLES ###################################################

model_list = ['xgb']
region_list = ['HOA'] 
aggregations = ['cluster'] # Options: 'cluster', 'all', 'county'
experiment_list = ['RUN_FINAL_20'] 
leads = [0, 1, 2, 3, 4, 8, 12] 

with_WPG = False 
end_date = '2022-06-01 00:00:00' 

design_variables = [(experiment, model_type, aggregation, region) 
                    for experiment in experiment_list 
                    for model_type in model_list 
                    for aggregation in aggregations 
                    for region in region_list]

#######################################################################################################################
################################################### EXECUTION LOOP ####################################################
#######################################################################################################################

for experiment, model_type, aggregation, region in design_variables:
    
    print(f'Starting Execution: {experiment} | Model: {model_type}')
    
    # Initialize WandB for cloud experiment tracking
    wandb.init(
        project="drought_forecasting",
        name=f"Final_{experiment}_{model_type}",
        config={"leads": leads, "aggregation": aggregation}
    )

    traintest_ratio = int(experiment[-2:]) / 100 

    # Load Data
    input_path = DATA_FOLDER / 'input_master.csv'
    if not input_path.exists():
        raise FileNotFoundError(f"Data not found at {input_path}")
        
    input_master = pd.read_csv(input_path, index_col=0)
    input_master.index = pd.to_datetime(input_master.index)
    input_master.drop('year', axis=1, inplace=True, errors='ignore')

    if not with_WPG:
        input_master = input_master[input_master.columns.drop(list(input_master.filter(regex='WPG')))]

    # Clustering / Aggregation Logic
    if aggregation == 'cluster':
        cluster_list = ['p', 'ap', 'other']
    else:
        cluster_list = ['no_cluster']

    for cluster in cluster_list:
        preds_storage = pd.DataFrame()
        eval_stats = pd.DataFrame()

        if aggregation == 'cluster':
            input_df2 = input_master[input_master['lhz'] == cluster].dropna(axis=1, how='all')
            units = [f'cluster_{cluster}']
        else:
            input_df2 = input_master.copy()
            units = list(input_master['county'].unique())

        for county in units:
            for lead in leads:
                # Filter for lead time
                input_df3 = input_df2[input_df2['lead'] == lead].sort_index()
                if input_df3.empty: continue

                # Clean features for ML
                labels = input_df3['FEWS_CS']
                features = input_df3.drop(['lead', 'base_forecast', 'FEWS_CS'], axis=1, errors='ignore')
                cat_cols = features.select_dtypes(include=['object', 'category']).columns 
                features.drop(cat_cols, axis=1, inplace=True)

                # Split
                train_X, test_X, train_y, test_y = train_test_split(
                    features, labels, test_size=traintest_ratio, shuffle=False
                )

                #####################################################################
                # --- DYNAMIC HYPERPARAMETER LOADING (Linked to HP_tuning.py) ---
                #####################################################################
                hp_file_path = RESULTS_DIR / 'HP_results' / f"{aggregation}_{experiment}_{region}_{cluster}_{model_type}" / f"best_params_{model_type}_L{lead}.json"
                
                # Defaults if HP_tuning hasn't run yet
                defaults = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.01}
                
                # Load JSON from GCS via helper function
                tuned_params = load_best_params(hp_file_path, defaults)

                if model_type == 'xgb':
                    model_obj = XGBRegressor(**tuned_params, random_state=42)
                else:
                    model_obj = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)

                # Fit
                model_obj.fit(train_X, train_y)
                preds = model_obj.predict(test_X)

                #####################################################################
                # --- EVALUATION & LOGGING ---
                #####################################################################
                mae = mean_absolute_error(test_y, preds)
                rmse = np.sqrt(mean_squared_error(test_y, preds))
                r2 = r2_score(test_y, preds)

                # Store Stats
                iter_stats = pd.DataFrame({
                    'county': county, 'lead': lead, 'mae': mae, 'rmse': rmse, 'r2': r2
                }, index=[0])
                eval_stats = pd.concat([eval_stats, iter_stats])

                # SHAP Plots (Sent directly to WandB)
                explainer = shap.Explainer(model_obj)
                shap_values = explainer(train_X)
                fig, ax = plt.subplots()
                shap.plots.bar(shap_values, show=False)
                wandb.log({f"SHAP_{county}_L{lead}": wandb.Image(plt)})
                plt.close(fig)

        # Save Final CSVs to GCS
        stats_name = f'eval_stats_{aggregation}_{cluster}_{experiment}.csv'
        eval_stats.to_csv(RESULTS_DIR / stats_name)
        wandb.save(str(RESULTS_DIR / stats_name))

    wandb.finish()

print("Execution finished. All results synced to GCS.")