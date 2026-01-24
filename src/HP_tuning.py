# -*- coding: utf-8 -*-
"""
Optimized for Google Vertex AI Hyperparameter Tuning
Original Author: Tim Busker / USER

Script that performs HP tuning on the data.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# SKLEARN 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# XGBOOST
from xgboost import XGBRegressor

# Cloud-native pathing
from src.ML_functions import *
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("Starting drought prediction training...")

# Vertex AI Environment Setup
BASE_DIR = Path(os.getenv("AIP_CHECKPOINT_DIR", os.getcwd()))
DATA_FOLDER = BASE_DIR / 'input_collector'
HP_RESULT_ROOT = BASE_DIR / 'ML_results' / 'HP_results'

HP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)

################################################### DESIGN VARIABLES ###################################################

model_list = ['xgb']
region_list = ['HOA']
aggregations = ['cluster'] 
experiment_list = ['RUN_FINAL_20'] 
leads = [0, 1, 2, 3, 4, 8, 12] 
CV_TYPE = 'TSS' 

with_WPG = False

design_variables = [(experiment, model_type, aggregation, region) 
                    for experiment in experiment_list 
                    for model_type in model_list 
                    for aggregation in aggregations 
                    for region in region_list]

#######################################################################################################################
################################################### EXECUTION LOOP ####################################################
#######################################################################################################################

for experiment, model_type, aggregation, region in design_variables:
    
    print(f'Starting HP Tuning: {experiment} | Model: {model_type}')
    
    traintest_ratio = int(experiment[-2:]) / 100

    # Load Data
    input_file = DATA_FOLDER / 'input_master.csv'
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input data at {input_file}")
        
    input_master = pd.read_csv(input_file, index_col=0)
    input_master.index = pd.to_datetime(input_master.index)
    input_master.drop('year', axis=1, inplace=True, errors='ignore')
    
    if not with_WPG:
        input_master = input_master[input_master.columns.drop(list(input_master.filter(regex='WPG')))]

    # Aggregation Setup
    if aggregation == 'cluster':
        cluster_list = ['p', 'ap', 'other']
    else:
        cluster_list = ['no_cluster']

    for cluster in cluster_list:
        HP_RUN_FOLDER = HP_RESULT_ROOT / f"{aggregation}_{experiment}_{region}_{cluster}_{model_type}"
        HP_RUN_FOLDER.mkdir(parents=True, exist_ok=True)

        if aggregation == 'cluster':
            input_df2 = input_master[input_master['lhz'] == cluster].dropna(axis=1, how='all')
            units = [f'cluster_{cluster}']
        else:
            input_df2 = input_master.copy()
            units = ['all']

        for county in units:
            for lead in leads:
                # Filter for lead time
                input_df3 = input_df2[input_df2['lead'] == lead].sort_index()
                if input_df3.empty: 
                    continue

                # 1. SYNCHRONIZED CLEANING
                input_df3 = input_df3.replace([np.inf, -np.inf], np.nan)
                input_df3 = input_df3.dropna(subset=['FEWS_CS'])

                if input_df3.empty:
                    logger.warning(f"Skipping {county} Lead {lead}: No valid labels found.")
                    continue

                # 2. DEFINE LABELS AND FEATURES
                labels = input_df3['FEWS_CS']
                features = input_df3.drop(['lead', 'base_forecast', 'FEWS_CS'], axis=1, errors='ignore')
                
                # 3. Handle categorical features
                cat_cols = features.select_dtypes(include=['object', 'category']).columns 
                features.drop(cat_cols, axis=1, inplace=True)

                # 4. Split
                train_X, test_X, train_y, test_y = train_test_split(
                    features, labels, test_size=traintest_ratio, shuffle=False
                )

                # Define the Search Grid
                if model_type == 'xgb':
                    base_model = XGBRegressor(random_state=42)
                    grid = { 
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [4, 6, 8],
                        "n_estimators": [200, 400],
                        "subsample": [0.7, 1.0]
                    }
                else:
                    base_model = RandomForestRegressor(random_state=42)
                    grid = {
                        'n_estimators': [200, 400],
                        'max_depth': [4, 10, None],
                        'max_features': ['sqrt', None]
                    }

                # Setup Cross-Validation
                cv_strategy = TimeSeriesSplit(n_splits=5)

                # Run Search
                search = GridSearchCV(
                    estimator=base_model, 
                    param_grid=grid, 
                    scoring='neg_mean_squared_error', 
                    cv=cv_strategy, 
                    n_jobs=-1
                )
                
                search.fit(train_X, train_y)

                # --- SAVE THE BEST PARAMETERS ---
                best_params = search.best_params_
                json_path = HP_RUN_FOLDER / f"best_params_{model_type}_L{lead}.json"
                save_best_params(best_params, json_path)

                # Save the full CV results
                results_df = pd.DataFrame(search.cv_results_)
                results_df.to_excel(HP_RUN_FOLDER / f'CV_details_L{lead}.xlsx')

print("Hyperparameter tuning finished. Best settings saved to GCS.")