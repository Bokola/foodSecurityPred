"""
Created on Mon Jan 23 15:33:24 2023

@author: Tim Busker 

This script executes the machine learning models on the High Performance Cluster (HPC).
It uses the input_data.csv file created by the input_collector.py and feature_engineering.py scripts. 
Current implementation allows for a random forest model or an XGBoost model, on multiple spatial levels
(county, country, livelihood zone, all). It was also tested whether the variation of FEWS IPC was
 a good way of pooling the counties. This was not the case. 

Reorganized for vertex AI environment by Basil Okola.

"""
import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import wandb
from src.ML_functions import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("BUCKET_NAME", "")
CLEAN_BUCKET = BUCKET_NAME.replace("gs://", "").replace("gs:/", "").strip("/")
BASE_DIR = Path(f"/gcs/{CLEAN_BUCKET}") if CLEAN_BUCKET else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
RESULTS_DIR = BASE_DIR / "ML_results"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

model_list = ['xgb']
region_list = ['HOA'] 
experiment_list = ['RUN_FINAL_20']
cluster_list = ['p', 'ap', 'other'] 
leads = [0, 1, 2, 3, 4, 8, 12]
aggregation = 'cluster'

def clean_scientific_brackets(df):
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace("[", "").replace("]", "").strip()) for c in df.columns]
    return df

def run_ml_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    input_master_raw = pd.read_csv(DATA_FOLDER / "input_master.csv", low_memory=False)
    input_master_raw = clean_scientific_brackets(input_master_raw)

    for experiment in experiment_list:
        for model_type in model_list:
            for region in region_list:
                for cluster in cluster_list:
                    scenario = f"{aggregation}_{experiment}_{region}_{cluster}"
                    wandb.init(project="drought_forecasting", name=scenario, reinit=True)
                    
                    df_run = input_master_raw.copy()
                    if region != 'HOA': df_run = df_run[df_run['country'] == region]
                    df_run = df_run[df_run['lhz'] == cluster]

                    preds_storage, shap_values_all = pd.DataFrame(), []

                    for lead in leads:
                        df = df_run[df_run["lead"] == lead].sort_index().copy()
                        if df.empty or len(df) < 5: continue
                        
                        target_col = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
                        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
                        X = df.drop(columns=[target_col, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0", "country"], errors="ignore")
                        X = X.select_dtypes(include=[np.number]).fillna(0.0)

                        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
                        
                        hp_path = HP_RESULT_ROOT / f"{aggregation}_{experiment}_{region}_{model_type}" / f"best_params_{model_type}_L{lead}.json"
                        params = load_best_params(hp_path, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.01})
                        
                        model = XGBRegressor(**params, random_state=42)
                        model.fit(train_X, train_y)

                        preds = model.predict(test_X)
                        p_data = pd.DataFrame({
                            'observed': test_y.values, 'prediction': preds, 'lead': lead, 
                            'county': df.loc[test_X.index, 'county'].values, 'cluster': cluster
                        }, index=test_X.index)
                        preds_storage = pd.concat([preds_storage, p_data])

                    preds_storage.to_excel(RESULTS_DIR / f"raw_model_output_{scenario}.xlsx")
                    wandb.finish()

if __name__ == "__main__":
    run_ml_pipeline()