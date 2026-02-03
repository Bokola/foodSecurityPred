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
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import wandb
from src.ML_functions import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PATHS ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
CLEAN_BUCKET = BUCKET_NAME.replace("gs://", "").replace("gs:/", "").strip("/")
BASE_DIR = Path(f"/gcs/{CLEAN_BUCKET}") if CLEAN_BUCKET else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
RESULTS_DIR = BASE_DIR / "ML_results"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

# --- CONFIG ---
leads = [0, 1, 2, 3, 4, 8, 12]
cluster_list = ['p', 'ap', 'other']
experiment = 'RUN_FINAL_20'
region = 'HOA'
aggregation = 'cluster'

def nuclear_clean(df):
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace("[", "").replace("]", "").strip()) for c in df.columns]
    meta = {"county", "lhz", "date", "lead", "country", "FEWSCS", "FEWS_CS", "Unnamed0"}
    for col in df.columns:
        if col not in meta:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True), errors='coerce')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float64)
    return df

def run_ml_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    input_path = DATA_FOLDER / "input_master.csv"
    if not input_path.exists(): return
    df_master = nuclear_clean(pd.read_csv(input_path, low_memory=False))

    for cluster in cluster_list:
        scenario = f"{aggregation}_{experiment}_{region}_{cluster}"
        logger.info(f"Processing: {scenario}")
        wandb.init(project="drought_forecasting", name=scenario, reinit=True)
        
        df_c = df_master[df_master['lhz'] == cluster].copy()
        preds_all, shap_v, shap_d, shap_b = [], [], [], []

        for lead in leads:
            df_l = df_c[df_c["lead"] == lead].sort_index().copy()
            if len(df_l) < 5: continue
            
            target = "FEWSCS" if "FEWSCS" in df_l.columns else "FEWS_CS"
            X_df = df_l.drop(columns=[target, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0", "country"], errors="ignore").select_dtypes(include=[np.number])
            
            X_np = X_df.values.astype(np.float64)
            y_np = df_l[target].values.astype(np.float64)
            train_X, test_X, train_y, test_y = train_test_split(X_np, y_np, test_size=0.2, shuffle=False)
            
            hp_path = HP_RESULT_ROOT / f"{aggregation}_{experiment}_{region}_xgb" / f"best_params_xgb_L{lead}.json"
            params = load_best_params(hp_path, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.01})
            
            model = XGBRegressor(**params, random_state=42)
            model.fit(train_X, train_y)

            # --- Predictions ---
            p = model.predict(test_X)
            idx = df_l.index[int(len(df_l)*0.8):]
            preds_all.append(pd.DataFrame({
                'date': df_l.loc[idx, 'date'].values,
                'observed': test_y, 'prediction': p, 'lead': lead,
                'county': df_l.loc[idx, 'county'].values, 'cluster': cluster
            }))

            # --- THE FINAL SHAP SOLUTION: KERNEL EXPLAINER ---
            try:
                # 1. Use raw slice for background (immune to kmeans/indexing errors)
                background = train_X[:10] if len(train_X) > 10 else train_X
                
                # 2. KernelExplainer is a black-box. It NEVER looks at XGBoost C++ strings.
                explainer = shap.KernelExplainer(model.predict, background)
                
                # 3. Calculate values (returns a list of arrays or a single array)
                sv = explainer.shap_values(test_X)
                
                # Handle multi-output return format just in case
                if isinstance(sv, list): sv = sv[0]
                
                meta = df_l.loc[idx, ['county', 'date', 'lead', 'lhz']].reset_index(drop=True)
                shap_v.append(pd.concat([meta, pd.DataFrame(sv, columns=X_df.columns)], axis=1))
                shap_d.append(pd.concat([meta, pd.DataFrame(test_X, columns=X_df.columns)], axis=1))
                
                # Base value is a simple scalar in KernelExplainer
                shap_b.append(pd.DataFrame({
                    'base_value': [float(explainer.expected_value)], 
                    'lead': [lead]
                }))
                
            except Exception as e:
                logger.error(f"SHAP failed at {cluster} L{lead}: {e}")

        # --- EXPORTS (Exactly as Plots Notebook expects) ---
        if preds_all:
            out_preds = pd.concat(preds_all).set_index('date')
            out_preds.to_excel(RESULTS_DIR / f"raw_model_output_{scenario}.xlsx")
            if shap_v:
                v_df = pd.concat(shap_v)
                v_df.to_excel(RESULTS_DIR / f"feature_importances_{scenario}.xlsx")
                v_df.to_excel(RESULTS_DIR / f"shap_values_{scenario}.xlsx")
                pd.concat(shap_d).to_excel(RESULTS_DIR / f"shap_data_{scenario}.xlsx")
                pd.concat(shap_b).to_excel(RESULTS_DIR / f"shap_base_values_{scenario}.xlsx")
        wandb.finish()

if __name__ == "__main__":
    run_ml_pipeline()