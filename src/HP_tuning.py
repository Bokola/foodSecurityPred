# -*- coding: utf-8 -*-
"""
Drought Forecasting: Hyperparameter Tuning
Optimized for: Vertex AI & Robustness against Metadata string errors.
Includes: Generalization scoring and full CV result exports.
"""

import os
import re
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from src.ML_functions import save_best_params

# Logging configuration for cloud monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dynamic Path Resolution
BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_DIR = Path(f"/gcs/{BUCKET_NAME.replace('gs://', '')}") if BUCKET_NAME else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
HP_RESULT_ROOT = BASE_DIR / "HP_results"
HP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)

def clean_scientific_brackets(df):
    """
    Purifies DataFrames by removing bracketed notation from headers and values.
    Prevents XGBoost from inheriting string metadata like '[5E-1]'.
    """
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace("[", "").replace("]", "").strip()) for c in df.columns]
    
    cols_to_skip = {"county", "lhz", "base_forecast", "FEWS_CS", "FEWSCS", "date", "Unnamed0", "lead"}
    for col in df.columns:
        if col in cols_to_skip: continue
        cleaned = df[col].astype(str).str.replace(r'[^0-9.eE\-]', '', regex=True)
        df[col] = pd.to_numeric(cleaned, errors="coerce")
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float64)
    return df

def run_hp_tuning():
    input_path = DATA_FOLDER / "input_master.csv"
    input_master = pd.read_csv(input_path, low_memory=False)
    input_master = clean_scientific_brackets(input_master)
    
    # Tuning logic per Climate Cluster
    for cluster in ["p", "ap", "other"]:
        hp_folder = HP_RESULT_ROOT / f"cluster_RUN_FINAL_20_HOA_{cluster}_xgb"
        hp_folder.mkdir(parents=True, exist_ok=True)
        df_cluster = input_master[input_master["lhz"] == cluster]

        for lead in [0, 1, 2, 3, 4, 8, 12]:
            df = df_cluster[df_cluster["lead"] == lead].sort_index().copy()
            if df.empty: continue

            target_col = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).astype(np.float64)
            X = df.drop(columns=[target_col, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0"], errors="ignore")
            X = X.select_dtypes(include=[np.number]).astype(np.float64).fillna(0)

            # Define Parameter Grid (matching your original XGB setup)
            param_grid = {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 6],
                "n_estimators": [200, 400],
                "subsample": [0.8, 1.0]
            }

            # TimeSeriesSplit is crucial for drought forecasting to prevent temporal leakage
            tscv = TimeSeriesSplit(n_splits=3)

            grid_xgb = GridSearchCV(
                estimator=XGBRegressor(
                    random_state=42, 
                    objective="reg:squarederror", 
                    base_score=0.5, # Critical fix for the [5E-1] bug
                    tree_method="hist"
                ),
                param_grid=param_grid,
                cv=tscv,
                n_jobs=-1, # Use all available cores
                scoring="neg_mean_squared_error",
                return_train_score=True 
            )

            logger.info(f"--- Tuning XGBoost for Cluster: {cluster}, Lead: {lead} ---")
            grid_xgb.fit(X, y)

            # --- PROCESS & SAVE RESULTS (Incorporating oldHP_tuning.py logic) ---
            cv_results_df = pd.DataFrame(grid_xgb.cv_results_)
            
            # Recreate the 'Generalization Score' from your original script
            # It identifies the gap between training and testing performance
            cv_results_df['generalization_score'] = cv_results_df['mean_train_score'] - cv_results_df['mean_test_score']
            cv_results_df['gen_score_rank'] = cv_results_df['generalization_score'].rank(ascending=True)
            
            # Sort by rank_test_score as per original script
            cv_results_df = cv_results_df.sort_values(by='rank_test_score')

            # Save full CV results to Excel (Original HPC requirement)
            cv_filename = hp_folder / f"CV_results_xgb_L{lead}.xlsx"
            cv_results_df.to_excel(cv_filename, index=False)

            # Save best params to JSON for ML_execution.py
            best_params = grid_xgb.best_params_
            save_best_params(best_params, hp_folder / f"best_params_xgb_L{lead}.json")
            
            logger.info(f"✅ HP tuning done for cluster {cluster} lead {lead}. Best MSE: {-grid_xgb.best_score_:.4f}")

if __name__ == "__main__":
    run_hp_tuning()