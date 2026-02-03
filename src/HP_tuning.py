"""
Drought Forecasting: Hyperparameter Tuning
Optimized for: Vertex AI & Robustness against Metadata string errors.
Includes: Generalization scoring and full CV result exports.

Originally authored by Tim Busker and reorganized for vertex AI by Basil Okola.
"""
import os
import re
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from src.ML_functions import save_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PATHS ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
CLEAN_BUCKET = BUCKET_NAME.replace("gs://", "").replace("gs:/", "").strip("/")
BASE_DIR = Path(f"/gcs/{CLEAN_BUCKET}") if CLEAN_BUCKET else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

# --- CONFIG ---
leads = [0, 1, 2, 3, 4, 8, 12]
cluster_list = ['p', 'ap', 'other']
experiment = 'RUN_FINAL_20'
region = 'HOA'
aggregation = 'cluster'

def nuclear_clean(df):
    """
    Identical cleaning logic to ML_execution.py.
    Removes brackets from headers and values to ensure feature consistency.
    """
    df = df.copy()
    # 1. Clean headers: [precip] -> precip
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace("[", "").replace("]", "").strip()) for c in df.columns]
    
    meta = {"county", "lhz", "date", "lead", "country", "FEWSCS", "FEWS_CS", "Unnamed0"}
    for col in df.columns:
        if col not in meta:
            # 2. Clean values: [2.65E0] -> 2.65
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True), 
                errors='coerce'
            )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float64)
    return df

def run_hp_tuning():
    input_path = DATA_FOLDER / "input_master.csv"
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        return

    # Load and Clean
    df_all = nuclear_clean(pd.read_csv(input_path, low_memory=False))

    hp_folder = HP_RESULT_ROOT / f"{aggregation}_{experiment}_{region}_xgb"
    hp_folder.mkdir(parents=True, exist_ok=True)

    for cluster in cluster_list:
        df_c = df_all[df_all['lhz'] == cluster]
        
        for lead in leads:
            df = df_c[df_c["lead"] == lead].sort_index().copy()
            if len(df) < 10:
                continue

            target = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
            drop_cols = [target, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0", "country"]
            
            X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            X = X.select_dtypes(include=[np.number]).astype(np.float64)
            y = df[target].astype(np.float64)

            # --- PRE-EMPTIVE BASE_SCORE FIX ---
            # We calculate a clean float mean to prevent XGBoost from 
            # auto-generating the bracketed string [2.5E0] during CV.
            clean_mean = float(np.mean(y))
            
            model = XGBRegressor(base_score=clean_mean, random_state=42)
            
            # Grid Search
            param_grid = {
                "max_depth": [3, 4, 6],
                "learning_rate": [0.01, 0.05],
                "n_estimators": [200, 400]
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(
                model, 
                param_grid, 
                cv=tscv, 
                scoring="neg_mean_absolute_error", 
                n_jobs=-1
            )
            
            logger.info(f"Tuning {cluster} Lead {lead}...")
            grid.fit(X, y)
            
            # Save results
            save_best_params(grid.best_params_, hp_folder / f"best_params_xgb_L{lead}.json")

if __name__ == "__main__":
    run_hp_tuning()