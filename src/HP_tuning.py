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

# --- DESIGN VARIABLES (Notebook Compatible) ---
model_list = ['xgb']
region_list = ['HOA'] 
cluster_list = ['p', 'ap', 'other'] 
experiment_list = ['RUN_FINAL_20']
leads = [0, 1, 2, 3, 4, 8, 12]
aggregation = 'cluster' 

def clean_scientific_brackets(df):
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace("[", "").replace("]", "").strip()) for c in df.columns]
    cols_to_skip = {"county", "lhz", "base_forecast", "FEWS_CS", "FEWSCS", "date", "Unnamed0", "lead", "country"}
    for col in df.columns:
        if col in cols_to_skip: continue
        cleaned = df[col].astype(str).str.replace(r'[^0-9.eE\-]', '', regex=True)
        df[col] = pd.to_numeric(cleaned, errors="coerce")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float64)
    return df

def run_hp_tuning():
    input_path = DATA_FOLDER / "input_master.csv"
    if not input_path.exists():
        logger.error(f"Input not found at {input_path}")
        return
    input_master_raw = pd.read_csv(input_path, low_memory=False)
    input_master_raw = clean_scientific_brackets(input_master_raw)

    for experiment in experiment_list:
        for model_type in model_list:
            for region in region_list:
                # Directory structure matching the scenario logic
                hp_folder_root = HP_RESULT_ROOT / f"{aggregation}_{experiment}_{region}_{model_type}"
                hp_folder_root.mkdir(parents=True, exist_ok=True)
                
                for cluster in cluster_list:
                    df_cluster = input_master_raw.copy()
                    if region != 'HOA':
                        df_cluster = df_cluster[df_cluster['country'] == region]
                    df_cluster = df_cluster[df_cluster['lhz'] == cluster]

                    for lead in leads:
                        df = df_cluster[df_cluster["lead"] == lead].sort_index().copy()
                        if df.empty or len(df) < 10: continue

                        if region == 'HOA':
                            df = pd.get_dummies(df, columns=['country'], prefix='country')

                        target_col = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
                        y = df[target_col].fillna(0.0).astype(np.float64)
                        X = df.drop(columns=[target_col, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0", "country"], errors="ignore")
                        X = X.select_dtypes(include=[np.number]).astype(np.float64)

                        tscv = TimeSeriesSplit(n_splits=3)
                        param_grid = {"max_depth": [3, 4, 6], "learning_rate": [0.01, 0.05], "n_estimators": [200, 400]}
                        
                        grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
                        logger.info(f"🚀 Tuning: {region} | {cluster} | Lead {lead}")
                        grid.fit(X, y)

                        save_best_params(grid.best_params_, hp_folder_root / f"best_params_{model_type}_L{lead}.json")

if __name__ == "__main__":
    run_hp_tuning()