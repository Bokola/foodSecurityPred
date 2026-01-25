# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import wandb
from src.ML_functions import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_DIR = Path(f"/gcs/{BUCKET_NAME.replace('gs://', '')}") if BUCKET_NAME else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
RESULTS_DIR = BASE_DIR / "ML_results"
PLOTS_DIR = RESULTS_DIR / "plots"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

def clean_scientific_brackets(df):
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

def run_ml_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    input_master = pd.read_csv(DATA_FOLDER / "input_master.csv", low_memory=False)
    input_master = clean_scientific_brackets(input_master)

    for cluster in ["p", "ap", "other"]:
        wandb.init(project="drought_forecasting", name=f"Final_xgb_{cluster}")
        df_cluster = input_master[input_master["lhz"] == cluster]
        
        for lead in [0, 1, 2, 3, 4, 8, 12]:
            df = df_cluster[df_cluster["lead"] == lead].sort_index().copy()
            if df.empty: continue
            
            target_col = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).astype(np.float64)
            X = df.drop(columns=[target_col, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0"], errors="ignore")
            X = X.select_dtypes(include=[np.number]).astype(np.float64).fillna(0.0)

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            hp_file = HP_RESULT_ROOT / f"cluster_RUN_FINAL_20_HOA_{cluster}_xgb" / f"best_params_xgb_L{lead}.json"
            params = load_best_params(hp_file, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.01})
            
            # Use explicit numeric parameters for safety
            model = XGBRegressor(**params, random_state=42, objective="reg:squarederror", base_score=0.5)
            model.fit(train_X, train_y)

            # --- SHAP BLOCK: THE OPTIMIZED KERNEL POWERHOUSE ---
            try:
                plt.close('all')
                logger.info(f"Computing Optimized Kernel SHAP for {cluster} L{lead}...")
                
                # 1. Background set: 40 samples is sufficient for convergence in most tabular tasks
                background = train_X.sample(min(40, len(train_X)))
                
                # 2. Evaluation set: 50 samples creates a dense, readable summary plot
                evaluation = train_X.sample(min(50, len(train_X)))
                
                # 3. KernelExplainer ignores XGBoost metadata and uses the .predict() method directly
                explainer = shap.KernelExplainer(model.predict, background)
                
                # 4. nsamples controls the accuracy. 100 is a good balance for speed
                shap_vals = explainer.shap_values(evaluation, nsamples=100, silent=True)

                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_vals, 
                    features=evaluation, 
                    feature_names=list(train_X.columns), 
                    show=False,
                    plot_type="dot"
                )
                plt.title(f"Feature Importance (SHAP): {cluster} - Lead {lead}")
                plt.savefig(PLOTS_DIR / f"SHAP_{cluster}_L{lead}.png", bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                logger.info(f"✅ KERNEL SHAP SUCCESS for {cluster} L{lead}")

            except Exception as e:
                logger.error(f"❌ SHAP FAILED for {cluster} L{lead}: {e}")

        wandb.finish()

if __name__ == "__main__":
    run_ml_pipeline()