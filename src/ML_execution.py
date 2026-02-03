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
import shap
import wandb
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from src.ML_functions import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def nuclear_clean(df):
    """Standardized cleaning to ensure lowercase names and recovered metadata."""
    df = df.copy()
    # Clean headers and force lowercase
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace("[", "").replace("]", "").strip()).lower() for c in df.columns]
    
    # Map all possible variations to the names the execution loop expects
    rename_map = {
        "fews_cs": "fewscs", 
        "unnamed0": "date", 
        "unnamed_0": "date", 
        "time": "date",
        "datetime": "date"
    }
    df = df.rename(columns=rename_map)
    
    # If 'date' is still missing, treat the first column as date
    if 'date' not in df.columns and len(df.columns) > 0:
        if df.columns[0].startswith('unnamed'):
            df = df.rename(columns={df.columns[0]: 'date'})
    
    # Clean numeric data (handle bracketed strings like [1.2])
    meta = {"county", "lhz", "date", "lead", "country", "fewscs"}
    for col in df.columns:
        if col not in meta:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True), errors='coerce')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0).astype(np.float64)
    return df

def run_ml_pipeline():
    # --- Paths Configuration ---
    BUCKET_NAME = os.getenv("BUCKET_NAME", "").replace("gs://", "").strip("/")
    BASE_DIR = Path(f"/gcs/{BUCKET_NAME}") if BUCKET_NAME else Path(os.getcwd())
    DATA_FOLDER = BASE_DIR / "input_collector"
    RESULTS_DIR = BASE_DIR / "ML_results"
    HP_RESULT_ROOT = BASE_DIR / "HP_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = DATA_FOLDER / "input_master.csv"
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return

    df_master = nuclear_clean(pd.read_csv(input_path, low_memory=False))

    leads = [0, 1, 2, 3, 4, 8, 12]
    cluster_list = df_master['lhz'].unique()

    for cluster in cluster_list:
        # Match the scenario name used in the notebook
        scenario = f"cluster_RUN_FINAL_20_HOA_{cluster}"
        wandb.init(project="drought_forecasting", name=scenario, reinit=True)
        
        df_c = df_master[df_master['lhz'] == cluster].copy()
        preds_all, shap_v, shap_d, shap_b = [], [], [], []

        for lead in leads:
            df_l = df_c[df_c["lead"] == lead].sort_index().copy()
            if len(df_l) < 10: continue
            
            # 1. Isolate Metadata before dropping columns
            idx_split = int(len(df_l) * 0.8)
            meta_test = df_l.iloc[idx_split:].copy()
            
            # 2. Prepare Training Data (Lowercase references)
            target = "fewscs"
            drop_cols = ["fewscs", "lead", "county", "lhz", "date", "country", "base_forecast", "unnamed0"]
            
            X_df = df_l.drop(columns=[c for c in drop_cols if c in df_l.columns], errors="ignore").select_dtypes(include=[np.number])
            X_np = X_df.values.astype(np.float64)
            y_np = df_l[target].values.astype(np.float64)
            
            train_X, test_X, train_y, test_y = train_test_split(X_np, y_np, test_size=0.2, shuffle=False)
            
            # 3. Load Tuning Results
            hp_path = HP_RESULT_ROOT / "cluster_RUN_FINAL_20_HOA_xgb" / f"best_params_xgb_L{lead}.json"
            params = load_best_params(hp_path, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05})
            
            model = XGBRegressor(**params)
            model.fit(train_X, train_y)

            # 4. Predictions
            preds = model.predict(test_X)
            preds_all.append(pd.DataFrame({
                'date': meta_test['date'].values,
                'observed': test_y,
                'prediction': preds,
                'lead': lead,
                'county': meta_test['county'].values,
                'cluster': cluster
            }))

            # 5. Kernel SHAP for XAI (Unified File Names)
            try:
                bg = train_X[:10] if len(train_X) > 10 else train_X
                explainer = shap.KernelExplainer(model.predict, bg)
                sv = explainer.shap_values(test_X)
                if isinstance(sv, list): sv = sv[0]
                
                # Metadata for SHAP files
                m_df = meta_test[['county', 'date', 'lead', 'lhz']].reset_index(drop=True)
                
                shap_v.append(pd.concat([m_df, pd.DataFrame(sv, columns=X_df.columns)], axis=1))
                shap_d.append(pd.concat([m_df, pd.DataFrame(test_X, columns=X_df.columns)], axis=1))
                shap_b.append(pd.DataFrame({'base_value': [float(explainer.expected_value)], 'lead': [lead]}))
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")

        # --- Save Final Outputs with Notebook-Matched Names ---
        if preds_all:
            pd.concat(preds_all).set_index('date').to_excel(RESULTS_DIR / f"raw_model_output_{scenario}.xlsx")
            
            if shap_v:
                pd.concat(shap_v).to_excel(RESULTS_DIR / f"feature_importances_{scenario}.xlsx", index=False)
                pd.concat(shap_v).to_excel(RESULTS_DIR / f"shap_values_{scenario}.xlsx", index=False)
                pd.concat(shap_d).to_excel(RESULTS_DIR / f"shap_data_{scenario}.xlsx", index=False)
                pd.concat(shap_b).to_excel(RESULTS_DIR / f"shap_base_values_{scenario}.xlsx", index=False)
        
        wandb.finish()

if __name__ == "__main__":
    run_ml_pipeline()