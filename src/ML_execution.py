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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap
import wandb
from src.ML_functions import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PATHS (Vertex AI / GCS Compatible) ---
BUCKET_NAME = os.getenv("BUCKET_NAME", "")
CLEAN_BUCKET = BUCKET_NAME.replace("gs://", "").replace("gs:/", "").strip("/")
BASE_DIR = Path(f"/gcs/{CLEAN_BUCKET}") if CLEAN_BUCKET else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
RESULTS_DIR = BASE_DIR / "ML_results"
PLOTS_DIR = RESULTS_DIR / "plots"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

# --- DESIGN VARIABLES (The Full Matrix) ---
model_list = ['xgb']
region_list = ['HOA']  # Includes Kenya, Ethiopia, Somalia
experiment_list = ['RUN_FINAL_20']
# We need to run for each cluster type
cluster_list = ['p', 'ap', 'other'] 
leads = [0, 1, 2, 3, 4, 8, 12]

# Matrix including Experiment, Model, Region, and Cluster
design_variables = [
    (exp, model, reg, cluster) 
    for exp in experiment_list 
    for model in model_list 
    for reg in region_list
    for cluster in cluster_list
]

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

def run_ml_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    input_path = DATA_FOLDER / "input_master.csv"
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        return

    input_master_raw = pd.read_csv(input_path, low_memory=False)
    input_master_raw = clean_scientific_brackets(input_master_raw)

    for experiment, model_type, region, cluster in design_variables:
        # Initialize W&B run for this specific combination
        run_name = f"{experiment}_{region}_{cluster}_{model_type}"
        wandb.init(project="drought_forecasting", name=run_name, reinit=True)
        
        # Filtering by Region AND Cluster
        df_run = input_master_raw.copy()
        if region != 'HOA':
            df_run = df_run[df_run['country'] == region]
        df_run = df_run[df_run['lhz'] == cluster]

        preds_storage = pd.DataFrame()
        eval_stats = pd.DataFrame()

        for lead in leads:
            df = df_run[df_run["lead"] == lead].sort_index().copy()
            if df.empty or len(df) < 5: continue
            
            # One-hot encode country if multiple countries exist in this cluster
            if region == 'HOA':
                df = pd.get_dummies(df, columns=['country'], prefix='country', prefix_sep='_')

            target_col = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).astype(np.float64)
            X = df.drop(columns=[target_col, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0", "country"], errors="ignore")
            X = X.select_dtypes(include=[np.number]).astype(np.float64).fillna(0.0)

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # HP folder path matching the original nested naming convention
            hp_path = HP_RESULT_ROOT / f"cluster_{experiment}_{region}_{model_type}" / f"best_params_{model_type}_L{lead}.json"
            params = load_best_params(hp_path, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.01})
            
            model = XGBRegressor(**params, random_state=42, objective="reg:squarederror")
            model.fit(train_X, train_y)

            # Predictions & Logging
            preds = model.predict(test_X)
            mae, rmse = mean_absolute_error(test_y, preds), np.sqrt(mean_squared_error(test_y, preds))
            wandb.log({f"L{lead}/mae": mae, f"L{lead}/rmse": rmse})

            # Store results
            p_data = pd.DataFrame({
                'observed': test_y.values, 'prediction': preds, 
                'lead': lead, 'county': df.loc[test_X.index, 'county'].values,
                'cluster': cluster, 'region': region
            }, index=test_X.index)
            preds_storage = pd.concat([preds_storage, p_data])
            eval_stats = pd.concat([eval_stats, pd.DataFrame({'lead': [lead], 'mae': [mae], 'rmse': [rmse]})])

            # SHAP (Fidelity=100)
            try:
                bg = train_X.sample(min(len(train_X), 100))
                explainer = shap.KernelExplainer(model.predict, bg)
                shap_vals = explainer.shap_values(test_X.sample(min(len(test_X), 100)), nsamples=500, silent=True)
                
                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_vals, features=test_X.sample(min(len(test_X), 100)), show=False)
                plt.title(f"SHAP: {region}-{cluster} L{lead}")
                plot_path = PLOTS_DIR / f"SHAP_{region}_{cluster}_L{lead}.png"
                plt.savefig(plot_path, bbox_inches='tight')
                wandb.log({f"plots/SHAP_L{lead}": wandb.Image(str(plot_path))})
                plt.close(fig)
            except Exception as e:
                logger.error(f"SHAP failed for {cluster} L{lead}: {e}")

        # Save Excel files with distinct naming
        file_suffix = f"{experiment}_{region}_{cluster}_{model_type}"
        preds_storage.to_excel(RESULTS_DIR / f"preds_{file_suffix}.xlsx")
        eval_stats.to_excel(RESULTS_DIR / f"stats_{file_suffix}.xlsx")
        
        # Log to W&B
        artifact = wandb.Artifact(name=f"outputs_{region}_{cluster}", type="results")
        artifact.add_file(str(RESULTS_DIR / f"preds_{file_suffix}.xlsx"))
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    run_ml_pipeline()