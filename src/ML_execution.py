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
# BASE_DIR = Path(f"/gcs/{BUCKET_NAME}") if BUCKET_NAME else Path(os.getcwd())
# Strip "gs://" or "gs:/" if it exists in the environment variable
CLEAN_BUCKET = BUCKET_NAME.replace("gs://", "").replace("gs:/", "").strip("/")

# Construct the mount path correctly
if CLEAN_BUCKET:
    BASE_DIR = Path(f"/gcs/{CLEAN_BUCKET}")
else:
    BASE_DIR = Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
RESULTS_DIR = BASE_DIR / "ML_results"
PLOTS_DIR = RESULTS_DIR / "plots"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

def clean_scientific_brackets(df):
    """Purifies headers and values from scientific bracket metadata."""
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
    
    input_path = DATA_FOLDER / "input_master.csv"
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        return

    input_master = pd.read_csv(input_path, low_memory=False)
    input_master = clean_scientific_brackets(input_master)

    for cluster in ["p", "ap", "other"]:
        # Initialize a new W&B run for each cluster
        run = wandb.init(
            project="drought_forecasting", 
            name=f"Final_xgb_{cluster}", 
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )
        
        preds_storage = pd.DataFrame()
        features_df_full = pd.DataFrame()
        eval_stats = pd.DataFrame()
        shap_values_master = pd.DataFrame()
        shap_data_master = pd.DataFrame()
        shap_base_master = pd.DataFrame()

        df_cluster = input_master[input_master["lhz"] == cluster]
        
        for lead in [0, 1, 2, 3, 4, 8, 12]:
            df = df_cluster[df_cluster["lead"] == lead].sort_index().copy()
            if df.empty or len(df) < 5: 
                continue
            
            target_col = "FEWSCS" if "FEWSCS" in df.columns else "FEWS_CS"
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).astype(np.float64)
            X = df.drop(columns=[target_col, "lead", "base_forecast", "county", "lhz", "date", "Unnamed0"], errors="ignore")
            X = X.select_dtypes(include=[np.number]).astype(np.float64).fillna(0.0)

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            hp_file = HP_RESULT_ROOT / f"cluster_RUN_FINAL_20_HOA_{cluster}_xgb" / f"best_params_xgb_L{lead}.json"
            params = load_best_params(hp_file, {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.01})
            
            model = XGBRegressor(**params, random_state=42, objective="reg:squarederror", base_score=0.5)
            model.fit(train_X, train_y)

            # --- 1. Predictions & Metrics ---
            preds = model.predict(test_X)
            mae = mean_absolute_error(test_y, preds)
            rmse = np.sqrt(mean_squared_error(test_y, preds)) # version-safe
            r2 = r2_score(test_y, preds)

            # Log metrics to W&B
            wandb.log({f"L{lead}/mae": mae, f"L{lead}/rmse": rmse, f"L{lead}/r2": r2})

            p_data = pd.DataFrame({
                'observed': test_y.values, 
                'prediction': preds, 
                'base1_preds': df.loc[test_X.index, 'base_forecast'].values,
                'lead': lead, 
                'county': df.loc[test_X.index, 'county'].values
            }, index=test_X.index)
            preds_storage = pd.concat([preds_storage, p_data])

            eval_stats = pd.concat([eval_stats, pd.DataFrame({'lead': [lead], 'mae': [mae], 'rmse': [rmse], 'r2': [r2]})])

            # --- 2. High-Fidelity SHAP ---
            try:
                # Adaptive sampling: Fidelity=100, Density=200
                bg_size = min(len(train_X), 100)
                eval_size = min(len(test_X), 200)
                
                background = train_X.sample(bg_size)
                evaluation = test_X.sample(eval_size)
                
                explainer = shap.KernelExplainer(model.predict, background)
                # Use .shap_values() for better argument support across versions
                shap_vals = explainer.shap_values(evaluation, nsamples=500, silent=True)

                # Store components for Excel
                sv = pd.DataFrame(shap_vals, columns=X.columns); sv['lead'] = lead
                shap_values_master = pd.concat([shap_values_master, sv])

                sd = pd.DataFrame(evaluation.values, columns=X.columns); sd['lead'] = lead
                shap_data_master = pd.concat([shap_data_master, sd])
                
                bv = pd.DataFrame({'base_values': [explainer.expected_value] * eval_size}); bv['lead'] = lead
                shap_base_master = pd.concat([shap_base_master, bv])

                # Save plot locally and log to W&B Media
                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_vals, features=evaluation, feature_names=list(X.columns), show=False)
                plt.title(f"SHAP High Fidelity: {cluster} L{lead}")
                plot_path = PLOTS_DIR / f"SHAP_{cluster}_L{lead}.png"
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                wandb.log({f"plots/SHAP_L{lead}": wandb.Image(str(plot_path))})
                plt.close(fig)
                
                logger.info(f"✅ SHAP SUCCESS: {cluster} L{lead}")

            except Exception as e:
                logger.error(f"❌ SHAP FAILED for lead {lead}: {e}")

        # --- 3. Save Results and Upload Artifacts ---
        # Define local paths
        path_preds = RESULTS_DIR / f'raw_model_output_cluster_{cluster}.xlsx'
        path_stats = RESULTS_DIR / f'verif_unit_level_cluster_{cluster}.xlsx'
        path_shap = RESULTS_DIR / f'shap_values_cluster_{cluster}.xlsx'

        # Export to Excel
        preds_storage.to_excel(path_preds)
        eval_stats.to_excel(path_stats)
        shap_values_master.to_excel(path_shap)
        shap_data_master.to_excel(RESULTS_DIR / f'shap_data_cluster_{cluster}.xlsx')
        shap_base_master.to_excel(RESULTS_DIR / f'shap_base_values_cluster_{cluster}.xlsx')

        # Create W&B Artifact (File upload)
        artifact = wandb.Artifact(name=f"ml_outputs_{cluster}", type="dataset")
        artifact.add_file(str(path_preds))
        artifact.add_file(str(path_stats))
        artifact.add_file(str(path_shap))
        
        # Add cluster plots to artifact
        for p in PLOTS_DIR.glob(f"SHAP_{cluster}_L*.png"):
            artifact.add_file(str(p))

        wandb.log_artifact(artifact)
        wandb.finish()
        logger.info(f"✨ Finished Cluster {cluster}. Artifacts uploaded to W&B.")

if __name__ == "__main__":
    run_ml_pipeline()