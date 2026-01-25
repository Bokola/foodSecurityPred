# -*- coding: utf-8 -*-
"""
Optimized for Google Vertex AI Hyperparameter Tuning
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from src.ML_functions import save_best_params

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# PATHS
BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_DIR = Path(f"/gcs/{BUCKET_NAME.replace('gs://', '')}") if BUCKET_NAME else Path(os.getcwd())
DATA_FOLDER = BASE_DIR / "input_collector"
HP_RESULT_ROOT = BASE_DIR / "HP_results"
HP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)

def clean_scientific_brackets(df: pd.DataFrame) -> pd.DataFrame:
    """Safely cleans scientific and bracketed string notation."""
    cols_to_skip = {"county", "lhz", "base_forecast", "FEWS_CS", "date", "Unnamed: 0"}
    for col in df.columns:
        if col in cols_to_skip: continue
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str)
            if not sample.empty and sample.iloc[0].count("-") == 2: continue
            df[col] = df[col].astype(str).str.replace(r"[\[\]\s]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def run_hp_tuning():
    input_path = DATA_FOLDER / "input_master.csv"
    input_master = pd.read_csv(input_path)
    input_master.columns = input_master.columns.astype(str).str.strip()
    input_master.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

    if "date" in input_master.columns:
        input_master.index = pd.to_datetime(input_master["date"])
        input_master.drop(columns=["date"], inplace=True)

    input_master["lead"] = pd.to_numeric(input_master["lead"], errors="coerce").fillna(0).astype(int)

    leads = [0, 1, 2, 3, 4, 8, 12]
    
    for cluster in ["p", "ap", "other"]:
        hp_folder = HP_RESULT_ROOT / f"cluster_RUN_FINAL_20_HOA_{cluster}_xgb"
        hp_folder.mkdir(parents=True, exist_ok=True)
        df_cluster = input_master[input_master["lhz"] == cluster]

        for lead in leads:
            df = df_cluster[df_cluster["lead"] == lead].sort_index().copy()
            if df.empty: continue

            df = clean_scientific_brackets(df).dropna(subset=["FEWS_CS"])
            if df.empty: continue

            y = df["FEWS_CS"]
            X = df.drop(["lead", "base_forecast", "FEWS_CS", "county", "lhz"], axis=1, errors="ignore")
            
            # ✅ NUMERIC-ONLY FIX (Prevents GridSearchCV conversion errors)
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.select_dtypes(include=[np.number]).astype(float).fillna(0)

            search = GridSearchCV(
                XGBRegressor(random_state=42),
                {"learning_rate": [0.01, 0.1], "max_depth": [4, 6], "n_estimators": [200, 400]},
                cv=TimeSeriesSplit(n_splits=3), n_jobs=1, scoring="neg_mean_squared_error"
            )

            logger.info(f"Tuning {cluster} Lead {lead}...")
            search.fit(X, y)
            save_best_params(search.best_params_, hp_folder / f"best_params_xgb_L{lead}.json")

if __name__ == "__main__":
    run_hp_tuning()