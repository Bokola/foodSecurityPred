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

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# PATHS (Vertex / Local compatible)
# ---------------------------------------------------------------------
BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_DIR = Path(f"/gcs/{BUCKET_NAME.replace('gs://', '')}") if BUCKET_NAME else Path(os.getcwd())

DATA_FOLDER = BASE_DIR / "input_collector"
HP_RESULT_ROOT = BASE_DIR / "HP_results"
HP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# DESIGN VARIABLES
# ---------------------------------------------------------------------
model_list = ["xgb"]
aggregations = ["cluster"]
leads = [0, 1, 2, 3, 4, 8, 12]
experiment_list = ["RUN_FINAL_20"]
region_list = ["HOA"]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def clean_scientific_brackets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove brackets and coerce numeric columns safely."""
    cols_to_skip = {"county", "lhz", "base_forecast", "FEWS_CS", "date"}

    for col in df.columns:
        if col in cols_to_skip:
            continue

        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str)
            if not sample.empty and sample.iloc[0].count("-") == 2:
                continue

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\[\]]", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
def run_hp_tuning():
    input_path = DATA_FOLDER / "input_master.csv"
    input_master = pd.read_csv(input_path)

    # --- HARD GUARDS ---
    input_master.columns = input_master.columns.astype(str).str.strip()
    input_master.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

    if "date" in input_master.columns:
        input_master.index = pd.to_datetime(input_master["date"])
        input_master.drop(columns=["date"], inplace=True)

    input_master["lead"] = pd.to_numeric(
        input_master["lead"], errors="coerce"
    ).fillna(0).astype(int)

    for experiment in experiment_list:
        for model_type in model_list:
            for aggregation in aggregations:
                for region in region_list:

                    cluster_list = ["p", "ap", "other"] if aggregation == "cluster" else ["no_cluster"]

                    for cluster in cluster_list:
                        hp_folder = HP_RESULT_ROOT / f"{aggregation}_{experiment}_{region}_{cluster}_{model_type}"
                        hp_folder.mkdir(parents=True, exist_ok=True)

                        df_cluster = (
                            input_master[input_master["lhz"] == cluster]
                            if aggregation == "cluster"
                            else input_master.copy()
                        )

                        for lead in leads:
                            df = df_cluster[df_cluster["lead"] == lead].sort_index().copy()
                            if df.empty:
                                continue

                            df = clean_scientific_brackets(df)
                            df = df.dropna(subset=["FEWS_CS"])
                            if df.empty:
                                continue

                            y = df["FEWS_CS"]
                            X = df.drop(
                                ["lead", "base_forecast", "FEWS_CS", "county", "lhz"],
                                axis=1,
                                errors="ignore",
                            )

                            # ✅ NUMERIC-ONLY FEATURES (CRITICAL FIX)
                            X = X.select_dtypes(include=[np.number]).astype(float)

                            param_grid = (
                                {
                                    "learning_rate": [0.01, 0.1],
                                    "max_depth": [4, 6],
                                    "n_estimators": [200, 400],
                                }
                                if model_type == "xgb"
                                else {"n_estimators": [200], "max_depth": [4, 10]}
                            )

                            model = (
                                XGBRegressor(random_state=42)
                                if model_type == "xgb"
                                else RandomForestRegressor(random_state=42)
                            )

                            # ❗ n_jobs=1 prevents Vertex crashes
                            search = GridSearchCV(
                                model,
                                param_grid,
                                cv=TimeSeriesSplit(n_splits=3),
                                n_jobs=1,
                            )

                            search.fit(X, y)
                            save_best_params(
                                search.best_params_,
                                hp_folder / f"best_params_{model_type}_L{lead}.json",
                            )

                            logger.info(f"Saved HP | {cluster} | Lead {lead}")


if __name__ == "__main__":
    run_hp_tuning()
