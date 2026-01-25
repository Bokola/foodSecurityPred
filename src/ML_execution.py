# -*- coding: utf-8 -*-
"""
Optimized for Google Vertex AI
Original Author: Tim Busker
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ✅ headless-safe
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap
import wandb

from src.ML_functions import load_best_params

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
# PATHS
# ---------------------------------------------------------------------
BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_DIR = Path(f"/gcs/{BUCKET_NAME.replace('gs://', '')}") if BUCKET_NAME else Path(os.getcwd())

DATA_FOLDER = BASE_DIR / "input_collector"
RESULTS_DIR = BASE_DIR / "ML_results"
PLOTS_DIR = RESULTS_DIR / "plots"
HP_RESULT_ROOT = BASE_DIR / "HP_results"

for folder in [RESULTS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# DESIGN
# ---------------------------------------------------------------------
model_list = ["xgb"]
region_list = ["HOA"]
aggregations = ["cluster"]
experiment_list = ["RUN_FINAL_20"]
leads = [0, 1, 2, 3, 4, 8, 12]

design_variables = [
    (e, m, a, r)
    for e in experiment_list
    for m in model_list
    for a in aggregations
    for r in region_list
]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def clean_scientific_brackets(df):
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
# MAIN PIPELINE
# ---------------------------------------------------------------------
def run_ml_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for experiment, model_type, aggregation, region in design_variables:
        logger.info(f"Starting Execution: {experiment} | {model_type}")

        wandb.init(
            project="drought_forecasting",
            name=f"Final_{experiment}_{model_type}",
            settings=wandb.Settings(start_method="thread"),
        )

        traintest_ratio = int(experiment[-2:]) / 100

        input_master = pd.read_csv(DATA_FOLDER / "input_master.csv")
        input_master.columns = input_master.columns.astype(str).str.strip()
        input_master.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

        if "date" in input_master.columns:
            input_master.index = pd.to_datetime(input_master["date"])
            input_master.drop(columns=["date"], inplace=True)

        input_master["lead"] = pd.to_numeric(
            input_master["lead"], errors="coerce"
        ).fillna(0).astype(int)

        cluster_list = ["p", "ap", "other"] if aggregation == "cluster" else ["no_cluster"]

        for cluster in cluster_list:
            eval_stats = []

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

                # ✅ NUMERIC-ONLY GUARANTEE
                X = X.select_dtypes(include=[np.number]).astype(float)

                train_X, test_X, train_y, test_y = train_test_split(
                    X, y, test_size=traintest_ratio, shuffle=False
                )

                hp_file = (
                    HP_RESULT_ROOT
                    / f"{aggregation}_{experiment}_{region}_{cluster}_{model_type}"
                    / f"best_params_{model_type}_L{lead}.json"
                )

                params = load_best_params(
                    hp_file,
                    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.01},
                )

                model = XGBRegressor(**params, random_state=42)
                model.fit(train_X, train_y)
                preds = model.predict(test_X)

                eval_stats.append(
                    {
                        "cluster": cluster,
                        "lead": lead,
                        "mae": mean_absolute_error(test_y, preds),
                        "rmse": np.sqrt(mean_squared_error(test_y, preds)),
                        "r2": r2_score(test_y, preds),
                    }
                )

                try:
                    explainer = shap.TreeExplainer(model)
                    # shap_vals = explainer.shap_values(train_X.sample(min(100, len(train_X))))
                    # shap.summary_plot(shap_vals, train_X, plot_type="bar", show=False)
                    shap_X = train_X.sample(min(100, len(train_X))).to_numpy(dtype=float)

                    shap_vals = explainer.shap_values(shap_X)

                    shap.summary_plot(
                        shap_vals,
                        features=shap_X,
                        feature_names=train_X.columns,
                        plot_type="bar",
                        show=False,
                    )

                    wandb.log({f"SHAP_{cluster}_L{lead}": wandb.Image(plt)})
                    plt.close()
                except Exception as e:
                    logger.warning(f"SHAP failed: {e}")

            pd.DataFrame(eval_stats).to_csv(
                RESULTS_DIR / f"eval_stats_{cluster}.csv", index=False
            )

        wandb.finish()


if __name__ == "__main__":
    run_ml_pipeline()
