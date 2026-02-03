"""
local test environment
"""
# -*- coding: utf-8 -*-
"""
local test environment: Robust pathing and scenario-based naming compatibility
"""
import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# --- ROBUST PATHING ---
# Adds current directory and 'src' directory to the python path
current_dir = os.getcwd()
sys.path.append(current_dir)
if os.path.exists(os.path.join(current_dir, 'src')):
    sys.path.append(os.path.join(current_dir, 'src'))

# Mock Environment
os.environ["WANDB_MODE"] = "offline"
os.environ["BUCKET_NAME"] = "" 

def create_mock_data(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    rows = 180
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=rows, freq="MS"),
        "lead": [0, 1] * 90,
        "FEWS_CS": np.random.randint(1, 5, size=rows),
        "country": ["Kenya", "Ethiopia"] * 90,
        "lhz": ["p", "ap"] * 90,
        "county": ["County_A", "County_B"] * 90,
        "precip": ["[0.123]"] * rows,
        "base_forecast": [1.0] * rows,
        "observed_previous_IPC": np.random.randint(1, 5, size=rows),
        "observed_previous_year_IPC": np.random.randint(1, 5, size=rows),
        "FEWS_prediction": np.random.randint(1, 5, size=rows)
    })
    df.to_csv(path / "input_master.csv", index=False)
    print("✅ Mock data created.")

def run_test():
    base = Path.cwd() / "local_test_env"
    if base.exists(): shutil.rmtree(base)
    base.mkdir()

    data_folder = base / "input_collector"
    create_mock_data(data_folder)

    # --- DYNAMIC IMPORT ---
    try:
        import HP_tuning as hpt
        import ML_execution as mle
        print("✅ Modules imported from root.")
    except ImportError:
        try:
            from src import HP_tuning as hpt
            from src import ML_execution as mle
            print("✅ Modules imported from 'src' folder.")
        except ImportError as e:
            print(f"❌ Critical Error: Could not find HP_tuning or ML_execution. {e}")
            return

    # Patch paths for local run
    for script in [hpt, mle]:
        script.BASE_DIR = base
        script.DATA_FOLDER = data_folder
        script.HP_RESULT_ROOT = base / "test_HP_results"
        script.leads = [0, 1] 
        script.aggregation = 'cluster'
        script.cluster_list = ['p', 'ap'] # Simplified for test

    mle.RESULTS_DIR = base / "test_ML_results"
    mle.PLOTS_DIR = base / "test_ML_results" / "plots"

    print("STAGING: Tuning...")
    hpt.run_hp_tuning()

    print("STAGING: Executing...")
    mle.run_ml_pipeline()

    # --- VERIFICATION ---
    experiment, region, aggregation = "RUN_FINAL_20", "HOA", "cluster"
    print("\n--- VERIFICATION ---")
    for cluster in ['p', 'ap']:
        scenario = f"{aggregation}_{experiment}_{region}_{cluster}"
        file = mle.RESULTS_DIR / f"raw_model_output_{scenario}.xlsx"
        if file.exists():
            print(f"✅ Success: Verified {file.name}")
        else:
            print(f"❌ Error: Missing {file.name}")

if __name__ == "__main__":
    run_test()