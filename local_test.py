"""
local test environment
"""
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Mock Environment
os.environ["WANDB_MODE"] = "offline"
os.environ["BUCKET_NAME"] = "" 

def create_mock_data(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    rows = 180
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=rows, freq="MS"),
        "lead": [0, 1, 4] * 60,
        "FEWS_CS": np.random.randint(1, 5, size=rows),
        "country": ["Kenya", "Ethiopia", "Somalia"] * 60,
        "lhz": ["p", "ap", "other"] * 60,
        "county": ["County_A", "County_B", "County_C"] * 60,
        "precip": ["[0.123]"] * rows,
        "base_forecast": [1.0] * rows
    })
    df.to_csv(path / "input_master.csv")
    print("✅ Mock data with Kenya & Clusters created.")

def run_test():
    base = Path.cwd() / "local_test_env"
    if base.exists(): shutil.rmtree(base)
    base.mkdir()

    data_folder = base / "input_collector"
    create_mock_data(data_folder)

    import src.HP_tuning as hpt
    import src.ML_execution as mle

    # Patch paths for local run
    for script in [hpt, mle]:
        script.BASE_DIR = base
        script.DATA_FOLDER = data_folder
        script.HP_RESULT_ROOT = base / "test_HP_results"
        script.leads = [0, 1] # Fast test

    mle.RESULTS_DIR = base / "test_ML_results"
    mle.PLOTS_DIR = base / "test_ML_results" / "plots"

    print("STAGING: Tuning...")
    hpt.run_hp_tuning()

    print("STAGING: Executing...")
    mle.run_ml_pipeline()

    # Verify Cluster-Specific Outputs
    for cluster in ['p', 'ap', 'other']:
        file = base / "test_ML_results" / f"preds_RUN_FINAL_20_HOA_{cluster}_xgb.xlsx"
        if file.exists():
            print(f"✅ Success: Results for cluster '{cluster}' (Region: HOA) verified.")
        else:
            print(f"❌ Error: Results for cluster '{cluster}' missing.")

if __name__ == "__main__":
    run_test()