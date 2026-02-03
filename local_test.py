"""
local test environment - Updated for Nuclear Cleaning Verification
"""
import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Mock Environment
os.environ["WANDB_MODE"] = "offline"
os.environ["BUCKET_NAME"] = "" 

def create_mock_data(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    rows = 200
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=rows, freq="MS"),
        "lead": [0, 1] * 100,
        "FEWS_CS": np.random.randint(1, 5, size=rows),
        "country": ["Kenya"] * 200,
        "lhz": ["p"] * 200,
        "county": ["County_X"] * 200,
        "precip": ["[2.652E0]"] * rows, # The failure string
        "base_forecast": [1.0] * rows
    })
    df.to_csv(path / "input_master.csv", index=False)

def run_test():
    base = Path.cwd() / "local_test_env"
    if base.exists(): shutil.rmtree(base)
    base.mkdir()
    data_folder = base / "input_collector"
    create_mock_data(data_folder)

    import HP_tuning as hpt
    import ML_execution as mle

    for script in [hpt, mle]:
        script.BASE_DIR, script.DATA_FOLDER = base, data_folder
        script.HP_RESULT_ROOT = base / "HP_results"
        script.leads, script.cluster_list = [0, 1], ['p']

    mle.RESULTS_DIR = base / "ML_results"

    hpt.run_hp_tuning()
    mle.run_ml_pipeline()

    # Verification
    scenario = "cluster_RUN_FINAL_20_HOA_p"
    required = [f"raw_model_output_{scenario}.xlsx", f"feature_importances_{scenario}.xlsx", f"shap_values_{scenario}.xlsx"]
    for f in required:
        if (mle.RESULTS_DIR / f).exists():
            print(f"✅ Found: {f}")
            if "feature" in f or "shap_values" in f:
                df = pd.read_excel(mle.RESULTS_DIR / f, index_col=0)
                if 'county' in df.columns: print(f"   - Metadata 'county' verified in {f}")

if __name__ == "__main__":
    run_test()