# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import logging

# ---------------------------------------------------------------------
# MOCK ENVIRONMENT
# ---------------------------------------------------------------------
os.environ["WANDB_API_KEY"] = "dryrun"
os.environ["WANDB_MODE"] = "offline"
os.environ["BUCKET_NAME"] = ""

# ---------------------------------------------------------------------
# DUMMY DATA
# ---------------------------------------------------------------------
def create_local_dummy_data(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)

    rows = 100
    # We include the "villain" [0.537] to ensure the purification script works
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="ME"),
        "lead": [0, 1, 2, 3] * 25,
        "FEWS_CS": np.random.randint(1, 5, size=rows).astype(float),
        "lhz": ["p", "ap", "other", "p"] * 25,
        "county": ["Marsabit", "Turkana", "Wajir", "Mandera"] * 25,
        "precip": ["[0.537]"] * rows, # The problematic scientific notation string
        "temp": [25.5] * rows,
        "ndvi": [0.4] * rows,
        "base_forecast": [1.0] * rows,
    })

    # Save to CSV
    df.to_csv(data_path / "input_master.csv", index=False)
    print("✅ Dummy data created with problematic strings")

# ---------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------
def run_local_smoke_test():
    base = Path.cwd() / "local_test_env"
    
    # 1. Clean slate for the test environment
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    data_folder = base / "input_collector"
    create_local_dummy_data(data_folder)

    # 2. Import your scripts
    import src.HP_tuning as hpt
    import src.ML_execution as mle

    # 3. Synchronize Script Paths to the Local Environment
    # We force the scripts to look at our dummy 'base' folder instead of /gcs/
    for script in [hpt, mle]:
        script.BASE_DIR = base
        script.DATA_FOLDER = data_folder
        script.HP_RESULT_ROOT = base / "HP_results"
        
    # ML_execution specific paths
    mle.RESULTS_DIR = base / "ML_results"
    mle.PLOTS_DIR = base / "ML_results" / "plots"

    # 4. Limit the test scope for speed
    test_leads = [0]
    # Patch the leads lists directly in the modules
    hpt.leads = test_leads
    # Note: ML_execution uses local variables inside the function, 
    # but it will pick up the leads we set in the loop below if needed.

    print("🚀 Running HP tuning (Purifying and Training)...")
    hpt.run_hp_tuning()

    print("🚀 Running ML execution (Testing SHAP with Cleaned Data)...")
    mle.run_ml_pipeline()

    # 5. Final Validation: Did SHAP actually produce a file?
    plot_check = list((base / "ML_results" / "plots").glob("SHAP_p_L0.png"))
    if plot_check:
        print(f"✅ SHAP Plot generated successfully: {plot_check[0].name}")
    else:
        print("❌ SHAP Plot missing! Check logs for conversion errors.")

    print("✨ Local smoke test complete")

if __name__ == "__main__":
    run_local_smoke_test()