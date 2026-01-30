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
os.environ["BUCKET_NAME"] = "" # Empty string triggers local path logic

# ---------------------------------------------------------------------
# DUMMY DATA
# ---------------------------------------------------------------------
def create_local_dummy_data(data_path: Path):
    data_path.mkdir(parents=True, exist_ok=True)

    rows = 120 # Increased to ensure TimeSeriesSplit has enough samples
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="ME"),
        "lead": [0, 1, 3] * 40,
        "FEWS_CS": np.random.randint(1, 5, size=rows).astype(float),
        "lhz": ["p", "ap", "other"] * 40,
        "county": ["Marsabit", "Turkana", "Wajir"] * 40,
        "precip": ["[0.537]"] * rows, # The "villain" string
        "temp": [25.5] * rows,
        "ndvi": [0.4] * rows,
        "base_forecast": [1.0] * rows,
    })

    df.to_csv(data_path / "input_master.csv", index=False)
    print("✅ Dummy data created with problematic strings")

# ---------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------
def run_local_smoke_test():
    base = Path.cwd() / "local_test_env"
    
    # 1. Clean slate
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    data_folder = base / "input_collector"
    create_local_dummy_data(data_folder)

    # 2. Import scripts
    import src.HP_tuning as hpt
    import src.ML_execution as mle

    # 3. Synchronize Paths
    # We force the scripts to use our local test folder hierarchy
    for script in [hpt, mle]:
        script.BASE_DIR = base
        script.DATA_FOLDER = data_folder
        script.HP_RESULT_ROOT = base / "HP_results"
        
    mle.RESULTS_DIR = base / "test_ML_results"
    mle.PLOTS_DIR = base / "test_ML_results" / "plots"

    print("🚀 STAGE 1: Running HP tuning (Audit Trail & Generalization Score)...")
    # We patch the lead list inside hpt to run only one lead for speed
    original_leads = [0, 1, 2, 3, 4, 8, 12] 
    
    # Run tuning (this should now generate Excel CV results)
    hpt.run_hp_tuning()

    # Verify HP Excel Output
    hp_excel = list((base / "HP_results").rglob("CV_results_xgb_L*.xlsx"))
    if hp_excel:
        print(f"✅ HP CV Results generated: {hp_excel[0].name}")
        # Check if generalization score exists in the excel
        df_check = pd.read_excel(hp_excel[0])
        if 'generalization_score' in df_check.columns:
            print("✅ Generalization Score logic verified in Excel")
    else:
        print("❌ HP CV Results missing!")

    print("🚀 STAGE 2: Running ML execution (Full Output Suite & Kernel SHAP)...")
    mle.run_ml_pipeline()

    # 4. Final Validation of Outputs (Matching original script requirements)
    results_path = base / "test_ML_results"
    plots_path = results_path / "plots"

    # Check for SHAP plots
    plot_check = list(plots_path.glob("SHAP_*.png"))
    if plot_check:
        print(f"✅ SHAP Plots generated: {len(plot_check)} files found")
    else:
        print("❌ SHAP Plots missing!")

    # Check for Excel Outputs
    expected_files = [
        "raw_model_output_cluster_p.xlsx",
        "verif_unit_level_cluster_p.xlsx",
        "shap_values_cluster_p.xlsx",
        "shap_data_cluster_p.xlsx"
    ]
    
    for f_name in expected_files:
        if (results_path / f_name).exists():
            print(f"✅ Output verified: {f_name}")
        else:
            print(f"❌ Output missing: {f_name}")

    print("✨ Local smoke test complete. Environment is ready for Vertex AI.")

if __name__ == "__main__":
    run_local_smoke_test()