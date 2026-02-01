# 🌍 Predictive food-insecurity Intelligence System


[![Vertex AI](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EE4C2C?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)

An automated, cloud-native machine learning pipeline for regional drought forecasting. This system leverages **Gradient Boosted Trees** and **Automated MLOps** to predict drought severity (FEWS NET Classification) across specific regions with lead times ranging from 0 to 12 months. The model was originally written by Tim Busker to run on a hosted HPC cluster. I re-wrote it to run on vertex AI.

---

## 🛠️ Tech Stack & Ecosystem

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | **Vertex AI Pipelines (KFP)** | Manages sequential execution of HP tuning and training. |
| **Compute** | **Google Artifact Registry** | Stores versioned Docker images for reproducible environments. |
| **Intelligence** | **XGBoost / Random Forest** | Core regressor models for multi-lead time forecasting. |
| **Observability** | **Weights & Biases (W&B)** | Live experiment tracking and SHAP value visualization. |
| **Automation** | **GitHub Actions (OIDC)** | "Push-to-Deploy" CI/CD with Workload Identity Federation. |
| **Data Engine** | **GCS / Pathlib** | Cloud-native storage for master datasets and results. |

---

## 🏗️ Model Architecture

The system employs a **Modular Aggregation Strategy** to capture spatial nuances while maintaining computational efficiency.

### 1. Multi-Lead Forecasting
Separate models are trained for each lead time $L \in \{0, 1, 2, 3, 4, 8, 12\}$. This accounts for the diminishing signal-to-noise ratio as the forecast horizon increases.



### 2. Clustering Logic
To handle regional heterogeneity, we use a **Clustered Regression** approach:
* **Pastoral (P)**: Focuses on vegetation and rainfall indices.
* **Agro-Pastoral (AP)**: Incorporates soil moisture and crop-specific metrics.
* **Other**: Generic land-use zones.

### 3. Hyperparameter Bridge
A custom **JSON-based Handover** mechanism connects the Tuning phase to the Execution phase:
* **Step A (Tuning)**: Performs `TimeSeriesSplit` cross-validation to find optimal parameters.
* **Step B (Execution)**: Dynamically loads these parameters from GCS to train the production model.

---

## 🚀 Deployment Pipeline (CI/CD)

The project follows a **"Code as Infrastructure"** philosophy. 



### Automated Workflow:
1.  **Code Commit**: Developer pushes code to the `main` branch.
2.  **Identity Handshake**: GitHub authenticates with GCP via **Workload Identity Federation (OIDC)**.
3.  **Containerization**: GitHub builds a Debian-based Docker image containing all spatial dependencies (`GDAL`, `PROJ`).
4.  **Push**: Image is pushed to **Artifact Registry**.
5.  **Trigger**: `pipeline_deploy.py` is executed on the GitHub runner, submitting the job to **Vertex AI Pipelines**.

---

## 📁 Project Structure

```text
├── .github/workflows/
│   └── deploy.yml           # CI/CD Automation script
├── input_collector/         # Training data (CSV/Parquet)
├── src/
│   ├── ML_functions.py      # Core helper functions & GCS logic
│   ├── HP_tuning.py         # Grid search & parameter optimization
│   └── ML_execution.py      # Production training & SHAP analysis
├── Dockerfile               # Production environment definition
├── pipeline_deploy.py       # Vertex AI Orchestrator (KFP)
└── pyproject.toml           # Dependency management (uv)
```

## 📊 Monitoring & Results

All results are automatically versioned and stored for auditability and research.

* **Storage**: All results are synced to the `ML_results/` directory in **Google Cloud Storage**.
* **Metrics**: Detailed **Evaluation Statistics** (MAE, RMSE, and $R^2$) are generated for every lead time and county.
* **Explainability**: **SHAP Summary plots** are logged directly to the **Weights & Biases** dashboard, providing global and local feature importance tracking.



---

## 🛠️ Local Setup

To test the environment or run small-scale experiments locally:

1.  **Generate Lockfile**: 

    ```bash
    uv lock
    ```
    
2.  **Build Locally**: 

    ```bash
    docker build -t drought-test .
    ```

3.  **Run Test**: 

    ```bash
    docker run -it drought-test /bin/bash
    ```