import os
from datetime import datetime
from kfp import dsl, compiler
from google.cloud import aiplatform

# --- 1. Configuration ---
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BUCKET_NAME = f"{PROJECT_ID}-drought-ml"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/drought-repo/drought-model:latest"

# --- 2. Define Components ---
@dsl.container_component
def hyperparameter_tuning_op():
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "-m", "src.HP_tuning"]
    )

@dsl.container_component
def model_execution_op():
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "-m", "src.ML_execution"]
    )

# --- 3. Define the Pipeline ---
@dsl.pipeline(name="drought-forecasting-pipeline")
def drought_pipeline():
    # 1. Tuning Task
    tuning_task = hyperparameter_tuning_op()
    tuning_task.set_env_variable(name="WANDB_API_KEY", value=WANDB_API_KEY)
    tuning_task.set_env_variable(name="BUCKET_NAME", value=f"gs://{BUCKET_NAME}")
    
    # Resource Allocation
    tuning_task.set_cpu_limit('4')
    tuning_task.set_memory_limit('16G')
    tuning_task.set_retry(num_retries=1) 
    tuning_task.set_display_name("HP Tuning: XGBoost Clusters")
    
    # 2. Execution Task
    execution_task = model_execution_op()
    execution_task.set_env_variable(name="WANDB_API_KEY", value=WANDB_API_KEY)
    execution_task.set_env_variable(name="BUCKET_NAME", value=f"gs://{BUCKET_NAME}")
    
    # ✅ SHAP is memory-intensive. 32G is safe; 64G is "bulletproof" for deep trees.
    execution_task.set_cpu_limit('8')
    execution_task.set_memory_limit('32G') 
    
    # ✅ Caching is disabled to ensure we always get fresh SHAP plots and WandB logs
    execution_task.set_caching_options(enable_caching=False)
    execution_task.set_display_name("Training & XAI: SHAP Explanations")
    
    # Dependency management
    execution_task.after(tuning_task)

# --- 4. Submission Logic ---
if __name__ == "__main__":
    PACKAGE_PATH = "drought_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=drought_pipeline,
        package_path=PACKAGE_PATH
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job_id = f"drought-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    job = aiplatform.PipelineJob(
        display_name="Kenya-Drought-ML-Pipeline",
        template_path=PACKAGE_PATH,
        job_id=job_id,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False 
    )

    job.submit()
    
    print(f"✅ Pipeline submitted: {job_id}")