import os
from datetime import datetime
from kfp import dsl, compiler
from google.cloud import aiplatform

# --- 1. Configuration ---
# These are pulled from your GitHub Actions env block
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BUCKET_NAME = f"{PROJECT_ID}-drought-ml"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

# Ensure this matches your Artifact Registry path exactly
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/drought-repo/drought-model:latest"

# --- 2. Define Components (KFP v2 Syntax) ---
@dsl.container_component
def hyperparameter_tuning_op():
    """
    Component for Hyperparameter Tuning.
    Uses -m to ensure 'src' is treated as a package, preventing ModuleNotFoundErrors.
    """
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "-m", "src.HP_tuning"]
    )

@dsl.container_component
def model_execution_op():
    """
    Component for Final Model Training and SHAP Explainability.
    """
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "-m", "src.ML_execution"],
        # pass api key
        env ={
            "WANDB_API_KEY": WANDB_API_KEY,
            # Ensure the script knows where to save results so they persist
            "AIP_CHECKPOINT_DIR": f"/gcs/{BUCKET_NAME.replace('gs://', '')}/output"
        }
    )

# --- 3. Define the Pipeline ---
@dsl.pipeline(name="drought-forecasting-pipeline")
def drought_pipeline():
    # 1. Tuning Task: 4 CPUs and 16GB is balanced for Kenya-scoped data.
    tuning_task = hyperparameter_tuning_op()
    tuning_task.set_cpu_limit('4')
    tuning_task.set_memory_limit('16G')
    tuning_task.set_retry(num_retries=1) 
    tuning_task.set_display_name("Hyperparameter Tuning (Kenya)")
    
    # 2. Execution Task: Higher RAM (32G) allocated for memory-heavy SHAP values.
    execution_task = model_execution_op()
    execution_task.set_cpu_limit('8')
    execution_task.set_memory_limit('32G')
    execution_task.set_display_name("Model Training & XAI (Kenya)")
    
    # Ensures tuning finishes before training starts
    execution_task.after(tuning_task)

# --- 4. Compile and Submit to Vertex AI ---
if __name__ == "__main__":
    # Compile the pipeline to a JSON file (standard KFP v2 output)
    PACKAGE_PATH = "drought_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=drought_pipeline,
        package_path=PACKAGE_PATH
    )

    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Unique Job ID for tracking in the console
    job_id = f"drought-kenya-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    job = aiplatform.PipelineJob(
        display_name="Kenya-Drought-ML-Pipeline",
        template_path=PACKAGE_PATH,
        job_id=job_id,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False  # Set to True to skip steps that haven't changed
    )

    # Submit the job to the cloud
    job.submit()
    
    print("-" * 30)
    print(f"✅ Pipeline submitted successfully!")
    print(f"🔗 View Job: https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{job_id}")
    print("-" * 30)