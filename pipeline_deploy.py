import os
from datetime import datetime
from kfp import dsl, compiler
from google.cloud import aiplatform

# --- 1. Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
BUCKET_NAME = f"{PROJECT_ID}-drought-ml"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/drought-repo/drought-model:latest"

# --- 2. Define Components (KFP v2 Syntax) ---
@dsl.container_component
def hyperparameter_tuning_op():
    """Component for Hyperparameter Tuning."""
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/HP_tuning.py"]
    )

@dsl.container_component
def model_execution_op():
    """Component for Final Model Training/Execution."""
    return dsl.ContainerSpec(
        image=IMAGE_URI,
        command=["python", "src/ML_execution.py"]
    )

# --- 3. Define the Pipeline ---
@dsl.pipeline(
    name="drought-forecasting-pipeline",
    description="Automated drought prediction pipeline with SHAP and W&B logging."
)
def drought_pipeline():
    # Step 1: Run Tuning
    tuning_task = hyperparameter_tuning_op()
    
    # Step 2: Run Execution (only after tuning is finished)
    execution_task = model_execution_op()
    execution_task.after(tuning_task)

# --- 4. Compile and Submit to Vertex AI ---
if __name__ == "__main__":
    # Compile the pipeline to a JSON file
    PACKAGE_PATH = "drought_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=drought_pipeline,
        package_path=PACKAGE_PATH
    )

    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Create the Pipeline Job
    job_id = f"drought-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    job = aiplatform.PipelineJob(
        display_name="Drought-ML-Pipeline-Run",
        template_path=PACKAGE_PATH,
        job_id=job_id,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False  # Set to True if you want to skip finished steps
    )

    # Submit the job
    job.submit()
    print(f"Pipeline submitted successfully! View it here: https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{job_id}")