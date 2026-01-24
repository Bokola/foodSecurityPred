import os
from kfp import dsl, compiler
from google.cloud import aiplatform

# --- CONFIGURATION ---
PROJECT_ID = "your-google-project-id"
REGION = "us-central1"
BUCKET_NAME = f"gs://{PROJECT_ID}-drought-ml"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/drought-repo/model:latest"

@dsl.pipeline(name="drought-tuning-and-execution")
def drought_pipeline():
    # Step 1: HP Tuning
    tuning_task = dsl.ContainerOp(
        name="hp-tuning",
        image=IMAGE_URI,
        command=["uv", "run", "python", "-m", "src.HP_tuning"]
    )

    # Step 2: Final Execution (Dependent on Tuning)
    execution_task = dsl.ContainerOp(
        name="final-execution",
        image=IMAGE_URI,
        command=["uv", "run", "python", "-m", "src.ML_execution"]
    )
    
    # Enforce order: Execution only starts when Tuning finishes successfully
    execution_task.after(tuning_task)

# --- DEPLOYMENT ---
compiler.Compiler().compile(drought_pipeline, "pipeline.yaml")

aiplatform.init(project=PROJECT_ID, location=REGION)
job = aiplatform.PipelineJob(
    display_name="Drought_Forecasting_Sequential_Run",
    template_path="pipeline.yaml",
    pipeline_root=f"{BUCKET_NAME}/pipeline_root",
)
job.run()