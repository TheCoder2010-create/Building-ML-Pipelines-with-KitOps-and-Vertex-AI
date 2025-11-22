from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform
import os

# Configuration
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION', 'us-central1')
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts"

@component(
    base_image="python:3.9",
    packages_to_install=['pykitops', 'google-cloud-storage']
)
def unpack_modelkit_op(
    modelkit_uri: str,
    output_path: Output[Dataset]
) -> dict:
    """Unpacks a ModelKit from registry"""
    import subprocess
    import json
    import os
    
    # Pull ModelKit
    subprocess.run(['kit', 'pull', modelkit_uri], check=True)
    
    # Unpack to output path
    unpack_dir = output_path.path
    os.makedirs(unpack_dir, exist_ok=True)
    
    subprocess.run([
        'kit', 'unpack', modelkit_uri, 
        '-d', unpack_dir
    ], check=True)
    
    # Return metadata
    metadata = {
        'modelkit_uri': modelkit_uri,
        'unpack_path': unpack_dir,
        'data_path': f"{unpack_dir}/data",
        'model_path': f"{unpack_dir}/models",
        'code_path': f"{unpack_dir}/src"
    }
    
    return metadata


@component(
    base_image="python:3.9",
    packages_to_install=['pandas', 'scikit-learn', 'numpy']
)
def train_model_op(
    data_path: str,
    model_output: Output[Model],
    metrics: Output[Metrics]
):
    """Trains ML model from unpacked data"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import pickle
    import os
    
    # Load training data
    # Note: data_path from unpack_modelkit_op points to the root of unpacked dir
    # We need to append /data/train.csv based on our Kitfile structure
    train_csv_path = f"{data_path}/data/train.csv"
    print(f"Loading data from: {train_csv_path}")
    
    train_df = pd.read_csv(train_csv_path)
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Evaluate
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    # Save model
    os.makedirs(os.path.dirname(model_output.path), exist_ok=True)
    with open(model_output.path, 'wb') as f:
        pickle.dump(model, f)
    
    # Log metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("f1_score", f1)
    
    print(f"Training complete: Accuracy={accuracy:.4f}, F1={f1:.4f}")


@component(
    base_image="python:3.9",
    packages_to_install=['google-cloud-aiplatform']
)
def deploy_model_op(
    project_id: str,
    region: str,
    model_path: str,
    model_display_name: str,
    endpoint_display_name: str
) -> str:
    """Deploys model to Vertex AI Endpoint"""
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=region)
    
    # Upload model
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    
    # Create or get endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"'
    )
    
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name
        )
    
    # Deploy model
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model_display_name,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1 # Reduced for demo
    )
    
    return endpoint.resource_name


@component(
    base_image="python:3.9",
    packages_to_install=['pykitops', 'pyyaml']
)
def pack_new_modelkit_op(
    model_path: str,
    modelkit_name: str,
    modelkit_version: str,
    registry_uri: str
):
    """Packs trained model into new ModelKit version"""
    import subprocess
    import yaml
    import os
    import shutil
    
    # Create a working directory
    work_dir = "new_modelkit"
    os.makedirs(work_dir, exist_ok=True)
    
    # Copy model
    model_dir = os.path.join(work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    # KFP passes model_path as a file path usually, but sometimes a dir
    if os.path.isfile(model_path):
        shutil.copy(model_path, os.path.join(model_dir, "model.pkl"))
    else:
        # If it's a directory, find the model file
        for root, dirs, files in os.walk(model_path):
            for file in files:
                shutil.copy(os.path.join(root, file), os.path.join(model_dir, "model.pkl"))
                break

    # Create Kitfile
    kitfile = {
        'manifestVersion': 'v1.0',
        'package': {
            'name': modelkit_name,
            'version': modelkit_version,
            'description': 'Trained model from Vertex AI Pipeline'
        },
        'model': {
            'name': modelkit_name,
            'path': './models/model.pkl',
            'framework': 'scikit-learn'
        }
    }
    
    with open(os.path.join(work_dir, 'Kitfile'), 'w') as f:
        yaml.dump(kitfile, f)
    
    # Pack and push
    tag = f"{registry_uri}/{modelkit_name}:{modelkit_version}"
    
    # We need to run kit from within the work_dir
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        subprocess.run(['kit', 'pack', '.', '-t', tag], check=True)
        subprocess.run(['kit', 'push', tag], check=True)
        print(f"New ModelKit pushed: {tag}")
    finally:
        os.chdir(cwd)


@dsl.pipeline(
    name="kitops-vertex-training-pipeline",
    description="End-to-end ML pipeline using KitOps and Vertex AI",
    pipeline_root=PIPELINE_ROOT
)
def training_pipeline(
    project_id: str,
    region: str,
    modelkit_uri: str,
    model_display_name: str = "kitops-model",
    endpoint_display_name: str = "kitops-endpoint",
    deploy_threshold: float = 0.85
):
    """Complete training pipeline integrating KitOps"""
    
    # Step 1: Unpack ModelKit
    unpack_task = unpack_modelkit_op(modelkit_uri=modelkit_uri)
    
    # Step 2: Train model
    train_task = train_model_op(
        data_path=unpack_task.outputs['output_path']
    )
    
    # Step 3: Conditional deployment based on metrics
    with dsl.Condition(
        train_task.outputs['metrics'].metadata['accuracy'] >= deploy_threshold,
        name="check-accuracy"
    ):
        deploy_task = deploy_model_op(
            project_id=project_id,
            region=region,
            model_path=train_task.outputs['model_output'],
            model_display_name=model_display_name,
            endpoint_display_name=endpoint_display_name
        )
        
        # Step 4: Pack new ModelKit version
        # Note: We need to construct the registry URI. 
        # Assuming standard format: REGION-docker.pkg.dev/PROJECT_ID/ml-modelkits
        # We can pass this as a parameter or construct it
        registry_uri = f"{region}-docker.pkg.dev/{project_id}/ml-modelkits"
        
        pack_task = pack_new_modelkit_op(
            model_path=train_task.outputs['model_output'],
            modelkit_name="sentiment-classifier",
            modelkit_version="v1.1.0", # In real world, this should be dynamic
            registry_uri=registry_uri
        )


# Compile pipeline
def compile_pipeline(output_file: str = "pipeline.json"):
    """Compiles pipeline to JSON"""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=output_file
    )
    print(f"Pipeline compiled to {output_file}")


# Run pipeline
def run_pipeline():
    """Executes pipeline on Vertex AI"""
    
    if not PROJECT_ID:
        raise ValueError("PROJECT_ID environment variable not set")
        
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    job = aiplatform.PipelineJob(
        display_name="kitops-training-pipeline",
        template_path="pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            'project_id': PROJECT_ID,
            'region': REGION,
            'modelkit_uri': f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-modelkits/sentiment-classifier:v1.0.0",
            'model_display_name': 'sentiment-classifier-model',
            'endpoint_display_name': 'sentiment-classifier-endpoint',
            'deploy_threshold': 0.5 # Lowered for demo to ensure deployment
        },
        enable_caching=True
    )
    
    job.submit()
    print(f"Pipeline submitted: {job.resource_name}")
    return job


if __name__ == "__main__":
    # Compile and run
    compile_pipeline()
    # job = run_pipeline() # Commented out to avoid auto-run without config
