# Building ML Pipelines with KitOps and Vertex AI

This guide demonstrates how to combine KitOps, an open-source ML packaging tool, with Google Cloud's Vertex AI Pipelines to create robust, reproducible, and production-ready machine learning workflows. By leveraging KitOps' standardized ModelKit packaging with Vertex AI's serverless pipeline execution, teams can achieve seamless collaboration between data scientists, developers, and platform engineers.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Core Concepts](#core-concepts)
4. [Environment Setup](#environment-setup)
5. [Creating Your First ModelKit](#creating-your-first-modelkit)
6. [Building Vertex AI Pipelines with KitOps](#building-vertex-ai-pipelines-with-kitops)
7. [Advanced Integration Patterns](#advanced-integration-patterns)
8. [CI/CD Integration](#cicd-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### The Integration Flow

<img width="1024" height="1024" alt="diagram" src="https://github.com/user-attachments/assets/d68163f9-31cb-4253-8c1a-65e3e3d79164" />


### Key Benefits

- **Standardized Packaging**: All model artifacts, code, datasets, and configs in one versioned package
- **OCI Compliance**: Store ModelKits alongside container images in existing registries
- **Reproducibility**: Immutable, tamper-proof artifacts ensure consistent deployments
- **Serverless Execution**: Vertex AI handles infrastructure, scaling, and orchestration
- **Audit Trail**: Complete lineage tracking for compliance (EU AI Act ready)

---

## Prerequisites

### Required Tools

1. **Google Cloud Project** with billing enabled
2. **Kit CLI** (latest version)
3. **Google Cloud SDK** (gcloud CLI)
4. **Python 3.8+** with pip
5. **Docker** (optional, for local testing)

### Required APIs

Enable these Google Cloud APIs:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com
```

### IAM Permissions

Your service account needs these roles:

- `roles/aiplatform.user` - Vertex AI operations
- `roles/artifactregistry.writer` - Push/pull artifacts
- `roles/storage.admin` - GCS bucket access
- `roles/iam.serviceAccountUser` - Pipeline execution

---

## Core Concepts

### KitOps Fundamentals

#### ModelKit

A ModelKit is an OCI-compliant artifact containing:
- **Model weights**: Trained model files (ONNX, SavedModel, PyTorch, etc.)
- **Code**: Training scripts, inference code, preprocessing
- **Datasets**: Training/validation data or references
- **Configuration**: Hyperparameters, environment specs
- **Documentation**: README, model cards, metadata

#### Kitfile

YAML manifest describing the ModelKit contents:

```yaml
manifestVersion: v1.0
package:
  name: my-model
  version: 1.0.0
  description: Classification model for production

model:
  name: classifier
  path: ./models/model.onnx
  framework: onnx
  
code:
  - path: ./src/
    description: Training and inference code

datasets:
  - name: training_data
    path: ./data/train.csv
    description: Training dataset (10k samples)

docs:
  - path: ./README.md
  - path: ./model_card.md
```

### Vertex AI Pipelines Fundamentals

#### Pipeline Components

Self-contained execution units defined as Python functions or containers:

```python
from kfp.v2.dsl import component

@component(base_image="python:3.9")
def preprocess_data(input_path: str) -> str:
    """Preprocesses raw data"""
    import pandas as pd
    # Component logic here
    return output_path
```

#### Pipeline Definition

DAG connecting components with input/output dependencies:

```python
from kfp.v2 import dsl

@dsl.pipeline(
    name="ml-training-pipeline",
    description="End-to-end training pipeline"
)
def training_pipeline(
    project_id: str,
    region: str,
    modelkit_uri: str
):
    # Define pipeline tasks
    unpack_task = unpack_modelkit_op(modelkit_uri)
    train_task = train_model_op(unpack_task.outputs['model_path'])
    deploy_task = deploy_model_op(train_task.outputs['model'])
```

---

## Environment Setup

### 1. Install Kit CLI

**macOS (Homebrew)**:
```bash
brew tap kitops-ml/kitops
brew install kitops
```

**Linux**:
```bash
curl -L https://github.com/kitops-ml/kitops/releases/latest/download/kitops-linux-x86_64.tar.gz | tar -xz
sudo mv kit /usr/local/bin/
```

**Windows (PowerShell)**:
```powershell
Invoke-WebRequest -Uri "https://github.com/kitops-ml/kitops/releases/latest/download/kitops-windows-x86_64.zip" -OutFile "kitops.zip"
Expand-Archive -Path "kitops.zip" -DestinationPath "C:\Program Files\kitops"
```

Verify installation:
```bash
kit version
```

### 2. Configure Google Cloud

```bash
# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Set region
export REGION="us-central1"
gcloud config set compute/region $REGION
```

### 3. Create Artifact Registry Repository

```bash
# Create repository for ModelKits
gcloud artifacts repositories create ml-modelkits \
  --repository-format=docker \
  --location=$REGION \
  --description="KitOps ModelKits repository"

# Configure Docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### 4. Install Python Dependencies

```bash
pip install --upgrade \
  google-cloud-aiplatform==1.59.0 \
  kfp==2.7.0 \
  google-cloud-pipeline-components==2.14.0 \
  pykitops
```

### 5. Create GCS Bucket for Pipeline Artifacts

```bash
export BUCKET_NAME="${PROJECT_ID}-pipeline-artifacts"
gcloud storage buckets create gs://${BUCKET_NAME} \
  --location=$REGION
```

---

## Creating Your First ModelKit

### Project Structure

```
ml-project/
├── Kitfile
├── models/
│   └── model.pkl
├── src/
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
├── data/
│   ├── train.csv
│   └── validation.csv
├── config/
│   └── hyperparameters.yaml
└── docs/
    ├── README.md
    └── model_card.md
```

### Step 1: Create Kitfile

Create `Kitfile` in your project root:

```yaml
manifestVersion: v1.0

package:
  name: sentiment-classifier
  version: 1.0.0
  description: BERT-based sentiment classification model
  authors:
    - name: ML Team
      email: ml-team@company.com
  license: Apache-2.0

model:
  name: bert-sentiment
  path: ./models/model.pkl
  framework: scikit-learn
  version: 1.0.0
  description: Fine-tuned BERT for sentiment analysis

code:
  - path: ./src/train.py
    description: Training script
  - path: ./src/predict.py
    description: Inference script
  - path: ./src/requirements.txt
    description: Python dependencies

datasets:
  - name: training_data
    path: ./data/train.csv
    description: 50k labeled reviews for training
  
  - name: validation_data
    path: ./data/validation.csv
    description: 10k labeled reviews for validation

config:
  - path: ./config/hyperparameters.yaml
    description: Model hyperparameters and training config

docs:
  - path: ./docs/README.md
    description: Model documentation
  - path: ./docs/model_card.md
    description: Model card with performance metrics
```

### Step 2: Pack ModelKit

```bash
# Navigate to project directory
cd ml-project

# Pack the ModelKit
kit pack . -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/sentiment-classifier:v1.0.0

# Verify local ModelKit
kit list
kit inspect ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/sentiment-classifier:v1.0.0
```

### Step 3: Push to Registry

```bash
# Push to Artifact Registry
kit push ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/sentiment-classifier:v1.0.0

# Verify in registry
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits
```

### Step 4: Pull and Unpack (Test)

```bash
# Pull ModelKit
kit pull ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/sentiment-classifier:v1.0.0

# Unpack to local directory
kit unpack ${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/sentiment-classifier:v1.0.0 -d ./unpacked

# Verify contents
ls -la ./unpacked
```

---

## Building Vertex AI Pipelines with KitOps

### Pipeline Architecture

```
<img width="816" height="510" alt="vertex ai pipline " src="https://github.com/user-attachments/assets/88f21763-a039-462c-a70a-81aa6fb07ed4" />

```

### Example 1: Basic Training Pipeline

Create `pipeline.py`:

```python
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
    
    # Load training data
    train_df = pd.read_csv(f"{data_path}/train.csv")
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
        max_replica_count=3
    )
    
    return endpoint.resource_name


@component(
    base_image="python:3.9",
    packages_to_install=['pykitops']
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
            'path': model_path,
            'framework': 'scikit-learn'
        }
    }
    
    with open('Kitfile', 'w') as f:
        yaml.dump(kitfile, f)
    
    # Pack and push
    tag = f"{registry_uri}/{modelkit_name}:{modelkit_version}"
    subprocess.run(['kit', 'pack', '.', '-t', tag], check=True)
    subprocess.run(['kit', 'push', tag], check=True)
    
    print(f"New ModelKit pushed: {tag}")


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
        pack_task = pack_new_modelkit_op(
            model_path=train_task.outputs['model_output'],
            modelkit_name="sentiment-classifier",
            modelkit_version="v1.1.0",
            registry_uri=f"{region}-docker.pkg.dev/{project_id}/ml-modelkits"
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
            'deploy_threshold': 0.85
        },
        enable_caching=True
    )
    
    job.submit()
    print(f"Pipeline submitted: {job.resource_name}")
    return job


if __name__ == "__main__":
    # Compile and run
    compile_pipeline()
    job = run_pipeline()
    
    # Wait for completion (optional)
    # job.wait()
```

### Run the Pipeline

```bash
# Set environment variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Run pipeline
python pipeline.py
```

---

## Advanced Integration Patterns

### Pattern 1: Multi-Stage Pipeline with Model Versioning

```python
@dsl.pipeline(
    name="multi-stage-modelkit-pipeline",
    description="Training, validation, and versioning pipeline"
)
def multi_stage_pipeline(
    project_id: str,
    base_modelkit_uri: str,
    experiment_name: str
):
    """
    Advanced pipeline with:
    - Multiple model variants
    - A/B testing preparation
    - Automatic versioning
    """
    
    # Stage 1: Unpack base ModelKit
    unpack_task = unpack_modelkit_op(modelkit_uri=base_modelkit_uri)
    
    # Stage 2: Train multiple model variants in parallel
    models = []
    for i, config in enumerate([
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 15},
        {'n_estimators': 300, 'max_depth': 20}
    ]):
        train_task = train_model_variant_op(
            data_path=unpack_task.outputs['output_path'],
            hyperparameters=config,
            variant_name=f"variant_{i}"
        )
        models.append(train_task)
    
    # Stage 3: Compare models
    comparison_task = compare_models_op(
        models=[m.outputs['model_output'] for m in models]
    )
    
    # Stage 4: Deploy best model
    deploy_task = deploy_best_model_op(
        best_model=comparison_task.outputs['best_model'],
        project_id=project_id
    )
    
    # Stage 5: Create versioned ModelKit
    pack_task = pack_versioned_modelkit_op(
        model_path=comparison_task.outputs['best_model'],
        metrics=comparison_task.outputs['metrics'],
        version="auto"  # Auto-increment version
    )
```

### Pattern 2: Continuous Training with ModelKit Updates

```python
@component(packages_to_install=['pykitops', 'google-cloud-storage'])
def check_for_new_data_op(
    bucket_name: str,
    last_training_date: str
) -> bool:
    """Checks if new training data is available"""
    from google.cloud import storage
    from datetime import datetime
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='training-data/')
    
    for blob in blobs:
        if blob.time_created > datetime.fromisoformat(last_training_date):
            return True
    return False


@dsl.pipeline(name="continuous-training-pipeline")
def continuous_training_pipeline(
    project_id: str,
    data_bucket: str,
    modelkit_uri: str
):
    """Pipeline that runs on schedule to retrain with new data"""
    
    # Check for new data
    check_task = check_for_new_data_op(
        bucket_name=data_bucket,
        last_training_date="2025-01-01T00:00:00"
    )
    
    # Only retrain if new data exists
    with dsl.Condition(check_task.output == True, name="new-data-available"):
        
        # Unpack current ModelKit
        unpack_task = unpack_modelkit_op(modelkit_uri=modelkit_uri)
        
        # Load new data
        load_task = load_new_data_op(bucket_name=data_bucket)
        
        # Merge datasets
        merge_task = merge_datasets_op(
            existing_data=unpack_task.outputs['output_path'],
            new_data=load_task.outputs['data_path']
        )
        
        # Retrain
        train_task = train_model_op(
            data_path=merge_task.outputs['merged_data']
        )
        
        # Deploy if improved
        with dsl.Condition(
            train_task.outputs['metrics'].metadata['accuracy'] > 0.90
        ):
            deploy_task = deploy_model_op(
                project_id=project_id,
                model_path=train_task.outputs['model_output']
            )
            
            # Create new ModelKit version
            pack_task = pack_new_modelkit_op(
                model_path=train_task.outputs['model_output'],
                version="auto-increment"
            )
```

### Pattern 3: ModelKit Promotion Pipeline

```python
@component
def validate_modelkit_op(
    modelkit_uri: str,
    validation_tests: list
) -> dict:
    """Runs validation tests on ModelKit"""
    import subprocess
    import json
    
    results = {'passed': True, 'tests': {}}
    
    # Pull and inspect ModelKit
    subprocess.run(['kit', 'pull', modelkit_uri], check=True)
    
    # Run validation tests
    for test in validation_tests:
        # Example: Check model file exists
        # Example: Validate data schemas
        # Example: Run smoke tests
        result = run_test(test, modelkit_uri)
        results['tests'][test] = result
        if not result:
            results['passed'] = False
    
    return results


@dsl.pipeline(name="modelkit-promotion-pipeline")
def promotion_pipeline(
    dev_modelkit_uri: str,
    staging_modelkit_uri: str,
    prod_modelkit_uri: str
):
    """
    Promotes ModelKit through environments:
    DEV -> STAGING -> PRODUCTION
    """
    
    # Validate in dev
    dev_validation = validate_modelkit_op(
        modelkit_uri=dev_modelkit_uri,
        validation_tests=['schema_check', 'smoke_test', 'security_scan']
    )
    
    with dsl.Condition(
        dev_validation.outputs['passed'] == True,
        name="dev-validation-passed"
    ):
        # Promote to staging
        staging_promote = promote_modelkit_op(
            source_uri=dev_modelkit_uri,
            target_uri=staging_modelkit_uri
        )
        
        # Run integration tests in staging
        staging_tests = run_integration_tests_op(
            modelkit_uri=staging_modelkit_uri
        )
        
        with dsl.Condition(
            staging_tests.outputs['passed'] == True,
            name="staging-tests-passed"
        ):
            # Promote to production
            prod_promote = promote_modelkit_op(
                source_uri=staging_modelkit_uri,
                target_uri=prod_modelkit_uri
            )
            
            # Deploy to production endpoint
            prod_deploy = deploy_to_production_op(
                modelkit_uri=prod_modelkit_uri
            )
```

---

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/ml-pipeline.yaml`:

```yaml
name: ML Pipeline with KitOps

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'data/**'
      - 'Kitfile'
  workflow_dispatch:

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  MODELKIT_NAME: sentiment-classifier

jobs:
  build-and-push-modelkit:
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      id-token: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      
      - name: Install Kit CLI
        run: |
          curl -L https://github.com/kitops-ml/kitops/releases/latest/download/kitops-linux-x86_64.tar.gz | tar -xz
          sudo mv kit /usr/local/bin/
          kit version
      
      - name: Configure Docker for Artifact Registry
        run: |
          gcloud auth configure-docker ${REGION}-docker.pkg.dev
      
      - name: Build and push ModelKit
        run: |
          VERSION=$(cat version.txt)
          MODELKIT_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/${MODELKIT_NAME}:${VERSION}"
          
          kit pack . -t $MODELKIT_URI
          kit push $MODELKIT_URI
          
          echo "MODELKIT_URI=$MODELKIT_URI" >> $GITHUB_ENV
      
      - name: Trigger Vertex AI Pipeline
        run: |
          python -m pip install google-cloud-aiplatform kfp
          python scripts/trigger_pipeline.py \
            --project-id $PROJECT_ID \
            --region $REGION \
            --modelkit-uri $MODELKIT_URI

  run-tests:
    needs: build-and-push-modelkit
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
      
      - name: Run model validation tests
        run: |
          python -m pip install pytest pykitops
          pytest tests/model_validation/ -v
      
      - name: Check pipeline status
        run: |
          python scripts/check_pipeline_status.py \
            --project-id $PROJECT_ID \
            --region $REGION
```

### GitLab CI/CD Pipeline

Create `.gitlab-ci.yml`:

```yaml
stages:
  - build
  - test
  - deploy

variables:
  REGION: us-central1
  MODELKIT_REGISTRY: ${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/ml-modelkits

before_script:
  - echo $GCP_SERVICE_KEY | gcloud auth activate-service-account --key-file=-
  - gcloud config set project $GCP_PROJECT_ID

build_modelkit:
  stage: build
  image: google/cloud-sdk:latest
  script:
    - curl -L https://github.com/kitops-ml/kitops/releases/latest/download/kitops-linux-x86_64.tar.gz | tar -xz
    - mv kit /usr/local/bin/
    - gcloud auth configure-docker ${REGION}-docker.pkg.dev
    - |
      VERSION=$(cat version.txt)
      MODELKIT_URI="${MODELKIT_REGISTRY}/sentiment-classifier:${VERSION}"
      kit pack . -t $MODELKIT_URI
      kit push $MODELKIT_URI
      echo "MODELKIT_URI=$MODELKIT_URI" > modelkit.env
  artifacts:
    reports:
      dotenv: modelkit.env

test_modelkit:
  stage: test
  image: python:3.9
  dependencies:
    - build_modelkit
  script:
    - pip install pytest pykitops google-cloud-aiplatform
    - kit pull $MODELKIT_URI
    - kit unpack $MODELKIT_URI -d ./test_env
    - pytest tests/ -v

deploy_pipeline:
  stage: deploy
  image: python:3.9
  dependencies:
    - build_modelkit
  script:
    - pip install google-cloud-aiplatform kfp
    - python scripts/deploy_pipeline.py --modelkit-uri $MODELKIT_URI
  only:
    - main
```

### Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        PROJECT_ID = credentials('gcp-project-id')
        REGION = 'us-central1'
        GCP_CREDENTIALS = credentials('gcp-service-account')
        MODELKIT_NAME = 'sentiment-classifier'
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    sh '''
                        curl -L https://github.com/kitops-ml/kitops/releases/latest/download/kitops-linux-x86_64.tar.gz | tar -xz
                        sudo mv kit /usr/local/bin/
                        
                        gcloud auth activate-service-account --key-file=${GCP_CREDENTIALS}
                        gcloud config set project ${PROJECT_ID}
                        gcloud auth configure-docker ${REGION}-docker.pkg.dev
                    '''
                }
            }
        }
        
        stage('Build ModelKit') {
            steps {
                script {
                    def version = readFile('version.txt').trim()
                    env.MODELKIT_URI = "${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-modelkits/${MODELKIT_NAME}:${version}"
                    
                    sh """
                        kit pack . -t ${MODELKIT_URI}
                        kit push ${MODELKIT_URI}
                    """
                }
            }
        }
        
        stage('Validate ModelKit') {
            steps {
                sh '''
                    python -m pip install pytest pykitops
                    kit pull ${MODELKIT_URI}
                    kit unpack ${MODELKIT_URI} -d ./validation
                    pytest tests/validation/ -v
                '''
            }
        }
        
        stage('Deploy Pipeline') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    python -m pip install google-cloud-aiplatform kfp
                    python scripts/trigger_vertex_pipeline.py \
                        --project-id ${PROJECT_ID} \
                        --region ${REGION} \
                        --modelkit-uri ${MODELKIT_URI}
                '''
            }
        }
        
        stage('Monitor Pipeline') {
            steps {
                sh '''
                    python scripts/monitor_pipeline.py \
                        --project-id ${PROJECT_ID} \
                        --region ${REGION} \
                        --timeout 3600
                '''
            }
        }
    }
    
    post {
        success {
            echo "Pipeline completed successfully!"
            // Send notifications
        }
        failure {
            echo "Pipeline failed!"
            // Send alerts
        }
    }
}
```

---

## Best Practices

### ModelKit Organization

#### 1. Naming Conventions

```yaml
# Use semantic versioning
package:
  name: model-name
  version: MAJOR.MINOR.PATCH

# Examples:
# 1.0.0 - Initial production release
# 1.1.0 - New feature (backward compatible)
# 1.1.1 - Bug fix
# 2.0.0 - Breaking change
```

#### 2. Directory Structure

```
modelkit-project/
├── Kitfile                    # ModelKit manifest
├── models/
│   ├── model.pkl             # Primary model
│   └── auxiliary_model.pkl   # Supporting models
├── src/
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── predict.py            # Inference script
│   ├── preprocess.py         # Data preprocessing
│   └── requirements.txt      # Python dependencies
├── data/
│   ├── train/                # Training data
│   ├── validation/           # Validation data
│   └── test/                 # Test data
├── config/
│   ├── hyperparameters.yaml  # Model config
│   └── deployment.yaml       # Deployment config
├── docs/
│   ├── README.md             # Project overview
│   ├── model_card.md         # Model documentation
│   └── API.md                # API documentation
└── tests/
    ├── test_model.py         # Model tests
    └── test_data.py          # Data validation tests
```

#### 3. Metadata Best Practices

```yaml
manifestVersion: v1.0

package:
  name: fraud-detection-model
  version: 2.1.0
  description: |
    XGBoost model for real-time fraud detection.
    Trained on 2M transactions from Q4 2024.
  authors:
    - name: Data Science Team
      email: ds-team@company.com
  license: Apache-2.0
  tags:
    - fraud-detection
    - xgboost
    - production
  metadata:
    training_date: "2025-01-15"
    dataset_version: "v3.2"
    performance_metrics:
      accuracy: 0.947
      precision: 0.923
      recall: 0.951
      f1_score: 0.937
    environment:
      python_version: "3.9"
      framework_versions:
        xgboost: "1.7.0"
        scikit-learn: "1.3.0"

model:
  name: fraud-detector
  path: ./models/xgboost_model.pkl
  framework: xgboost
  version: 1.7.0
  description: Primary fraud detection model
  
datasets:
  - name: training_data
    path: ./data/train.parquet
    description: 1.5M labeled transactions
    size: "2.3GB"
    
  - name: validation_data
    path: ./data/validation.parquet
    description: 300K labeled transactions
    size: "450MB"
```

### Vertex AI Pipeline Best Practices

#### 1. Component Design

```python
# Use type hints and documentation
@component(
    base_image="python:3.9-slim",
    packages_to_install=['pandas==2.0.0', 'scikit-learn==1.3.0']
)
def preprocess_data_op(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    scaling_method: str = "standard",
    handle_missing: str = "mean"
) -> dict:
    """
    Preprocesses raw data for model training.
    
    Args:
        input_data: Raw dataset with features and target
        output_data: Preprocessed dataset ready for training
        scaling_method: Feature scaling method ('standard' or 'minmax')
        handle_missing: Strategy for missing values ('mean', 'median', 'drop')
    
    Returns:
        Dictionary with preprocessing statistics
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Implementation here
    stats = {
        'rows_processed': len(df),
        'features_scaled': len(features),
        'missing_values_handled': missing_count
    }
    
    return stats
```

#### 2. Error Handling and Logging

```python
@component(packages_to_install=['google-cloud-logging'])
def robust_training_op(
    data_path: str,
    model_output: Output[Model]
):
    """Training component with comprehensive error handling"""
    from google.cloud import logging as cloud_logging
    import logging
    
    # Setup logging
    logging_client = cloud_logging.Client()
    logging_client.setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model training")
        
        # Load data with validation
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Validate data schema
        required_columns = ['feature1', 'feature2', 'target']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Train model
        model = train_model(df)
        logger.info("Training completed successfully")
        
        # Save model
        with open(model_output.path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_output.path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
```

#### 3. Caching Strategy

```python
@dsl.pipeline(
    name="optimized-pipeline",
    description="Pipeline with smart caching"
)
def optimized_pipeline(data_version: str):
    """
    Use caching for expensive operations that don't change often
    """
    
    # Cache data preprocessing (expensive, rarely changes)
    preprocess_task = preprocess_data_op(
        data_path=f"gs://bucket/data-{data_version}.csv"
    ).set_caching_options(True)
    
    # Don't cache training (always want fresh results)
    train_task = train_model_op(
        data_path=preprocess_task.outputs['output_data']
    ).set_caching_options(False)
    
    # Cache validation (deterministic given same model)
    validate_task = validate_model_op(
        model=train_task.outputs['model_output']
    ).set_caching_options(True)
```

### Security Best Practices

#### 1. Secure Credentials Management

```python
# Use Secret Manager for sensitive data
@component(packages_to_install=['google-cloud-secret-manager'])
def secure_data_access_op(
    project_id: str,
    secret_name: str,
    output_data: Output[Dataset]
):
    """Access data using secrets from Secret Manager"""
    from google.cloud import secretmanager
    
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    
    response = client.access_secret_version(request={"name": secret_path})
    api_key = response.payload.data.decode('UTF-8')
    
    # Use api_key to fetch data
    data = fetch_data_securely(api_key)
    
    # Never log secrets!
    print("Data fetched successfully")
```

#### 2. ModelKit Signing

```bash
# Sign ModelKit for verification
kit sign ${MODELKIT_URI} \
  --key-file=./private-key.pem

# Verify signature before deployment
kit verify ${MODELKIT_URI} \
  --key-file=./public-key.pem
```

#### 3. Access Control

```bash
# Grant minimum required permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud artifacts repositories add-iam-policy-binding ml-modelkits \
  --location=${REGION} \
  --member="serviceAccount:pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

### Performance Optimization

#### 1. Parallel Processing

```python
@dsl.pipeline(name="parallel-pipeline")
def parallel_training_pipeline():
    """Train multiple models in parallel"""
    
    # Unpack ModelKit once
    unpack_task = unpack_modelkit_op(modelkit_uri=MODELKIT_URI)
    
    # Train models in parallel with different hyperparameters
    with dsl.ParallelFor(
        items=[
            {'lr': 0.01, 'batch': 32},
            {'lr': 0.001, 'batch': 64},
            {'lr': 0.0001, 'batch': 128}
        ],
        name="parallel-training"
    ) as config:
        train_task = train_model_variant_op(
            data_path=unpack_task.outputs['output_path'],
            learning_rate=config.lr,
            batch_size=config.batch
        )
```

#### 2. Resource Optimization

```python
# Specify appropriate machine types for each component
@component(base_image="python:3.9")
def lightweight_preprocessing_op():
    """Light preprocessing - use smaller machine"""
    pass

# Set machine type in pipeline
preprocess_task = lightweight_preprocessing_op()
preprocess_task.set_cpu_limit('2')
preprocess_task.set_memory_limit('4G')

@component(base_image="tensorflow/tensorflow:2.13.0-gpu")
def heavy_training_op():
    """GPU-intensive training"""
    pass

# Use GPU for training
train_task = heavy_training_op()
train_task.set_accelerator_type('NVIDIA_TESLA_T4')
train_task.set_accelerator_limit(1)
train_task.set_cpu_limit('8')
train_task.set_memory_limit('32G')
```

#### 3. Data Handling

```python
@component
def efficient_data_loading_op(
    data_uri: str,
    output_data: Output[Dataset],
    use_streaming: bool = True
):
    """Load large datasets efficiently"""
    import pandas as pd
    
    if use_streaming:
        # Stream large files
        chunks = pd.read_csv(
            data_uri,
            chunksize=10000,
            dtype={'col1': 'int32', 'col2': 'float32'}  # Specify dtypes
        )
        
        # Process in chunks
        processed_chunks = []
        for chunk in chunks:
            processed = process_chunk(chunk)
            processed_chunks.append(processed)
        
        df = pd.concat(processed_chunks, ignore_index=True)
    else:
        df = pd.read_csv(data_uri)
    
    df.to_parquet(output_data.path, compression='snappy')
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: ModelKit Pull Failures

**Symptom**: `kit pull` fails with authentication error

**Solution**:
```bash
# Re-authenticate Docker
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Verify credentials
gcloud auth list

# Check repository permissions
gcloud artifacts repositories get-iam-policy ml-modelkits \
  --location=${REGION}

# Add read permissions if needed
gcloud artifacts repositories add-iam-policy-binding ml-modelkits \
  --location=${REGION} \
  --member="user:your-email@company.com" \
  --role="roles/artifactregistry.reader"
```

#### Issue 2: Pipeline Component Failures

**Symptom**: Component fails with "ModuleNotFoundError"

**Solution**:
```python
# Ensure all dependencies are listed in component decorator
@component(
    base_image="python:3.9",
    packages_to_install=[
        'pandas==2.0.0',
        'scikit-learn==1.3.0',
        'google-cloud-storage==2.10.0'
    ]
)
def fixed_component():
    # Your code here
    pass

# Or use a custom container
@component(
    base_image=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/containers/ml-base:latest"
)
def custom_container_component():
    pass
```

#### Issue 3: Out of Memory Errors

**Symptom**: Pipeline fails with OOM error

**Solution**:
```python
# Increase memory allocation
train_task = train_model_op()
train_task.set_memory_limit('64G')
train_task.set_cpu_limit('16')

# Or process data in chunks
@component
def memory_efficient_processing():
    import pandas as pd
    
    # Process in chunks
    for chunk in pd.read_csv('data.csv', chunksize=10000):
        process(chunk)
        
    # Use memory-efficient dtypes
    df = pd.read_csv('data.csv', dtype={
        'id': 'int32',
        'value': 'float32'
    })
```

#### Issue 4: ModelKit Unpacking Errors

**Symptom**: `kit unpack` fails or extracts incomplete files

**Solution**:
```bash
# Check ModelKit integrity
kit inspect ${MODELKIT_URI}

# Verify all layers
kit inspect ${MODELKIT_URI} --verbose

# Force re-pull
kit remove ${MODELKIT_URI}
kit pull ${MODELKIT_URI}

# Unpack with verbose output
kit unpack ${MODELKIT_URI} -d ./output --verbose
```

#### Issue 5: Vertex AI Permission Errors

**Symptom**: "Permission denied" when running the pipeline

**Solution**:
```bash
# Check service account permissions
gcloud projects get-iam-policy ${PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant required roles
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

### Debugging Techniques

#### 1. Enable Detailed Logging

```python
@component(packages_to_install=['google-cloud-logging'])
def debug_component():
    import logging
    from google.cloud import logging as cloud_logging
    
    # Setup detailed logging
    client = cloud_logging.Client()
    client.setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    logger.debug("Starting component execution")
    logger.info("Processing data")
    logger.warning("Potential issue detected")
    logger.error("Error occurred", exc_info=True)
```

#### 2. Local Testing

```python
# Test components locally before deploying
from kfp.v2 import compiler

# Compile to YAML for inspection
compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path='pipeline.yaml'
)

# Run individual components locally
@component(packages_to_install=['pandas'])
def test_component():
    import pandas as pd
    # Test logic
    return "Success"

# Execute locally
result = test_component()
print(result)
```

#### 3. Pipeline Monitoring

```python
# Monitor pipeline execution
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)

# Get pipeline job
job = aiplatform.PipelineJob.get('projects/.../locations/.../pipelineJobs/...')

# Check status
print(f"State: {job.state}")
print(f"Error: {job.error}")

# Get task details
for task_detail in job.task_details:
    print(f"Task: {task_detail.task_name}")
    print(f"State: {task_detail.state}")
    if task_detail.error:
        print(f"Error: {task_detail.error}")
```

---

## Appendix

### A. Complete Example Project

See the full example project structure at:
`https://github.com/kitops-ml/examples/vertex-ai-integration`

### B. Useful Commands Reference

#### KitOps Commands

```bash
# Pack ModelKit
kit pack <directory> -t <tag>

# Push to registry
kit push <modelkit-uri>

# Pull from registry
kit pull <modelkit-uri>

# Unpack locally
kit unpack <modelkit-uri> -d <output-dir>

# List local ModelKits
kit list

# Inspect ModelKit
kit inspect <modelkit-uri>

# Remove local ModelKit
kit remove <modelkit-uri>

# Tag ModelKit
kit tag <source-uri> <target-uri>

# Sign ModelKit
kit sign <modelkit-uri> --key-file <key-path>

# Verify signature
kit verify <modelkit-uri> --key-file <key-path>
```

#### Vertex AI Commands

```bash
# List pipelines
gcloud ai pipelines list --region=${REGION}

# Get pipeline details
gcloud ai pipelines describe <pipeline-id> --region=${REGION}

# List pipeline jobs
gcloud ai pipeline-jobs list --region=${REGION}

# Cancel pipeline job
gcloud ai pipeline-jobs cancel <job-id> --region=${REGION}

# List models
gcloud ai models list --region=${REGION}

# List endpoints
gcloud ai endpoints list --region=${REGION}

# Deploy model to endpoint
gcloud ai endpoints deploy-model <endpoint-id> \
  --model=<model-id> \
  --region=${REGION}
```

### C. Additional Resources

- **KitOps Documentation**: https://kitops.org/docs
- **KitOps GitHub**: https://github.com/kitops-ml/kitops
- **Vertex AI Documentation**: https://cloud.google.com/vertex-ai/docs
- **Vertex AI Pipelines Guide**: https://cloud.google.com/vertex-ai/docs/pipelines
- **KFP SDK Reference**: https://kubeflow-pipelines.readthedocs.io/
- **CNCF ModelPack Spec**: https://github.com/modelpack/model-spec
- **KitOps Discord**: https://discord.gg/Tapeh8agYy

### D. Glossary

- **ModelKit**: OCI-compliant artifact containing model, code, data, and configs
- **Kitfile**: YAML manifest defining ModelKit contents
- **Pipeline**: Directed acyclic graph (DAG) of ML workflow steps
- **Component**: Self-contained, reusable pipeline task
- **Artifact**: Output from a pipeline component (dataset, model, metrics)
- **OCI**: Open Container Initiative - standard for container formats
- **Vertex AI**: Google Cloud's unified ML platform
- **KFP**: Kubeflow Pipelines - workflow orchestration for ML

---

By combining KitOps' standardized packaging with Vertex AI's powerful pipeline orchestration, teams can build production-grade ML workflows that are:

- **Reproducible**: Immutable ModelKits ensure consistency
- **Collaborative**: Teams share artifacts securely via OCI registries
- **Scalable**: Vertex AI handles infrastructure automatically
- **Auditable**: Complete lineage tracking for compliance
- **Automated**: CI/CD integration enables continuous ML delivery

