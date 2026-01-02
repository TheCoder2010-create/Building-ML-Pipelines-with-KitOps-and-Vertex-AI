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



Additional Resources

- **KitOps Documentation**: https://kitops.org/docs
- **KitOps GitHub**: https://github.com/kitops-ml/kitops
- **Vertex AI Documentation**: https://cloud.google.com/vertex-ai/docs
- **Vertex AI Pipelines Guide**: https://cloud.google.com/vertex-ai/docs/pipelines
- **KFP SDK Reference**: https://kubeflow-pipelines.readthedocs.io/
- **CNCF ModelPack Spec**: https://github.com/modelpack/model-spec
- **KitOps Discord**: https://discord.gg/Tapeh8agYy

. Glossary

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
