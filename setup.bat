@echo off
echo Setting up environment for KitOps + Vertex AI Pipeline...

echo Installing Python dependencies...
pip install -r requirements.txt
pip install -r ml-project/src/requirements.txt

echo.
echo Checking for Kit CLI...
where kit >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Kit CLI not found. Please install it from https://kitops.ml
) else (
    echo [OK] Kit CLI found.
)

echo.
echo Checking for Google Cloud CLI...
where gcloud >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] gcloud CLI not found. Please install Google Cloud SDK.
) else (
    echo [OK] gcloud CLI found.
)

echo.
echo Setup complete.
echo.
echo To run the pipeline:
echo 1. Set your PROJECT_ID and REGION environment variables.
echo 2. Authenticate with gcloud: gcloud auth login
echo 3. Run: python pipeline.py
pause
