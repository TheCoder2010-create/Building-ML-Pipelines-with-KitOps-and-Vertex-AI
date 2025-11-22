from kfp.v2.dsl import component, Input, Output, Dataset, Model

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
    # stats = {
    #     'rows_processed': len(df),
    #     'features_scaled': len(features),
    #     'missing_values_handled': missing_count
    # }
    
    return {}

@component(packages_to_install=['google-cloud-logging'])
def robust_training_op(
    data_path: str,
    model_output: Output[Model]
):
    """Training component with comprehensive error handling"""
    from google.cloud import logging as cloud_logging
    import logging
    import os
    import pandas as pd
    import pickle
    
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
        # model = train_model(df)
        logger.info("Training completed successfully")
        
        # Save model
        # with open(model_output.path, 'wb') as f:
        #     pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_output.path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

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
    # data = fetch_data_securely(api_key)
    
    # Never log secrets!
    print("Data fetched successfully")
