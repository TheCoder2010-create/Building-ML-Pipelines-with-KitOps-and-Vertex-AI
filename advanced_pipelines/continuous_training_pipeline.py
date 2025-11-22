from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

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

@component
def unpack_modelkit_op(modelkit_uri: str) -> dict:
    # Placeholder
    return {'output_path': '/tmp/unpacked'}

@component
def load_new_data_op(bucket_name: str, data_path: Output[Dataset]):
    # Placeholder
    pass

@component
def merge_datasets_op(
    existing_data: str,
    new_data: Input[Dataset],
    merged_data: Output[Dataset]
):
    # Placeholder
    pass

@component
def train_model_op(
    data_path: Input[Dataset],
    model_output: Output[Model],
    metrics: Output[Metrics]
):
    # Placeholder
    pass

@component
def deploy_model_op(
    project_id: str,
    model_path: Input[Model]
):
    # Placeholder
    pass

@component
def pack_new_modelkit_op(
    model_path: Input[Model],
    version: str
):
    # Placeholder
    pass

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
