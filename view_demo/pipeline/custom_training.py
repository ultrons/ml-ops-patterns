import json
from typing import NamedTuple


from kfp import dsl
from kfp.v2 import compiler
from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    InputPath,
    OutputPath,
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    ClassificationMetrics,
    Metrics,
)

from kfp.v2.google.client import AIPlatformClient
from view_demo.utils import get_project_id
from google_cloud_pipeline_components import aiplatform as gcc_aip

import os

##### Run Parameters
from view_demo.utils import env_vars as evar
PROJECT_ID = evar.PROJECT_ID
DATASET_CSV = evar.DATASET_CSV
REGION = evar.REGION
BASE_IMAGE_URI = evar.BASE_IMAGE_URI
BASE_TRAINING_IMAGE = evar.BASE_TRAINING_IMAGE
SRC_ROOT = evar.SRC_ROOT
PIPELINE_ROOT = evar.PIPELINE_ROOT
STAGING_BUCKET = evar.STAGING_BUCKET
TENSORBOARD_INST = evar.TENSORBOARD_INST
TF_SERVING_IMAGE = evar.TF_SERVING_IMAGE
##### Preprocessing component
# Note: Output component file is commented out
# To allow execution of this code on cloud function where
# Source dir is READONLY
@component(
    base_image=BASE_IMAGE_URI,
    #output_component_file=f'{SRC_ROOT}/preprocess/preprocess.yaml',
)
def view_preprocess(
    project_id: str,
    raw_dataset: str,
    out_dataset: OutputPath(),
):
    from view_demo.preprocess import create_dataset
    bq_path = create_dataset(project_id=project_id, csv_path=raw_dataset)
    with open(out_dataset, 'w') as f:
        f.write(bq_path)

##### Custom Training component
@component(
    base_image=BASE_IMAGE_URI,
  #  output_component_file=f'{SRC_ROOT}/train/train.yaml',
)
def view_train(
    project_id: str,
    input_dataset_path: InputPath(),
    metrics: Output[Metrics],
    model: Output[Model],
    experiment_prefix: str ,
    staging_bucket: str ,
    context_window: int ,
    tensorboard_inst: str

) -> float :
    from google.cloud import aiplatform
    from datetime import datetime
    import logging
    with open(input_dataset_path) as f:
        logging.info(f"input_dataset is: {f.read()}")
    # Create and experiment tag
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_id = experiment_prefix + TIMESTAMP
    run_id = f'context-window-{context_window}'
    # Init AI Platform
    aiplatform.init(
        project=project_id,
        staging_bucket=staging_bucket,
        experiment=experiment_id
    )

    # Define the custom training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name="view-training",
        container_uri= BASE_TRAINING_IMAGE,
        model_serving_container_image_uri=TF_SERVING_IMAGE
    )
    logging.info(f"Type of experiment_id :{type(experiment_id)}")
    logging.info(f"Type of staging_bucket :{type(staging_bucket)}")
    logging.info(f"Type of context_window :{type(context_window)}")
    model_obj = job.run(
        replica_count=1,
        model_display_name="temp-prediction",
        args=[
            f'--experiment-id={experiment_id}',
            f'--staging-bucket={staging_bucket}',
            f'--context-window={context_window}'
        ],
        environment_variables={'AIP_MODEL_DIR': model.uri},
        base_output_dir=os.path.dirname(model.uri)
    )

    metrics_df = aiplatform.get_experiment_df(experiment_id)
    val_mae = metrics_df.loc[metrics_df['run_name'] == run_id]['metric.val_mae'].values[-1]
    val_mae = float(val_mae)
    metrics.log_metric('val_mae', val_mae)
    logging.info(f"Mean Error is:{val_mae}")
    return val_mae

# Showvase component organization
# only depicted for the utility components
from view_demo.utils import complib
complib = complib(BASE_IMAGE_URI, SRC_ROOT)
# Fail Op Component
fail_op = complib.fail_op()
# Component to extract model path(URI) from model object
model_to_uri = complib.model_to_uri()


# Pipeline Definition
@dsl.pipeline(
    name="view-test-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def view_pipeline(
    project_id: str = PROJECT_ID,
    raw_dataset: str = DATASET_CSV,
    staging_bucket: str = STAGING_BUCKET,
    mae_cutoff: float = 5.0,
    model_display_name: str = 'forecast-custom',
    context_window: int = 24,
    experiment_prefix: str = 'weather-prediction-',
    tensorboard_inst: str = TENSORBOARD_INST
):
    preprocess_task = view_preprocess(
        project_id=project_id,
        raw_dataset=raw_dataset
    )
    train_task = view_train(
        project_id=project_id,
        input_dataset=preprocess_task.outputs["out_dataset"],
        context_window=context_window,
        experiment_prefix=experiment_prefix,
        staging_bucket=staging_bucket,
        tensorboard_inst=tensorboard_inst
    )
    with dsl.Condition(train_task.outputs['output'] < mae_cutoff , name="mae_test"):
        get_model_task = model_to_uri(train_task.outputs['model'])
        model_upload_op = gcc_aip.ModelUploadOp(
            project=project_id,
            display_name=model_display_name,
            artifact_uri=get_model_task.outputs['output'],
            serving_container_image_uri=TF_SERVING_IMAGE,
            serving_container_environment_variables={"NOT_USED": "NO_VALUE"},
        )
        #model_upload_op.after(train_task)
        endpoint_create_op = gcc_aip.EndpointCreateOp(
            project=project_id,
            display_name="pipelines-created-endpoint",
        )
        model_deploy_op = gcc_aip.ModelDeployOp(  # noqa: F841
            project=project_id,
            endpoint=endpoint_create_op.outputs["endpoint"],
            model=model_upload_op.outputs["model"],
            deployed_model_display_name=model_display_name,
            machine_type="n1-standard-4",
        )
    with dsl.Condition(train_task.outputs['output'] > mae_cutoff , name="Low_Quality"):
        fail_task = fail_op()


if __name__ == '__main__':
    # Compile Pipeline
    from kfp.v2 import compiler as v2compiler
    v2compiler.Compiler().compile(pipeline_func=view_pipeline,
                                  package_path='view_pipeline_spec.json')

    from kfp.v2.google.client import AIPlatformClient  # noqa: F811

    api_client = AIPlatformClient(
        project_id=PROJECT_ID,
        region=REGION,
        )

    result = api_client.create_run_from_job_spec(
        job_spec_path="view_pipeline_spec.json",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=True,
        parameter_values={
            "mae_cutoff":3.1
        },
    )
    from view_demo.utils.check_pipeline_status import check_pipeline_status
    check_pipeline_status(api_client, result)
