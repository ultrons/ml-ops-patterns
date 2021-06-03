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
# Showvase component organization
# only depicted for the utility components
from view_demo.pipeline.custom_training import *

def trigger_pipeline(event, context):
    import tempfile
    tmpdir = tempfile.gettempdir()
    # Compile Pipeline
    from kfp.v2 import compiler as v2compiler
    v2compiler.Compiler().compile(pipeline_func=view_pipeline,
                                  package_path=f'{tmpdir}/view_pipeline_spec.json')

    from kfp.v2.google.client import AIPlatformClient  # noqa: F811

    api_client = AIPlatformClient(
        project_id=PROJECT_ID,
        region=REGION,
        )
    bucket_name = event['bucket']
    file_name = event['name']
    dataset_gcs_path = f'gs://{bucket_name}/{file_name}'

    result = api_client.create_run_from_job_spec(
        job_spec_path=f'{tmpdir}/view_pipeline_spec.json',
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={
            "raw_dataset": dataset_gcs_path
        },
    )
    #from view_demo.utils.check_pipeline_status import check_pipeline_status
    #check_pipeline_status(api_client, result)
