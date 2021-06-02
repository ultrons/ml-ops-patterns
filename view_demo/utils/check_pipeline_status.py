from typing import Dict
from kfp.v2.google.client import AIPlatformClient
def check_pipeline_status(
        api_client: AIPlatformClient,
        result: Dict
    ):
    """
    Polls the status of last executed pipeline run.
    Raise ValueError in case the run fails.
    """
    import time
    while True:
        pipeline_run_status = api_client.get_job(result['name'].split('/')[-1])['state']
        if pipeline_run_status == 'PIPELINE_STATE_FAILED':
            raise ValueError("Pipeline Run Failled")
        elif pipeline_run_status == 'PIPELINE_STATE_SUCCEEDED':
            print("Pipeline Run Finished")
            break
        else:
            time.sleep(60)
