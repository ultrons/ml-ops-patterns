import os

PROJECT_ID = 'pytorch-tpu-nfs'
DATASET_ID = 'view_dataset'
TABLE_ID = 'weather_time_series'
REGION = 'us-central1'
DATASET_CSV = 'gs://view-demo-dataset/forecasting/jena_climate_2009_2016.csv'
BASE_IMAGE_URI = f'gcr.io/{PROJECT_ID}/test-custom-container'
BASE_TRAINING_IMAGE = f'gcr.io/{PROJECT_ID}/test-custom-trainer:latest'
STAGING_BUCKET = 'gs://automl-samples'
PIPELINE_ROOT = f'{STAGING_BUCKET}/pipelines/staging'
UTIL_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_ROOT = os.path.dirname(UTIL_DIR)
CONTEXT_WINDOW = 24
EXPERIMENT_PREFIX = 'weather-exp'
TF_SERVING_IMAGE = 'gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-2:latest'
TENSORBOARD_INST = 'view-tensorboard'
