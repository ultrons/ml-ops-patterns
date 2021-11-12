from google.cloud import bigquery

project_id = 'pytorch-tpu-nfs'
dataset_id = 'view_dataset'
table_id = 'weather_time_series_named'
location = 'us-central1'
staging_bucket = 'automl-samples'
experiment_prefix = 'pytorch-forecasting'

sql = f"""
SELECT *
FROM  `{project_id}.{dataset_id}.{table_id}`
LIMIT 122
"""
client = bigquery.Client(location=location, project=project_id)
query_job = client.query(
  sql,
  # Location must match that of the dataset(s) referenced in the query.
  location=location,
)  # API request - starts the query

data = query_job.to_dataframe()
data = data.drop(columns=['Date_Time'])
sample = data.to_dict()
test_data = {'data': sample}


import json

test_data_json = json.dumps({"instances": [test_data]})
model_name='pt_tmp_forecast'
import requests
headers = {"content-type": "application/json"}
for _ in range(1):
    json_response = requests.post(f'http://localhost:7080/predictions/{model_name}', data=test_data_json, headers=headers)
    print(json_response.text)

