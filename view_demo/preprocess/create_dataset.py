import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

import tempfile
import argparse
import sys
import os

from view_demo.utils import get_project_id

def create_dataset(
    csv_path='gs://bench-datasets/jena_climate_2009_2016.csv',
    project_id=get_project_id(),
    dataset_id='view_dataset',
    table_id='weather_time_series'
):
    df = pd.read_csv(csv_path)

    # Convert to hourly dataset
    # slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]


    # Clean Data
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame
    df['wv (m/s)'].min()


    # Rename Columns to comply with BQ
    df.rename(columns={
        'p (mbar)': 'p__mbar',
        'T (degC)': 'T__degC',
        'Tpot (K)': 'Tpot__K',
        'Tdew (degC)': 'Tdew__degC',
        'rh (%)': 'rh__percent',
        'VPmax (mbar)': 'VPmax__mbar' ,
        'VPact (mbar)': 'VPact__mbar',
        'VPdef (mbar)': 'VPdef__mbar',
        'sh (g/kg)': 'sh__g_per_kg',
        'H2OC (mmol/mol)': 'H2OC__mmol_per_mol',
        'rho (g/m**3)': 'rho__gm_per_cubic_m',
        'max Wx': 'max_Wx',
        'max Wy': 'max_Wy',
        'Day sin': 'Day_sin',
        'Day cos': 'Day_cos',
        'Year sin': 'Year_sin',
        'Year cos': 'Year_cos',
        'Date Time': 'Date_Time',
        'wv (m/s)' : 'wv__m_per_s',
        'max. wv (m/s)': 'max_w__vm_per_s',
        'wd (deg)': 'wd__deg'

    }, inplace=True)

    # Write to BQ
    client = bigquery.Client(location="us-central1", project=project_id)
    print("Client creating using default project: {}".format(client.project))

    try:
        dataset = client.get_dataset(dataset_id)  # Make an API request.
        print("Dataset {} already exists".format(dataset_id))
    except NotFound:
        print("Dataset {} is not found, Creating..".format(dataset_id))
        dataset = client.create_dataset(dataset_id)

    table_ref = dataset.table(table_id)

    job_config = bigquery.LoadJobConfig(
        destination_table_description=table_ref,
        autodetect=True,
    )
    # Overwrite the table if already exists
    job_config.write_disposition = 'WRITE_TRUNCATE'

    job = client.load_table_from_dataframe(df, table_ref, location="us-central1")
    job.result()  # Waits for table load to complete.
    print("Loaded dataframe to {}".format(table_ref.path))

    return table_ref.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', dest='csv_path',
                        default="None", type=str, help='GCS Path to the raw dataset CSV')
    parser.add_argument('--project-id', dest='project_id',
                        default=get_project_id(), type=str, help='Project ID')
    parser.add_argument('--dataset-id', dest='dataset_id',
                        default="view_dataset", type=str, help='Name of the BQ Dataset where preprocessed data will be pushed')
    parser.add_argument('--table-id', dest='table_id',
                        default="weather_time_series", type=str, help='Name of the table under'
                        'the BQ Dataset where preprocessed data will be pushed')
    args = parser.parse_args()


    # Read the dataset into a dataframe
    csv_path = args.csv_path
    dataset_id = args.dataset_id
    table_id = args.table_id
    project_id = args.project_id
    create_dataset(csv_path, project_id, dataset_id ,table_id)

