from google.cloud import bigquery
import pandas as pd
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
from tensorflow import feature_column
import os


class trainer:
    def __init__(self,
            project_id='pytorch-tpu-nfs',
            location='us-central1',
            dataset_id='view_dataset',
            table_id='weather_time_series'
            ):
        self.project_id = project_id
        self.location = location
        self.dataset_id = dataset_id
        self.table_id = table_id

    def read_dataset(self):
        sql = f"""
SELECT *
FROM  `{self.project_id}.{self.dataset_id}.{self.table_id}`
"""
        client = bigquery.Client(location=self.location, project=self.project_id)
        query_job = client.query(
          sql,
          # Location must match that of the dataset(s) referenced in the query.
          location=self.location,
        )  # API request - starts the query

        self.df = query_job.to_dataframe()

    def create_split(self):
        date_time = pd.to_datetime(self.df.pop('Date_Time'), format='%d.%m.%Y %H:%M:%S')
        column_indices = {name: i for i, name in enumerate(self.df.columns)}
        n = len(self.df)
        train_df = self.df[0:int(n*0.7)]
        val_df = self.df[int(n*0.7):int(n*0.9)]
        test_df = self.df[int(n*0.9):]
        num_features = self.df.shape[1]
        return train_df, val_df, test_df

    def compile_and_fit(self, model, window, patience=2, max_epoch=1):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])
        tb_log_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR', './tb-logs')
        tb_callback = tf.keras.callbacks.TensorBoard(tb_log_dir, update_freq=1)
        history = model.fit(window.train, epochs=max_epoch,
                            validation_data=window.val,
                            callbacks=[early_stopping, tb_callback])
        return history


if __name__ == '__main__':
   from view_demo.models.baseline import linear
   from view_demo.preprocess.window import Generator as WindowGenerator
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--project-id', dest='project_id',
                       default="pytorch-tpu-nfs", type=str, help='Name of the GCP Project')
   parser.add_argument('--location', dest='location',
                       default="us-central1", type=str, help='GCP Region')
   parser.add_argument('--dataset-id', dest='dataset_id',
                       default="view_dataset", type=str, help='BQ Dataset')
   parser.add_argument('--table-id', dest='table_id',
                       default="weather_time_series", type=str, help='BQ Dataset Table')
   parser.add_argument('--experiment-id', dest='experiment_id',
                       default="weather-pred", type=str, help='Experiment Identifier for tracking purpose')
   parser.add_argument('--staging-bucket', dest='staging_bucket',
                       default=None, type=str, help='GCS Bucket used for staging AI Platform data')
   parser.add_argument('--context-window', dest='context_window',
                       default=24, type=int, help='Width of context to consider fro prediction')
   args = parser.parse_args()


   project_id = args.project_id
   location = args.location
   dataset_id = args.dataset_id
   table_id = args.table_id
   experiment_id = args.experiment_id
   staging_bucket = args.staging_bucket
   context_window = args.context_window
   run_id = f'context-window-{context_window}'

   experiment_tracking_on = staging_bucket is not None
   trainer = trainer(
      project_id,
      location,
      dataset_id,
      table_id
   )
   trainer.read_dataset()
   train_df, val_df, test_df = trainer.create_split()
   wide_window = WindowGenerator(
       input_width=context_window, label_width=context_window, shift=1,
       label_columns=['T__degC'],
       train_df=train_df,
       test_df=test_df,
       val_df=val_df)
   history = trainer.compile_and_fit(linear, wide_window)
   if experiment_tracking_on:
       from google.cloud import aiplatform
       aiplatform.init(
           project=project_id,
           staging_bucket=staging_bucket,
           experiment=experiment_id
       )
       aiplatform.start_run(run=run_id)
       aiplatform.log_metrics({"val_loss": history.history['val_loss'][-1]})
       aiplatform.log_metrics({"val_mae": history.history['val_mean_absolute_error'][-1]})
       aiplatform.log_metrics({"train_loss": history.history['loss'][-1]})
       aiplatform.log_metrics({"train_mae": history.history['mean_absolute_error'][-1]})

   model_path = os.environ.get('AIP_MODEL_DIR', f'{staging_bucket}/{experiment_id}/{run_id}/model/saved_model')
   tf.saved_model.save(linear, model_path)
