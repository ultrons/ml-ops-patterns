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

    def compile_and_fit(self, model, window, patience=2, max_epoch=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=max_epoch,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history


if __name__ == '__main__':
   from view_demo.models.baseline import linear
   from view_demo.preprocess.window import Generator as WindowGenerator

   trainer = trainer()
   trainer.read_dataset()
   train_df, val_df, test_df = trainer.create_split()
   wide_window = WindowGenerator(
       input_width=24, label_width=24, shift=1,
       label_columns=['T__degC'],
       train_df=train_df,
       test_df=test_df,
       val_df=val_df)
   history = trainer.compile_and_fit(linear, wide_window)
   tf.saved_model.save(linear, 'gs://automl-samples/models/t_forecase_baseline')
