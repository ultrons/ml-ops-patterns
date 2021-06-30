import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


class BaseLineModel(tf.keras.Model):

  def __init__(self):
    super(BaseLineModel, self).__init__()
    self.preprocessing_layer = preprocessing.Normalization()
    self.dense1 = tf.keras.layers.Dense(units=1)

  def call(self, inputs):
    x = self.preprocessing_layer(inputs)
    print("Hello!")
    return self.dense1(x)


linear = BaseLineModel()
