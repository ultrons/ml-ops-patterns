import tensorflow as tf

def compile_and_fit(model, window, patience=2, max_epoch=2):
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
