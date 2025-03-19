import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class FFNN:
    def __init__(self, input_shape, quantile=0.55, learning_rate=0.001, dropout_rate=0.1):
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self._build_model(input_shape)

    def _log_cosh_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))
    
    def _quantile_loss(self, y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * e, (self.quantile - 1) * e))
    
    def _build_model(self, input_shape):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        
        layer_sizes = [156, 26, 130, 26, 26, 78]
        
        for size in layer_sizes:
            model.add(layers.Dense(size, activation='swish'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=self._log_cosh_loss)
        return model
    
    def fit(self, X_train, y_train, epochs=12, batch_size=32, patience=3, validation_split=0.2):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, callbacks=[early_stopping], verbose=1)
        return history
    
    def predict(self, X_test):
        activations = self.model.predict(X_test).flatten()
        return np.clip(activations, 0, 1)
    
    def evaluate(self, y_test, y_pred):
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
