import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class LSTM:
    def __init__(self, features, timesteps=1, quantile=0.5, learning_rate=0.005, dropout_rate=0.1):
        self.features = features
        self.timesteps = timesteps
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _quantile_loss(self, y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * e, (self.quantile - 1) * e))
    
    def _build_model(self):
        inputs = layers.Input(shape=(self.timesteps, self.features))
        
        x = layers.LSTM(2 * self.features, return_sequences=True, dropout=self.dropout_rate)(inputs)
        x = layers.LayerNormalization()(x)
        
        x = layers.LSTM(2 * self.features, return_sequences=False, dropout=self.dropout_rate)(x)
        x = layers.LayerNormalization()(x)
        
        x = layers.Dense(2 * self.features, activation="swish")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                      loss=self._quantile_loss)
        return model
    
    def fit(self, X_train, y_train, epochs=25, batch_size=64, patience=3, validation_split=0.2):
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
