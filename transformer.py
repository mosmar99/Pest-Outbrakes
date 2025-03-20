import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class Transformer:
    def __init__(self, features, d_model=64, num_heads=4, ff_dim=128, num_transformer_blocks=3, quantile=0.5, learning_rate=0.008):
        self.features = features
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _quantile_loss(self, y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * e, (self.quantile - 1) * e))
    
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.4):
            super().__init__()
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.norm1 = layers.LayerNormalization()
            self.norm2 = layers.LayerNormalization()
            self.ffn = keras.Sequential([
                layers.Dense(ff_dim, activation="swish"),
                layers.Dense(embed_dim),
            ])
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dropout2 = layers.Dropout(dropout_rate)

        def call(self, inputs, training=False):
            attn_output = self.attention(inputs, inputs, inputs)  
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.norm1(inputs + attn_output)

            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.norm2(out1 + ffn_output)
    
    class PositionEncoding(layers.Layer):
        def __init__(self, max_len, d_model):
            super().__init__()
            self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=d_model)
        
        def call(self, x):
            positions = tf.range(start=0, limit=x.shape[1], delta=1)
            positions = self.position_embedding(positions)
            return x + positions
    
    def _build_model(self):
        inputs = layers.Input(shape=(self.features,))
        x = layers.Dense(self.d_model, activation="swish")(inputs)
        x = layers.Reshape((1, self.d_model))(x)
        pos_encoding = self.PositionEncoding(self.features, self.d_model)(x)
        
        for _ in range(self.num_transformer_blocks):
            x = self.TransformerBlock(embed_dim=self.d_model, num_heads=self.num_heads, ff_dim=self.ff_dim)(pos_encoding)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=self._quantile_loss)
        return model
    
    def fit(self, X_train, y_train, epochs=12, batch_size=64, patience=3, validation_split=0.2):
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