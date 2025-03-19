import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

class EnsembleNN:
    def __init__(self, input_dim, num_models=5, learning_rate=0.001, validation_split=0.2):
        self.input_dim = input_dim
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.models = []

    def build_base_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse',
                      metrics=['mae'])
        return model

    def fit(self, x_train, y_train, epochs=100, batch_size=256):
        self.models = []
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.validation_split, random_state=42)
        
        for i in range(self.num_models):
            print(f"Training model {i+1}/{self.num_models}")
            model = self.build_base_model()
            model.fit(x_train, y_train, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      validation_data=(x_val, y_val),
                      verbose=0)
            self.models.append(model)

    def predict(self, x_test):
        if not self.models:
            raise ValueError("No models trained. Call fit() first.")
        predictions = np.column_stack([model.predict(x_test, verbose=0) for model in self.models])
        return np.mean(predictions, axis=1)

# Example Usage:
# ensemble = EnsembleNN(input_dim=10, num_models=5)
# ensemble.fit(x_train, y_train)
# predictions = ensemble.predict(x_test) 
# print(predictions)
