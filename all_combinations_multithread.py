import os
import tensorflow as tf
from datamodule import datamodule
from transformer import Transformer
from lstm import LSTM
from FFNN import FFNN
from xgboost import XGBRegressor 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import multiprocessing
import concurrent.futures
import threading

# Set start method before importing anything else
# multiprocessing.set_start_method('fork', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set CPU thread management
# os.environ['OMP_NUM_THREADS'] = '6'
# os.environ['TF_NUM_INTEROP_THREADS'] = '6'
# os.environ['TF_NUM_INTRAOP_THREADS'] = '6'

# TensorFlow config for thread management
# tf.config.threading.set_inter_op_parallelism_threads(6)
# tf.config.threading.set_intra_op_parallelism_threads(6)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

# Function to calculate evaluation metrics
def eval_metrics(y_true, y_pred):
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    return r2, mae, mse

lock = threading.Lock()
# Initialize storage for results
model_results = {model: {'r2': 0, 'mae': 0, 'mse': 0, 'count': 0} for model in 
                 ['HistGradientBoostingRegressor', 'DecisionTreeRegressor', 'XGBRegressor', 'FFNN', 'SVR']}

crops = [datamodule.HOSTVETE, datamodule.VARKORN, datamodule.RAGVETE]
pests = {datamodule.HOSTVETE: ['Bladfläcksvampar', 'Brunrost', 'Svartpricksjuka','Gulrost', 'Mjöldagg'],
         datamodule.VARKORN: ['Sköldfläcksjuka', 'Kornets bladfläcksjuka', 'Mjöldagg'],
         datamodule.RAGVETE: ['Brunrost', 'Gulrost', 'Sköldfläcksjuka', 'Mjöldagg', 'Bladfläcksvampar']}

def process_crop_pest(crop, pest):
    # Load data
    dm = datamodule(crop)
    dm.default_process(target=pest)
    splits = dm.CV_test_train_split(n_folds=10)

    # Store predictions and actuals
    actuals = []
    predictions = {model: [] for model in [
        'HistGradientBoostingRegressor', 'DecisionTreeRegressor', 'XGBRegressor',
        'FFNN', 'SVR'
    ]}

    for X_train, X_test, y_train, y_test in splits:
        timesteps, features = 1, X_train.shape[1]
        X_train_lstm = np.array(X_train).reshape(-1, timesteps, features)
        X_test_lstm = np.array(X_test).reshape(-1, timesteps, features)

        # Initialize models
        models = {
            'FFNN': FFNN(input_shape=X_train.shape[1]),
            'SVR': SVR(kernel='rbf', C=10, gamma='auto', epsilon=0.01),
            'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'XGBRegressor': XGBRegressor()
        }

        # Train models
        for model_name, model in models.items():
            model.fit(X_train, y_train.squeeze())

        # Make predictions
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            predictions[model_name].append(pd.DataFrame(y_pred, index=X_test.index))

        actuals.append(y_test)

    # Aggregate results
    actuals = pd.concat(actuals)
    actuals = dm.inverse_scale(actuals)

    results = {model: {'r2': 0, 'mae': 0, 'mse': 0, 'count': 0, 'Data_size': len(dm.data_gdf)} for model in predictions.keys()}

    for model_name in predictions.keys():
        pred = pd.concat(predictions[model_name])
        pred = dm.inverse_scale(pred)
        r2, mae, mse = eval_metrics(actuals, pred)

        results[model_name]['r2'] = r2
        results[model_name]['mae'] = mae
        results[model_name]['mse'] = mse
        results[model_name]['count'] = 1

    return results, crop, pest  # ✅ Return results instead of modifying global variables

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for crop in crops:
            for pest in pests[crop]:
                futures.append(executor.submit(process_crop_pest, crop, pest))

        # Collect results after all processes complete
        results = {f'{future.result()[1]}_{future.result()[2]}':future.result()[0] for future in concurrent.futures.as_completed(futures)}

    save_object(results, os.path.join('results','results.pkl'))

    print(results)

    # Convert results into a DataFrame
    records = []
    for disease, models in results.items():
        for model, metrics in models.items():
            records.append({
                "Model": model,
                "Crop Pest Combination": disease,
                "R2": metrics.get("r2", ""),
                "MAE": metrics.get("mae", ""),
                "MSE": metrics.get("mse", ""),
                "Count": metrics.get("count", ""),
                "Data_size": metrics.get("Data_size", ""),
            })

    df = pd.DataFrame(records)

    df.to_csv(os.path.join('results', "model_results.csv"), index=False, encoding='utf-8')
    df.to_string(buf=open(os.path.join('results', "model_results.log"), "w", encoding='utf-8'))
    print(df)

    average_results = df.drop(['Crop Pest Combination', 'Count', 'Data_size'], axis=1).groupby('Model').mean().reset_index()
    average_results = average_results.sort_values(by='MSE', ascending=True)
    print(average_results)

    average_results.to_csv(os.path.join('results', "average_model_results.csv"), index=False, encoding='utf-8')
    average_results.to_string(buf=open(os.path.join('results', "average_model_results.log"), "w", encoding='utf-8'))