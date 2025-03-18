import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datamodule import datamodule
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, root_mean_squared_error
from pytorch_lightning.loggers import CSVLogger
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from FNN_model import FNNRegressor_s

np.random.seed(42)
torch.manual_seed(42)

dm = datamodule(datamodule.HOSTVETE) #Options: Varkorn, Hostvete, Ragvete, look in datamodule to find all pests to predict on
dm.default_process(target='Svartpricksjuka') #Svartpricksjuka
splits = dm.CV_test_train_split(random_state=42)
logger = CSVLogger("logs")

all_preds = []
all_tests = []

print(len(splits))

#Best hyperparameters from the Optuna trial:
best_learning_rate = 0.00023492087942582194
best_weight_decay = 8.676935089930829e-06
best_dropout = 0.07468054507323517
best_activation = 'leaky_relu'
hidden_sizes_best = [192, 224]
loss_fn_name="huber"

##### SETTINGS ABOVE BUT WITH THESE HIDDEN LAYER DIMENSIOS: 
# [128, 64] = MAE: 4.65033894931673 MSE: 63.59603440213558 R2: 0.8429499350624815
# [256, 128] = MAE: 4.658657738137636 MSE: 63.25179211591077 R2: 0.8438000395369836

i = 0

start_time = time.time()
for X_train, X_test, y_train, y_test in splits:
    print('Run:', i + 1)
    i += 1  
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values.ravel(), dtype=torch.float)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values.ravel(), dtype=torch.float)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=11)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=11)
    
    input_size = X_train_tensor.shape[1]
    model = FNNRegressor_s(
        input_size=input_size,
        hidden_sizes=hidden_sizes_best,
        learning_rate=best_learning_rate,
        weight_decay=best_weight_decay,
        activation=best_activation,
        dropout=best_dropout,
        loss_fn_name=loss_fn_name
        )
    
    trainer = pl.Trainer(
        max_epochs=100,    #you can lower the epochs if its slow
        logger=logger,
        enable_checkpointing=False
        )
    
    #train the model
    trainer.fit(model, train_loader)
    
    #evaluate on test split
    model.eval()
    y_pred_list = []
    for batch in test_loader:
        X_batch, _ = batch
        with torch.no_grad():
            preds_batch = model(X_batch)
        y_pred_list.append(preds_batch.cpu().numpy())
    y_pred = np.concatenate(y_pred_list)
    
    y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
    y_test_df = pd.DataFrame(y_test.values, index=X_test.index)
    
    #inverse scale the predictions for plot and metrics later on
    y_pred_df = dm.inverse_scale(y_pred_df)
    y_test_df = dm.inverse_scale(y_test_df)
    
    all_preds.append(y_pred_df)
    all_tests.append(y_test_df)


end_time = time.time()
total_time = end_time - start_time / 60
print(f"Total time: {total_time:.4f} minutes")

preds = pd.concat(all_preds)
tests = pd.concat(all_tests)

#print metrics
mae = mean_absolute_error(tests, preds)
mse = mean_squared_error(tests, preds)
rmse = root_mean_squared_error(tests, preds)
r2 = r2_score(tests, preds)
print('TEST:  MAE:', mae, 'MSE:', mse, 'R2:', r2)

#vizualize results
plt.figure(figsize=(10, 5))
plt.plot(range(400), preds.head(400), label='y_predicted')
plt.plot(range(400), tests.head(400), label='y_actual')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(tests, preds, s=10, alpha=0.8, lw=1.5)
plt.xlabel('Actual (Varde)')
plt.ylabel('Prediction (Varde)')
plt.show()
