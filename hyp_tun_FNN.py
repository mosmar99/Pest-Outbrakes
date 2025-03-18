import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datamodule import datamodule
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import optuna
from FNN_model import FNNRegressor_s  
import time

np.random.seed(42)
torch.manual_seed(42)

#generate splits
dm = datamodule(datamodule.HOSTVETE)  #Options: Varkorn, Hostvete, Ragvete, look in datamodule to find all pests to train on
dm.default_process(target='Svartpricksjuka')
splits = dm.CV_test_train_split(n_folds=10, random_state=42)
logger = CSVLogger("logs")

dm.test_train_split(random_state=42)
X_train_ht, X_test_ht, y_train_ht, y_test_ht = dm.get_test_train()

def objective(trial):
    #hyperparameters:
    loss_fn_name = trial.suggest_categorical("loss_fn", ["huber", "mse", "l1", "smoothl1"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "silu", "leaky_relu"])
    
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_sizes = []
    for i in range(n_layers):
        if i == 0:
            size = trial.suggest_int(f"hidden_size_{i}", 128, 512, step=64)
        else:
            size = trial.suggest_int(f"hidden_size_{i}", 32, 256, step=32)
        hidden_sizes.append(size)
    
    #only use one split to find params, might not be the best way, but time efficient, will use these params on all folds later
    X_train, X_val, y_train, y_val = splits[0]
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values.ravel(), dtype=torch.float)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val.values.ravel(), dtype=torch.float)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=11)
    
    input_size = X_train_tensor.shape[1]
    #initialize the model with hyperparameters
    model = FNNRegressor_s(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        activation=activation,
        dropout=dropout_rate,
        loss_fn_name=loss_fn_name
    )
    
    #using fewer epochs during tuning for speed
    trainer = pl.Trainer(
        max_epochs=50,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)
    
    #evaluate on the validation set.
    model.eval()
    y_pred_list = []
    for batch in val_loader:
        X_batch, _ = batch
        with torch.no_grad():
            preds_batch = model(X_batch)
        y_pred_list.append(preds_batch.cpu().numpy())
    y_pred = np.concatenate(y_pred_list)
    
    #compute MSE on the validation set, most important
    val_loss = mean_squared_error(y_val_tensor.numpy(), y_pred)
    return val_loss


start_time = time.time()
#create optuna study and optimize the objective.
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  #feel free to increase n_trials

print("Best trial:")
print(study.best_trial)
best_params = study.best_trial.params

#use the best hyperparameters on full dataset to see performance
all_preds = []
all_tests = []
n_layers_best = best_params["n_layers"]
best_hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(n_layers_best)]

i = 0

for X_train, X_test, y_train, y_test in splits:
    print('Run:', i + 1, 'on all folds.')
    i += 1
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values.ravel(), dtype=torch.float)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values.ravel(), dtype=torch.float)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    input_size = X_train_tensor.shape[1]
    model = FNNRegressor_s(
        input_size=input_size,
        hidden_sizes=best_hidden_sizes,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        activation=best_params["activation"],
        dropout=best_params["dropout"],
        loss_fn_name=best_params["loss_fn"]
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader)
    
    #evaluate
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
print(f"Training time: {total_time:.4f} minutes")

preds = pd.concat(all_preds)
tests = pd.concat(all_tests)

#print metrics
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
