import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datamodule import datamodule
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import optuna
from jbv_models.fnn.model import FNNRegressor  
import time
from jbv_models.fnn.datamodule import JBVDataModule
from optunapruning import CustomOptunaPruningCallback

N_TRIALS = 50
N_EPOCHS_OPTUNA = 50
N_EPOCHS_TRAIN = 100
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

#generate splits
dm = datamodule(datamodule.HOSTVETE)  #Options: Varkorn, Hostvete, Ragvete, look in datamodule to find all pests to train on
dm.default_process(target='Svartpricksjuka')
splits = dm.CV_test_train_split(n_folds=10, random_state=RANDOM_STATE)
logger = CSVLogger("logs")

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
    #dont use X_test and y_test
    X_train, X_val, y_train, y_val = splits[0]

    #use the 
    trial_data = JBVDataModule(X_train, y_train, X_test=X_val, y_test=y_val, batch_size=16, num_workers=11, val_size=0.2, random_state=RANDOM_STATE)
    
    input_size = X_train.shape[1]
    #initialize the model with hyperparameters
    model = FNNRegressor(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        activation=activation,
        dropout=dropout_rate,
        loss_fn_name=loss_fn_name
    )
    
    #using fewer epochs during tuning for speed
    #trainer = pl.Trainer(max_epochs=N_EPOCHS_OPTUNA, logger=False, enable_checkpointing=False)
    callbacks = []
    callbacks.append(CustomOptunaPruningCallback(trial, monitor="val_mse"))

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS_OPTUNA,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks
    )
    
    trainer.fit(model, trial_data)
    
    #evaluate on the validation set, dont use the training set here
    val_results = trainer.validate(model, datamodule=trial_data)
    
    #minimize MSE on the validation set, most important
    #val_loss = val_results[0]["val_loss"]
    #return val_loss
    val_mse = val_results[0]["val_mse"]
    return val_mse


start_time = time.time()
#create optuna study and optimize the objective.
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,   # dont prune the first 5 trials
        n_warmup_steps=10,     # dont prune the first 5 epochs in each trial
        interval_steps=1      # check for pruning every epoch after warmup
    )
)
study.optimize(objective, n_trials=N_TRIALS)  #feel free to increase n_trials

print("Best trial:")
print(study.best_trial)
best_params = study.best_trial.params

#use the best hyperparameters on full dataset to see performance
all_preds = []
all_tests = []
n_layers_best = best_params["n_layers"]
best_hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(n_layers_best)]

######### THIS WHOLE SECTION OF TRAINING THE MODEL ON BEST PARAMS, COULD BE REMOVE AND ONLY DONE IN train_fnn.py #############

for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    print('Run:', i + 1, 'on all folds.')

    full_data = JBVDataModule(
        X_train, y_train,
        X_test, y_test,
        batch_size=32, num_workers=11, val_size=0.2, random_state=RANDOM_STATE
    )
    
    input_size = X_train.shape[1]
    model = FNNRegressor(
        input_size=input_size,
        hidden_sizes=best_hidden_sizes,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        activation=best_params["activation"],
        dropout=best_params["dropout"],
        loss_fn_name=best_params["loss_fn"]
    )
    
    trainer = pl.Trainer(max_epochs=N_EPOCHS_TRAIN, logger=logger, enable_checkpointing=False)
    trainer.fit(model, full_data)
    
    #not really needed right now, but could be nice to look at the metrics, altough they are scaled down
    trainer.test(model, full_data)
    
    #predictions will be on the test set, as we dont have a 
    predictions = trainer.predict(model, full_data)
    y_pred = torch.cat(predictions).cpu().numpy()

    #inverse scale the predictions for plot and metrics later on
    y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
    y_test_df = pd.DataFrame(y_test.values, index=X_test.index)
    y_pred_df = dm.inverse_scale(y_pred_df)
    y_test_df = dm.inverse_scale(y_test_df)
    
    all_preds.append(y_pred_df)
    all_tests.append(y_test_df)

end_time = time.time()
total_time = (end_time - start_time) / 60
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
