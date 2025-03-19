import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datamodule import datamodule
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, root_mean_squared_error
from pytorch_lightning.loggers import CSVLogger
import time
import torch
import pytorch_lightning as pl
from jbv_models.fnn.model import FNNRegressor
from jbv_models.fnn.datamodule import JBVDataModule

np.random.seed(42)
torch.manual_seed(42)

dm = datamodule(datamodule.HOSTVETE) #Options: Varkorn, Hostvete, Ragvete, look in datamodule to find all pests to predict on
dm.default_process(target='Svartpricksjuka') #Svartpricksjuka
splits = dm.CV_test_train_split(random_state=42)
logger = CSVLogger("logs")

all_preds = []
all_tests = []

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

start_time = time.time()
for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    print('Run:', i + 1)  
    data = JBVDataModule(X_train, y_train, X_test, y_test,batch_size=32,num_workers=11,val_size=0.2, random_state=42)
    
    model = FNNRegressor(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes_best,
        learning_rate=best_learning_rate,
        weight_decay=best_weight_decay,
        activation=best_activation,
        dropout=best_dropout,
        loss_fn_name=loss_fn_name
        )
    
    trainer = pl.Trainer(max_epochs=100, logger=logger, enable_checkpointing=False)
    
    #train the model
    trainer.fit(model, data)
    
    #test the model
    trainer.test(model, data)

    #predict, will use the test data, if no data has been provided specifically to predict on
    predictions = trainer.predict(model, data)
    #convert to numpy
    y_pred = torch.cat(predictions).cpu().numpy()

    #convert and inverse scale the predictions for plot and final metrics
    y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
    y_test_df = pd.DataFrame(y_test.values, index=X_test.index)
    
    y_pred_df = dm.inverse_scale(y_pred_df)
    y_test_df = dm.inverse_scale(y_test_df)
    
    all_preds.append(y_pred_df)
    all_tests.append(y_test_df)


end_time = time.time()
total_time = (end_time - start_time) / 60
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
