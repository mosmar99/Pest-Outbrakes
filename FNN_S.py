import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datamodule import datamodule
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from pytorch_lightning.loggers import CSVLogger

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from FNN_model import FNNRegressor_s

field_id = 1807

dm = datamodule(datamodule.HOSTVETE) #RAGVETE, VARKORN, HOSTVETE
dm.default_process(target='Svartpricksjuka') #Svartpricksjuka
splits = dm.CV_test_train_split()
logger = CSVLogger("logs")

all_preds = []
all_tests = []

print(len(splits))

for X_train, X_test, y_train, y_test in splits:
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values.ravel(), dtype=torch.float)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values.ravel(), dtype=torch.float)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    input_size = X_train_tensor.shape[1]
    model = FNNRegressor_s(input_size=input_size)
    
    #
    trainer = pl.Trainer(
        max_epochs=100,    #you can lower the epochs if its slow
        logger=logger,
        enable_checkpointing=False,
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
    
    #original scale
    y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
    y_test_df = pd.DataFrame(y_test.values, index=X_test.index)
    
    y_pred_df = dm.inverse_scale(y_pred_df)
    y_test_df = dm.inverse_scale(y_test_df)
    
    all_preds.append(y_pred_df)
    all_tests.append(y_test_df)


preds = pd.concat(all_preds)
tests = pd.concat(all_tests)


plt.figure(figsize=(10, 5))
plt.plot(range(400), preds.head(400), label='y_pred')
plt.plot(range(400), tests.head(400), label='y_actual')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.scatter(preds, tests, s=10, alpha=0.8, lw=1.5)
plt.xlabel('Prediction (Varde)')
plt.ylabel('Actual (Varde)')
plt.show()


mae = mean_absolute_error(tests, preds)
mse = mean_squared_error(tests, preds)
r2 = r2_score(tests, preds)
print('TEST:  MAE:', mae, 'MSE:', mse, 'R2:', r2)

"""
# Plot correlation heatmap from the original geodataframe in the datamodule
corr_matrix = dm.data_gdf[['target', 'Nederbördsmängd_sum_year_cumulative']].corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Blue to red colormap

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,      # Show correlation values
            cmap=cmap,
            center=0,        # Center the colormap at 0
            vmin=-1, vmax=1,  # Set the range of the colormap
            square=True,     # Make the plot square
            fmt='.2f',       # Format the numbers to 2 decimal places
            linewidths=0.5,  # Width of the lines between cells
            cbar_kws={'shrink': .5, 'label': 'Correlation'})
plt.title('Linear Correlation to next week (target)', pad=20)
plt.tight_layout()
plt.show()
"""