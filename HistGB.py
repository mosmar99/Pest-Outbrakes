import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os

from datamodule import datamodule

from sklearn.ensemble import RandomForestRegressor as model
# from sklearn.ensemble import GradientBoostingRegressor as model
# from pygam import LinearGAM as model
# from sklearn.gaussian_process import GaussianProcessClassifier as classifier_model
# from sklearn.ensemble import HistGradientBoostingRegressor as model
# from sklearn.ensemble import HistGradientBoostingClassifier as classifier_model
# import umap.umap_ as umap
# from sklearn.neural_network import MLPRegressor as model
# from sklearn.neural_network import MLPClassifier as classifier_model
# from xgboost import XGBRegressor as model
# from xgboost import XGBClassifier as classifier_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import data_processing.jbv_process as jbv_process
# from shapely import Point

field_id = 1807

dm = datamodule(datamodule.HOSTVETE)
dm.default_process(target='Svartpricksjuka')
splits = dm.CV_test_train_split()

preds = []
tests = []
for X_train, X_test, y_train, y_test in splits:
    ml_model = model()
    ml_model.fit(X_train, y_train.squeeze())

    y_pred = pd.DataFrame(ml_model.predict(X_test), index=X_test.index)

    y_pred = dm.inverse_scale(y_pred)
    y_test = dm.inverse_scale(y_test)

    preds.append(y_pred)
    tests.append(y_test)

preds = pd.concat(preds)
tests = pd.concat(tests)

# Field_mask = dm.data_gdf['Series_id'].reindex_like(preds) == field_id
# plt.scatter(y_pred, y_test, label='y_pred')
plt.plot(range(200), preds.head(200), label='y_pred')
plt.plot(range(200), tests.head(200), label='y_actual')

plt.legend()
plt.show()

# plt.figure(figsize=(16,10))
# plt.plot(range(len(y_test)),y_test, color='blue', zorder=1, lw=2, label='TRUE')
# plt.plot(range(len(y_pred)),y_pred, color='cyan', lw=2, label='PREDICTED')
# plt.show()

plt.scatter(preds, tests, s=10, alpha=0.8, lw=1.5)
plt.xlabel('Prediction (Varde)')
plt.ylabel('Actual (Varde)')
plt.show()

# r2_scores = []
# for pest in vardes:
#     pest_mask = data_gdf['target_pest'] == pest
#     pest_r2 = r2_score(y_test[pest_mask.reindex_like(y_test)], y_pred[pest_mask.reindex_like(y_pred)])
#     r2_scores.append(pest_r2)

# plt.barh(vardes, r2_scores)
# plt.show()

corr_matrix = dm.data_gdf[['target',  'Nederbördsmängd_sum_year_cumulative']].corr()

# Create a custom colormap for correlation values
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Blue to red colormap

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True,  # Show correlation values
            cmap=cmap, 
            center=0,  # Center the colormap at 0
            vmin=-1, vmax=1,  # Set the range of the colormap
            square=True,  # Make the plot square
            fmt='.2f',  # Format the numbers to 2 decimal places
            linewidths=0.5,  # Width of the lines between cells
            cbar_kws={'shrink': .5, 'label': 'Correlation'})  # Colorbar settings

# Add a legend or title (optional)
plt.title('Linear Correlation to next week (target)', pad=20)
plt.tight_layout()

# Show the plot
plt.show()


print( 'TEST:  MAE:',mean_absolute_error(tests, preds), 'MSE:', mean_squared_error(tests, preds), 'R2:', r2_score(tests, preds))
