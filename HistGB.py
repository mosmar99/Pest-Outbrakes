import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os

from datamodule import datamodule

# from sklearn.ensemble import RandomForestRegressor as model
# from sklearn.ensemble import GradientBoostingRegressor as model
# from pygam import LinearGAM as model
# from sklearn.gaussian_process import GaussianProcessClassifier as classifier_model
from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.ensemble import HistGradientBoostingClassifier as classifier_model
# import umap.umap_ as umap
# from sklearn.neural_network import MLPRegressor as model
# from sklearn.neural_network import MLPClassifier as classifier_model
from xgboost import XGBRegressor
# from xgboost import XGBClassifier as classifier_model
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import data_processing.jbv_process as jbv_process
# from shapely import Point

field_id = 1807

dm = datamodule(datamodule.HOSTVETE)
dm.default_process(target='Svartpricksjuka')
sklearn_splits = dm.sklearn_CV_test_train_split()
splits = dm.CV_test_train_split()
X, y = dm.get_X_y()

param_grid = [
    {
        'loss': ['squared_error', 'absolute_error', 'poisson'],
        'learning_rate': [0.1, 0.12, 0.15],
        'max_leaf_nodes': [44, 48, 52, 54],
        'max_depth': [7, 9],
        'min_samples_leaf': [10],
        'max_features': [0.85]

    },
    {
        'loss': ['quantile'],
        'learning_rate': [0.1, 0.3, 0.5],
        'quantile': [0.1, 0.2],
        'max_leaf_nodes': [44, 48, 52, 54],
        'max_depth': [7, 9],
        'min_samples_leaf': [10],
        'max_features': [0.85]
    }
]

# param_grid = [
#     {
#         'objective': ['reg:squarederror', 'reg:pseudohubererror', 'reg:squaredlogerror'],
#         'eta': [0.15, 0.17, 0.2, 0.23],
#         'lambda': [0.7, 1.0, 1.3, 1.5, 1.8],
#         'alpha': [1,2,3,4,5],
#     }
# ]


# Convert to a list of dictionaries using ParameterGrid
param_grid_list = list(ParameterGrid(param_grid))

# Print first few parameter combinations to debug
for params in param_grid_list[:5]:  
    print(params)

print(y.sort_values(by=y.columns[0], ascending=True).head(20))

HistGB = HistGradientBoostingRegressor()
grid_search = GridSearchCV(
    estimator=HistGB,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=sklearn_splits,
    n_jobs=8,
    verbose=2,
    error_score='raise'
)

grid_search.fit(X, y.squeeze())
best_model = grid_search.best_estimator_

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Prevent truncation of wide output
pd.set_option('display.max_colwidth', None)  # Show full content of columns

results_df = pd.DataFrame(grid_search.cv_results_)
top_20 = results_df.sort_values(by="mean_test_score", ascending=False).head(20)
print(top_20[["mean_test_score", "params"]])

preds = []
tests = []

for X_train, X_test, y_train, y_test in splits:
    HistGBtest = XGBRegressor(**grid_search.best_params_)
    HistGBtest.fit(X_train, y_train.squeeze())
    y_pred = pd.DataFrame(HistGBtest.predict(X_test), index=X_test.index)

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

print('TEST:  MAE:', mean_absolute_error(tests, preds),
    'MSE:', mean_squared_error(tests, preds),
    'R2:', r2_score(tests, preds),
    'Best Hyperparameters for HistGB:', grid_search.best_params_)

# r2_scores = []
# for pest in vardes:
#     pest_mask = data_gdf['target_pest'] == pest
#     pest_r2 = r2_score(y_test[pest_mask.reindex_like(y_test)], y_pred[pest_mask.reindex_like(y_pred)])
#     r2_scores.append(pest_r2)

# plt.barh(vardes, r2_scores)
# plt.show()

# corr_matrix = dm.data_gdf[['target',  'Nederbördsmängd_sum_year_cumulative']].corr()

# # Create a custom colormap for correlation values
# cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Blue to red colormap

# # Create the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, 
#             annot=True,  # Show correlation values
#             cmap=cmap, 
#             center=0,  # Center the colormap at 0
#             vmin=-1, vmax=1,  # Set the range of the colormap
#             square=True,  # Make the plot square
#             fmt='.2f',  # Format the numbers to 2 decimal places
#             linewidths=0.5,  # Width of the lines between cells
#             cbar_kws={'shrink': .5, 'label': 'Correlation'})  # Colorbar settings

# # Add a legend or title (optional)
# plt.title('Linear Correlation to next week (target)', pad=20)
# plt.tight_layout()

# Show the plot
# plt.show()


# print( 'TEST:  MAE:',mean_absolute_error(tests, preds), 'MSE:', mean_squared_error(tests, preds), 'R2:', r2_score(tests, preds))
