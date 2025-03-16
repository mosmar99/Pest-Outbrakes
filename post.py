import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor as model
# from sklearn.ensemble import GradientBoostingRegressor as model
# from pygam import LinearGAM as model
# from sklearn.gaussian_process import GaussianProcessClassifier as classifier_model
from sklearn.ensemble import HistGradientBoostingRegressor as model
from sklearn.ensemble import HistGradientBoostingClassifier as classifier_model
# from sklearn.decomposition import PCA as reducer
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


data_df = pd.read_pickle("many.pkl")
data_gdf = gpd.GeoDataFrame(data_df)

print(data_gdf.columns)
print(data_gdf.shape)

# Field_mask = data_gdf['Series_id'] == field_id
# plt.plot(data_gdf[Field_mask]['graderingsdatum'], data_gdf[Field_mask]['varde'], label='varde')

# plt.legend()
# plt.show()

mean_utv = data_gdf[['utvecklingsstadium', 'week']].groupby('week').mean().reset_index()
mean_varde = data_gdf[['varde', 'week']].groupby('week').mean().reset_index()

data_gdf['varde_mean'] = pd.merge(data_gdf['week'], mean_varde, how='left', on='week')['varde']
data_gdf['utvecklingsstadium_mean'] = pd.merge(data_gdf['week'], mean_utv, how='left', on='week')['utvecklingsstadium']

data_gdf = data_gdf.sort_values(by=['Series_id', 'graderingsdatum'])

data_gdf = pd.get_dummies(data_gdf, columns=['delomrade', 'ekologisk', 'forforfrukt', 'forfrukt', 'jordart', 'lan', 'plojt', 'sort', 'broddbehandling', 'graderingstyp'], dtype='int64')
# data_gdf = data_gdf.drop(['sort', 'forfrukt', 'ekologisk', 'forforfrukt'], axis=1)

data_gdf['target'] = data_gdf['varde']

lag_weather = ['Lufttemperatur_min', 'Lufttemperatur_max', 'Nederbördsmängd_sum',
               'Nederbördsmängd_max', 'Solskenstid_sum', 'Daggpunktstemperatur_mean',
               'Daggpunktstemperatur_max', 'Relativ Luftfuktighet_mean', 'Långvågs-Irradians_mean']

lagged_weather = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,3) for col in lag_weather }

lagged_weather_df = pd.DataFrame(lagged_weather)
data_gdf = pd.concat([data_gdf, lagged_weather_df], axis=1)

data_gdf = data_gdf.loc[:, data_gdf.nunique() > 1]

lag_dependent = ['utvecklingsstadium', 'varde']

lagged_dependent = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,3) for col in lag_dependent }

lagged_dependent_df = pd.DataFrame(lagged_dependent)
data_gdf = pd.concat([data_gdf, lagged_dependent_df], axis=1)

data_gdf = data_gdf.dropna()

data_gdf.to_csv('data_gdf.csv', index=False)

numeric = data_gdf.select_dtypes(include=['float64', 'int64', 'UInt32', 'bool']).columns
X = data_gdf[numeric].drop(['target', 'varde_mean', 'utvecklingsstadium_mean'] + lag_dependent, axis=1)

scaler_X = MinMaxScaler()
X = pd.DataFrame(scaler_X.fit_transform(X),columns=X.columns , index=X.index)

y = data_gdf['target']
scaler_y = MinMaxScaler()

y = pd.DataFrame(scaler_y.fit_transform(y.to_numpy().reshape(-1, 1)), index=y.index)

X.to_csv('X.csv', index=None)
y.to_csv('y.csv', index=None)

test_year = 2023

print(X.columns)
test_mask = data_gdf['graderingsdatum'].dt.year >= test_year
train_mask = ~(test_mask)

print('testing on:',sum(test_mask)/len(test_mask))
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

X_train.to_csv('X_train.csv', index=None)
X_test.to_csv('X_test.csv', index=None)
y_train.to_csv('y_train.csv', index=None)
y_test.to_csv('y_test.csv', index=None)
