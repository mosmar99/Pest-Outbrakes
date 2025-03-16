import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor as model
# from sklearn.ensemble import GradientBoostingRegressor as model
# from pygam import LinearGAM as model
# from sklearn.gaussian_process import GaussianProcessClassifier as classifier_model
# from sklearn.ensemble import HistGradientBoostingRegressor as model
# from sklearn.ensemble import HistGradientBoostingClassifier as classifier_model
# from sklearn.decomposition import PCA
# import umap.umap_ as umap
# from sklearn.neural_network import MLPRegressor as model
# from sklearn.neural_network import MLPClassifier as classifier_model
from xgboost import XGBRegressor as model
# from xgboost import XGBClassifier as classifier_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import data_processing.jbv_process as jbv_process
# from shapely import Point
field_id = 1807


data_df = pd.read_pickle("more_pests.pkl")
data_gdf = gpd.GeoDataFrame(data_df)
data_gdf['utvecklingsstadium'] = data_gdf['utvecklingsstadium'].astype(np.float64)

print(data_gdf.columns)
print(data_gdf.shape)

# Field_mask = data_gdf['Series_id'] == field_id
# plt.plot(data_gdf[Field_mask]['graderingsdatum'], data_gdf[Field_mask]['varde'], label='varde')

# plt.legend()
# plt.show()

# mean_utv = data_gdf[['utvecklingsstadium', 'week']].groupby('week').mean().reset_index()
# mean_varde = data_gdf[['varde', 'week']].groupby('week').mean().reset_index()

# data_gdf['varde_mean'] = pd.merge(data_gdf['week'], mean_varde, how='left', on='week')['varde']
# data_gdf['utvecklingsstadium_mean'] = pd.merge(data_gdf['week'], mean_utv, how='left', on='week')['utvecklingsstadium']

data_gdf = data_gdf.sort_values(by=['Series_id', 'graderingsdatum'])

# data_gdf = pd.get_dummies(data_gdf, columns=['sort'], dtype='int64')
# data_gdf = data_gdf.drop(['sort', 'forfrukt', 'ekologisk', 'forforfrukt'], axis=1)

target = 'Svartpricksjuka'
data_gdf['target'] = data_gdf[target]

vardes = ['Bladfläcksvampar', 'Brunrost', 'Svartpricksjuka','Gulrost', 'Mjöldagg', 'Vetets bladfläcksjuka', 'Gräsbladlus', 'Sädesbladlus', 'Havrebladlus', 'Nederbörd']

# datasets = []
# for skadegorare in vardes:
#     pest = data_gdf.copy()
#     pest['target'] = pest[skadegorare]
#     pest['target_pest'] = skadegorare
#     datasets.append(pest)

# data_gdf = pd.concat(datasets).reset_index()

# data_gdf['target1'] = data_gdf['target'] * (data_gdf['utvecklingsstadium_mean']/100)
# data_gdf['target2'] = data_gdf['varde'] - (data_gdf['varde_mean'])
# data_gdf['target3'] = data_gdf['varde'] - data_gdf.groupby('Series_id')['varde'].shift(1)

# data_gdf['lwd'] = data_gdf['Daggpunktstemperatur_mean'] * (data_gdf['Relativ Luftfuktighet_mean']/100) * data_gdf['Nederbördsmängd_sum'] / data_gdf['Solskenstid_sum']
# data_gdf['d_at_dt'] = (data_gdf['Daggpunktstemperatur_mean'] - data_gdf['Lufttemperatur_min'])

# lag_weather = ['lwd', 'd_at_dt', 'Relativ Luftfuktighet_mean', 'Daggpunktstemperatur_mean']

# lagged_weather = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,2) for col in lag_weather }

# lagged_weather_df = pd.DataFrame(lagged_weather)
# data_gdf = pd.concat([data_gdf, lagged_weather_df], axis=1)

agg_year = {'Nederbördsmängd_sum': 'sum',
            'Daggpunktstemperatur_mean': 'mean',
            'Relativ Luftfuktighet_mean': 'mean',
            'Lufttemperatur_max': 'mean',
            'Solskenstid_sum': 'sum'}

data_gdf['year'] = data_gdf['graderingsdatum'].dt.year
data_gdf = data_gdf.sort_values(by=['Series_id', 'graderingsdatum'])
cumulative = data_gdf.groupby(['year', 'Series_id']).rolling(window=52, on='graderingsdatum', min_periods=1).agg(agg_year).reset_index()
col_names = {key: f'{key}_year_cumulative'for key in agg_year.keys()}
cumulative = cumulative.rename(columns=col_names)
data_gdf[list(col_names.values())] = cumulative[list(col_names.values())]

data_gdf = data_gdf.drop(['Lufttemperatur_min', 'Lufttemperatur_max',
       'Nederbördsmängd_sum', 'Nederbördsmängd_max', 'Solskenstid_sum', 'Daggpunktstemperatur_max',
       'Långvågs-Irradians_mean'], axis=1)

# for pest in vardes:
#     delta = data_gdf.groupby('Series_id')[pest].shift(1) - data_gdf.groupby('Series_id')[pest].shift(2)
#     steps = [5,10,15,20,25,30]
#     for step in steps:
#         data_gdf[f'{pest}_inc_{step}'] = delta > step
#         data_gdf[f'{pest}_dec_{step}'] = delta < -step


# data_gdf = data_gdf.loc[:, data_gdf.nunique() > 1]

# lag_dependent = ['utvecklingsstadium', 'Bladfläcksvampar', 'Brunrost', 'Svartprickssjuka', 
#                  'Gräsbladlus', 'Gulrost', 'Havrebladlus', 'Mjöldagg', 'Nederbörd', 'Sädesbladlus', 'Vetets bladfläcksjuka']

lagged_dependent = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,2) for col in vardes }

lagged_dependent_df = pd.DataFrame(lagged_dependent)
data_gdf = pd.concat([data_gdf, lagged_dependent_df], axis=1)

# data_gdf = pd.get_dummies(data_gdf, columns=['target_pest'])

objects = data_gdf.select_dtypes(include=['object']).columns
data_gdf = data_gdf.drop(objects, axis=1)

data_gdf = data_gdf.dropna()

# conditions = [
#     data_gdf['target3'] >= 10,
#     (data_gdf['target3'] > -10) & (data_gdf['target3'] < 10),
#     data_gdf['target3'] <= -10
# ]

# choices = ['+', '=', '-']

# data_gdf['direction'] = np.select(conditions, choices, default='none')

print(data_gdf.dtypes)
numeric = data_gdf.select_dtypes(include=['float64', 'int64', 'UInt32', 'bool']).columns
X = data_gdf[numeric].drop(['target', 'utvecklingsstadium'] + vardes, axis=1)
y = data_gdf['target']

scaler_X = MinMaxScaler()
X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)

scaler_y = MinMaxScaler()
y = pd.DataFrame(scaler_y.fit_transform(y.to_numpy().reshape(-1,1)), index=y.index)

test_year = 2022

print(X.columns)
test_mask = data_gdf['graderingsdatum'].dt.year == test_year
badyears = ((data_gdf['graderingsdatum'].dt.year != 2023) & (data_gdf['graderingsdatum'].dt.year != 2018))

train_mask = ~(test_mask)
print(sum(train_mask)/len(train_mask))

print('testing on:', sum(test_mask)/len(test_mask))
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

ml_model = model()
ml_model.fit(X_train, y_train)

y_pred = pd.Series(ml_model.predict(X_test), index=y_test.index)

y_pred = pd.Series(scaler_y.inverse_transform(y_pred.to_numpy().reshape(-1,1)).flatten(), index=y_pred.index)
y_test = pd.Series(scaler_y.inverse_transform(y_test.to_numpy().reshape(-1,1)).flatten(), index=y_pred.index)

Field_mask = X_test['Series_id'] == field_id
# plt.scatter(y_pred, y_test, label='y_pred')
plt.plot(data_gdf['graderingsdatum'].reindex_like(y_pred[Field_mask]), y_pred[Field_mask], label='y_pred')
plt.plot(data_gdf['graderingsdatum'].reindex_like(y_test[Field_mask]), y_test[Field_mask], label='y_actual')

plt.legend()
plt.show()

# plt.figure(figsize=(16,10))
# plt.plot(range(len(y_test)),y_test, color='blue', zorder=1, lw=2, label='TRUE')
# plt.plot(range(len(y_pred)),y_pred, color='cyan', lw=2, label='PREDICTED')
# plt.show()

plt.scatter(y_pred, y_test)
plt.xlabel('Prediction (Varde)')
plt.ylabel('Actual (Varde)')
plt.legend()
plt.show()

# r2_scores = []
# for pest in vardes:
#     pest_mask = data_gdf['target_pest'] == pest
#     pest_r2 = r2_score(y_test[pest_mask.reindex_like(y_test)], y_pred[pest_mask.reindex_like(y_pred)])
#     r2_scores.append(pest_r2)

# plt.barh(vardes, r2_scores)
# plt.show()


print( 'TEST:  MAE:',mean_absolute_error(y_test, y_pred), 'MSE:', mean_squared_error(y_test, y_pred), 'R2:', r2_score(y_test, y_pred))
# print( 'UTV Pred:  MAE:',mean_absolute_error(y4_test_pre, y4_pred), 'MSE:', mean_squared_error(y4_test_pre, y4_pred), 'R2:', r2_score(y4_test_pre, y4_pred))
# print( 'grad varde Pred:  MAE:',mean_absolute_error(y3_test_pre, y3_pred), 'MSE:', mean_squared_error(y3_test_pre, y3_pred), 'R2:', r2_score(y3_test_pre, y3_pred))
# print( 'TEST:  Acuracy:',accuracy_score(y_dir_test, y_dir_pred))
