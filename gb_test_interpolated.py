import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor as model
# from sklearn.ensemble import GradientBoostingRegressor as model
# from sklearn.gaussian_process import GaussianProcessRegressor as model
from sklearn.ensemble import HistGradientBoostingRegressor as model
# from sklearn.ensemble import HistGradientBoostingClassifier as classifier_model
# from sklearn.neural_network import MLPRegressor as model
from xgboost import XGBRegressor as model2
# from xgboost import XGBClassifier as classifier_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
# from shapely import Point
field_id = 1807


data_df = pd.read_pickle("test_out2.pkl")
data_gdf = gpd.GeoDataFrame(data_df)
print(data_gdf.columns)
print(data_gdf.shape)

Field_mask = data_gdf['Series_id'] == field_id
plt.plot(data_gdf[Field_mask]['graderingsdatum'], data_gdf[Field_mask]['varde'], label='varde')

plt.legend()
plt.show()

mean_utv = data_gdf[['utvecklingsstadium', 'week']].groupby('week').mean().reset_index()
mean_varde = data_gdf[['varde', 'week']].groupby('week').mean().reset_index()

data_gdf['varde_mean'] = pd.merge(data_gdf['week'], mean_varde, how='left', on='week')['varde']
data_gdf['utvecklingsstadium_mean'] = pd.merge(data_gdf['week'], mean_utv, how='left', on='week')['utvecklingsstadium']

data_gdf = data_gdf.sort_values(by=['Series_id', 'graderingsdatum'])

lag_weather = ['Lufttemperatur_min', 'Lufttemperatur_max', 'Nederbördsmängd_sum', 'Nederbördsmängd_max', 'Solskenstid_sum', 'Daggpunktstemperatur_mean', 'Relativ Luftfuktighet_mean', 'Långvågs-Irradians_mean']

lagged_weather = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,7) for col in lag_weather }

lagged_weather_df = pd.DataFrame(lagged_weather)
data_gdf = pd.concat([data_gdf, lagged_weather_df], axis=1)

lag_weather = ['utvecklingsstadium', 'varde']

lagged_weather = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(7,8) for col in lag_weather }

lagged_weather_df = pd.DataFrame(lagged_weather)
data_gdf = pd.concat([data_gdf, lagged_weather_df], axis=1)

# data_gdf = pd.get_dummies(data_gdf, columns=['sort', 'forfrukt', 'ekologisk'], dtype='int64')
data_gdf = data_gdf.drop(['sort', 'forfrukt', 'ekologisk'], axis=1)

data_gdf['target'] = data_gdf['varde'] * (data_gdf['utvecklingsstadium_mean']/100)
data_gdf['target2'] = data_gdf['varde'] * (data_gdf['varde_mean']/100)
data_gdf['target3'] = data_gdf['varde'] - data_gdf.groupby('Series_id')['varde'].shift(7)
data_gdf['target4'] = data_gdf['utvecklingsstadium']

# print(data_gdf[['varde', 'target']].head(20))

data_gdf = data_gdf.dropna()

print('Dropped NA:', data_gdf.shape)
print(data_gdf.columns)

numeric = data_gdf.select_dtypes(include=['float64', 'int64', 'UInt32', 'bool']).columns
X = data_gdf[numeric].drop(['target', 'target2', 'target3', 'target4', 'varde', 'utvecklingsstadium', 'week', 'Series_id', 'varde_mean', 'utvecklingsstadium_mean'], axis=1)

print(X.columns)
y = data_gdf['target']
y2 = data_gdf['target2']
y3 = data_gdf['target3']
y4 = data_gdf['target4']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_mask_pre = (data_gdf['graderingsdatum'].dt.year == 2016) | (data_gdf['graderingsdatum'].dt.year == 2018)
print('pre training on:', sum(train_mask_pre)/len(train_mask_pre))
X_train_pre, X_test_pre = X[train_mask_pre], X[~train_mask_pre]
y3_train_pre, y3_test_pre = y3[train_mask_pre], y3[~train_mask_pre]

ml_model3 = model2()
ml_model3.fit(X_train_pre, y3_train_pre)

X['t3_pred'] = pd.Series(ml_model3.predict(X), index=X.index)

X_train_pre2, X_test_pre2 = X[train_mask_pre], X[~train_mask_pre]
y4_train_pre, y4_test_pre = y4[train_mask_pre], y4[~train_mask_pre]

ml_model4 = model2()
ml_model4.fit(X_train_pre2, y4_train_pre)

X['utv_pred'] = pd.Series(ml_model4.predict(X), index=X.index)


print(X.columns)
test_mask = data_gdf['graderingsdatum'].dt.year == 2023
train_mask = ~(train_mask_pre | test_mask)
print('training on:', sum(train_mask)/len(train_mask))
print('testing on:',sum(test_mask)/len(test_mask))
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
y2_train, y2_test = y2[train_mask], y2[test_mask]

X_test = X_test[X_test['meassured_value']]
X_test = X_test[X_test['meassured_value']]

ml_model = model()
ml_model.fit(X_train, y_train)

ml_model2 = model()
ml_model2.fit(X_train, y2_train)

y_pred = pd.Series(ml_model.predict(X_test), index=X_test.index)
y2_pred = pd.Series(ml_model2.predict(X_test), index=X_test.index)
y3_pred = pd.Series(ml_model3.predict(X_test_pre), index=y3_test_pre.index)
y4_pred = pd.Series(ml_model4.predict(X_test_pre2), index=y4_test_pre.index)

y_pred = y_pred / (data_gdf['utvecklingsstadium_mean'].reindex_like(y_pred)/100)
y2_pred = y2_pred / (data_gdf['varde_mean'].reindex_like(y2_pred)/100)

y_pred = pd.concat([y_pred, y2_pred],axis=1).mean(axis=1)
# y_pred = y_pred
y_test = y_test / (data_gdf['utvecklingsstadium_mean'].reindex_like(y_test)/100)

Field_mask = data_gdf['Series_id'].reindex_like(y_pred) == field_id
Field_mask_test = data_gdf['Series_id'].reindex_like(y_test) == field_id
# plt.scatter(y_pred, y_test, label='y_pred')
plt.plot(data_gdf['graderingsdatum'].reindex_like(y_pred[Field_mask]), y_pred[Field_mask], label='y_pred')
plt.plot(data_gdf['graderingsdatum'].reindex_like(y_test[Field_mask_test]), y_test[Field_mask_test], label='y_actual')

plt.legend()
plt.show()

plt.scatter(y_pred, y_test.reindex_like(y_pred))
plt.xlabel('Prediction (Varde)')
plt.ylabel('Actual (Varde)')
plt.legend()
plt.show()

corr_matrix = data_gdf.drop(['Series_id', 'graderingsdatum', 'groda', 'skadegorare',
       'utvecklingsstadium', 'varde', 'week', 'geometry'], axis=1).corr()

# Create a custom colormap for correlation values
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Blue to red colormap

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[['target']], 
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

print( 'TEST:  MAE:',mean_absolute_error(y_test.reindex_like(y_pred), y_pred), 'MSE:', mean_squared_error(y_test.reindex_like(y_pred), y_pred), 'R2:', r2_score(y_test.reindex_like(y_pred), y_pred))
# print( 'UTV Pred:  MAE:',mean_absolute_error(y4_test_pre, y4_pred), 'MSE:', mean_squared_error(y4_test_pre, y4_pred), 'R2:', r2_score(y4_test_pre, y4_pred))
# print( 'grad varde Pred:  MAE:',mean_absolute_error(y3_test_pre, y3_pred), 'MSE:', mean_squared_error(y3_test_pre, y3_pred), 'R2:', r2_score(y3_test_pre, y3_pred))
# print( 'TEST:  Acuracy:',accuracy_score(y3_test_pre, y3_pred))