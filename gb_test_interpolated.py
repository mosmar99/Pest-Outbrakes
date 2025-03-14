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
# from xgboost import XGBRegressor as model
# from xgboost import XGBClassifier as classifier_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
# from shapely import Point
field_id = 1807


data_df = pd.read_pickle("little_interp.pkl")
data_gdf = gpd.GeoDataFrame(data_df)

print(data_gdf.columns)
print(data_gdf.shape)

Field_mask = data_gdf['Series_id'] == field_id
plt.plot(data_gdf[Field_mask]['graderingsdatum'], data_gdf[Field_mask]['varde'], label='varde')

plt.legend()
plt.show()

# mean_utv = data_gdf[['utvecklingsstadium', 'week']].groupby('week').mean().reset_index()
# mean_varde = data_gdf[['varde', 'week']].groupby('week').mean().reset_index()

# data_gdf['varde_mean'] = pd.merge(data_gdf['week'], mean_varde, how='left', on='week')['varde']
# data_gdf['utvecklingsstadium_mean'] = pd.merge(data_gdf['week'], mean_utv, how='left', on='week')['utvecklingsstadium']

data_gdf = data_gdf.sort_values(by=['Series_id', 'graderingsdatum'])

# data_gdf = pd.get_dummies(data_gdf, columns=['sort'], dtype='int64')
data_gdf = data_gdf.drop(['sort', 'forfrukt', 'ekologisk', 'forforfrukt'], axis=1)

# data_gdf['target'] = (data_gdf['varde'] - data_gdf['varde_mean']) * (data_gdf['utvecklingsstadium_mean']/100)
# data_gdf['target2'] = data_gdf['varde'] - (data_gdf['varde_mean'])
data_gdf['target3'] = data_gdf['varde'] - data_gdf.groupby('Series_id')['varde'].shift(1)
data_gdf['target4'] = data_gdf['utvecklingsstadium']
data_gdf['target5'] = data_gdf['varde']

data_gdf['lwd'] = (data_gdf['Relativ Luftfuktighet_mean']/100) * data_gdf['Nederbördsmängd_sum'] * data_gdf['Nederbördsmängd_min']

lag_weather = ['lwd', 'Lufttemperatur_mean', 'Lufttemperatur_min',
       'Lufttemperatur_max', 'Nederbördsmängd_sum', 'Nederbördsmängd_max',
       'Nederbördsmängd_min', 'Solskenstid_sum', 'Solskenstid_min',
       'Solskenstid_max', 'Daggpunktstemperatur_mean',
       'Daggpunktstemperatur_min', 'Daggpunktstemperatur_max',
       'Relativ Luftfuktighet_mean', 'Relativ Luftfuktighet_min',
       'Relativ Luftfuktighet_max', 'Långvågs-Irradians_mean',
       'Långvågs-Irradians_min', 'Långvågs-Irradians_max']

data_gdf = data_gdf.drop(lag_weather, axis=1)

# lagged_weather = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,14) for col in lag_weather }

# lagged_weather_df = pd.DataFrame(lagged_weather)
# data_gdf = pd.concat([data_gdf, lagged_weather_df], axis=1)

lag_dependent = ['utvecklingsstadium', 'varde', 'varde_nearby_mean', 'varde_nearby_min', 'varde_nearby_max']

lagged_dependent = {f'{days}w_{col}': data_gdf.groupby('Series_id')[col].shift(days) for days in [7,14] for col in lag_dependent }

lagged_dependent_df = pd.DataFrame(lagged_dependent)
data_gdf = pd.concat([data_gdf, lagged_dependent_df], axis=1)

# print(data_gdf[['varde', 'target']].head(20))

data_gdf = data_gdf.dropna()

conditions = [
    data_gdf['target3'] >= 5,
    (data_gdf['target3'] > -5) & (data_gdf['target3'] < 5),
    data_gdf['target3'] <= -5
]

choices = [0, 1, 2]

data_gdf['direction'] = np.select(conditions, choices, default='none').astype(np.uint32)

numeric = data_gdf.select_dtypes(include=['float64', 'int64', 'UInt32', 'bool']).columns
X = data_gdf[numeric].drop(['target3', 'target4', 'target5', 'Series_id', 'direction'] + lag_dependent, axis=1)

# y = data_gdf['target']
# y2 = data_gdf['target2']
y3 = data_gdf['target3']
y4 = data_gdf['target4']
y5 = data_gdf['target5']
y_dir = data_gdf['direction']

# dir_train_mask = ((data_gdf['graderingsdatum'].dt.year % 2) == 0) & (data_gdf['graderingsdatum'].dt.year < 2022)

# X_dir_train, X_dir_test = X[dir_train_mask], X[~dir_train_mask]
# y_dir_train, y_dir_test = y_dir[dir_train_mask], y_dir[~dir_train_mask]

# dir_model = classifier_model()
# dir_model.fit(X_dir_train, y_dir_train)

# X['dir_pred'] = pd.Series(dir_model.predict(X), index=X.index)
# X = pd.get_dummies(X, columns=['dir_pred'], dtype='int64')

print(X.columns)
test_mask = data_gdf['graderingsdatum'].dt.year >= 2023
train_mask = ~(test_mask)

print('testing on:',sum(test_mask)/len(test_mask))
X_train, X_test = X[train_mask], X[test_mask][data_gdf['meassured_value'].reindex_like(X[test_mask])]
# y_train, y_test = y[train_mask], y[test_mask]
# y2_train, y2_test = y2[train_mask], y2[test_mask]
y5_train, y5_test = y5[train_mask], y5[test_mask][data_gdf['meassured_value'].reindex_like(y5[test_mask])]

X_train.to_csv('X_train_interp.csv', index=None)
X_test.to_csv('X_test_interp.csv', index=None)
y5_train.to_csv('y_train_interp.csv', index=None)
y5_test.to_csv('y_test_interp.csv', index=None)

# ml_model = model()
# ml_model.fit(X_train, y_train)

# ml_model2 = model()
# ml_model2.fit(X_train, y2_train)

ml_model5 = model()
ml_model5.fit(X_train, y5_train)

# y_pred = pd.Series(ml_model.predict(X_test), index=y_test.index)
# y2_pred = pd.Series(ml_model2.predict(X_test), index=y2_test.index)
y5_pred = pd.Series(ml_model5.predict(X_test), index=y5_test.index)
# y4_pred = pd.Series(ml_model4.predict(X_test_pre2), index=y4_test_pre.index)
# y_dir_pred = pd.Series(dir_model.predict(X_dir_test), index=X_dir_test.index)

# y_pred = (y_pred / (data_gdf['utvecklingsstadium_mean'].reindex_like(y_pred)/100)) + (data_gdf['varde_mean'].reindex_like(y_pred))
# y2_pred = y2_pred + (data_gdf['varde_mean'].reindex_like(y2_pred))

# y_pred = pd.concat([y_pred, y2_pred, y5_pred],axis=1).mean(axis=1)
y_pred = y5_pred
y_test = y5_test

Field_mask = data_gdf['Series_id'].reindex_like(y_pred) == field_id
# plt.scatter(y_pred, y_test, label='y_pred')
plt.plot(data_gdf['graderingsdatum'].reindex_like(y_pred[Field_mask]), y_pred[Field_mask], label='y_pred')
plt.plot(data_gdf['graderingsdatum'].reindex_like(y_test[Field_mask]), y_test[Field_mask], label='y_actual')

plt.legend()
plt.show()

plt.figure(figsize=(16,10))
plt.plot(range(len(y_test)),y_test, color='blue', zorder=1, lw=2, label='TRUE')
plt.plot(range(len(y_pred)),y_pred, color='cyan', lw=2, label='PREDICTED')
plt.show()

plt.scatter(y_pred, y_test)
plt.xlabel('Prediction (Varde)')
plt.ylabel('Actual (Varde)')
plt.legend()
plt.show()

corr_matrix = data_gdf.drop(['Series_id', 'graderingsdatum', 'groda', 'skadegorare',
       'utvecklingsstadium', 'varde', 'week', 'geometry', 'direction'], axis=1).corr()

# Create a custom colormap for correlation values
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Blue to red colormap

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[['target5']], 
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

print( 'TEST:  MAE:',mean_absolute_error(y_test, y_pred), 'MSE:', mean_squared_error(y_test, y_pred), 'R2:', r2_score(y_test, y_pred))
# print( 'UTV Pred:  MAE:',mean_absolute_error(y4_test_pre, y4_pred), 'MSE:', mean_squared_error(y4_test_pre, y4_pred), 'R2:', r2_score(y4_test_pre, y4_pred))
# print( 'grad varde Pred:  MAE:',mean_absolute_error(y3_test_pre, y3_pred), 'MSE:', mean_squared_error(y3_test_pre, y3_pred), 'R2:', r2_score(y3_test_pre, y3_pred))
# print( 'TEST:  Acuracy:',accuracy_score(y_dir_test, y_dir_pred))