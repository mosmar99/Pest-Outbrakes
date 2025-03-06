import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import data_processing.smhi_api as smhi_api
import data_processing.smhi_processing as smhi_processing
import geopandas as gpd
import visualize as viz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor as model
# from sklearn.ensemble import GradientBoostingRegressor as model
# from sklearn.ensemble import HistGradientBoostingRegressor as model
# from sklearn.ensemble import BaggingRegressor as model
# from sklearn.ensemble import AdaBoostRegressor as model
# from sklearn.linear_model import LinearRegression as model
# from sklearn.svm import NuSVR as model
# from sklearn.svm import SVR as model
# from sklearn.svm import LinearSVR as model
# from sklearn.neural_network import MLPRegressor as model2
# from xgboost import XGBRegressor as model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == "__main__":
    # groda='höstvete'
    # skadegorare = 'Svartpricksjuka'
    # from_date = '2016-01-07'
    # to_date = '2024-01-01'

    # data_json = jbv_api.get_gradings(from_date=from_date, to_date=to_date, groda=groda, skadegorare=skadegorare)
    # print("---FETCHED JBV-DATA")

    # wanted_features = ['ekologisk', 'groda', 'skadegorare', 'graderingsdatum', 'utvecklingsstadium', 'varde', 'latitud', 'longitud', 'Region_id']
    # data_df = jbv_process.feature_extraction(data_json, wanted_features) 
    # print('1', data_df.shape)

    # agg = {'ekologisk': 'first',
    #        'groda': 'first',
    #        'skadegorare': 'first',
    #        'utvecklingsstadium': 'mean',
    #        'varde': 'mean',
    #        'latitud': 'first',
    #        'longitud': 'first'}
    # data_df = data_df.groupby('Region_id').resample('W-MON', on='graderingsdatum').agg(agg).reset_index()
    # print('2', data_df.shape)

    # # data_df = jbv_process.drop_rows_with_missing_values(data_df)
    # # data_df = jbv_process.drop_duplicates(data_df)
    # # print('3', data_df.shape)
    # data_df = jbv_process.clean_coordinate_format(data_df)
    # # print('4', data_df.shape)
    # data_gdf = jbv_process.sweref99tm_to_wgs84(data_df)
    # # print('5', data_gdf.shape)
    # data_gdf = jbv_process.remove_outside_sweden_coordinates(data_gdf)
    # # print('6', data_gdf.shape)
    # # data_gdf = jbv_process.aggregate_data_for_plantations(data_gdf, time_period='W-MON')
    # # print('7', data_gdf.shape)

    # temp = '2'
    # nederbord = '5'
    # sol = '10'
    # dewpoint = '39'
    # humidity = '6'
    # air_pressure = '9'
    # long_wave_irr = '24'
    # # wind_speed = '4'
    # params = [(temp, 'mean'),
    #           (nederbord, 'sum'),
    #           (sol, 'sum'),
    #           (dewpoint, 'mean'),
    #           (humidity, 'mean'),
    #           (air_pressure, 'mean'),
    #           (long_wave_irr, 'mean')]

    # data_gdf = smhi_processing.gather_weather_data(
    #     data_gdf,
    #     params,
    #     smhi_api.get_stations_on_parameter_id,
    #     smhi_api.get_station_data_on_key_param,
    #     from_date,
    #     to_date)
    # print('8', data_gdf.shape)

    # data_gdf.to_pickle("test_out.pkl")

    data_df = pd.read_pickle("test_out.pkl")
    data_gdf = gpd.GeoDataFrame(data_df)

    # data_gdf = data_gdf.dropna()
    print(data_gdf[data_gdf['Region_id'] == 129])

    data_gdf['week'] = data_gdf['graderingsdatum'].dt.isocalendar().week


    first_last_week = data_gdf.groupby(['Region_id']).agg({'week': ['first', 'last']})
    first_last_week.columns = ['week_first', 'week_last']
    first_last_week = first_last_week.reset_index()

    data_gdf = data_gdf.merge(first_last_week, on='Region_id')
    week_outlier_mask = (data_gdf['week_first'] >= 15) & (data_gdf['week_first'] <= 20) & (data_gdf['week_last'] >= 26) & (data_gdf['week_last'] <= 28)
    print(f'keeping {np.sum(week_outlier_mask)} rows, {np.sum(week_outlier_mask) / week_outlier_mask.shape[0]}%')
    data_gdf = data_gdf[week_outlier_mask].drop(['week_last','week_first'], axis=1)


    data_gdf['target_date'] = data_gdf['graderingsdatum'] + pd.DateOffset(weeks=1)

    lookup_gdf = data_gdf[['graderingsdatum', 'Region_id', 'varde']].rename(columns={'graderingsdatum': 'target_date', 'varde': 'target'})
    # print(data_gdf)
    # print(lookup_gdf)
    data_gdf = data_gdf.merge(lookup_gdf, on=['target_date', 'Region_id'], how='left').drop('target_date', axis=1)
    # print(data_gdf)



    # # Sort by Region and Date
    # data_gdf = data_gdf.sort_values(by=['Region_id', 'graderingsdatum'])
    # # Create a 1-week lag column
    # lag_cols = ['utvecklingsstadium',
    #             'varde',
    #             'Lufttemperatur_mean',
    #             'Nederbördsmängd_sum', 
    #             'Solskenstid_sum',
    #             'Daggpunktstemperatur_mean',
    #             'Relativ Luftfuktighet_mean',
    #             'Lufttryck reducerat havsytans nivå_mean',
    #             'Långvågs-Irradians_mean']
    
    # lag_col_names = [f'1w_{col}' for col in lag_cols]

    # data_gdf[lag_col_names] = data_gdf.groupby('Region_id')[lag_cols].shift(1)
    
    # print(data_gdf[['Solskenstid_sum', '1w_Solskenstid_sum']])



    # data_gdf['prev_date'] = data_gdf['graderingsdatum'] - pd.DateOffset(weeks=1)
    # lookup_gdf = data_gdf[['graderingsdatum', 'Region_id', 'varde']].rename(columns={'graderingsdatum': 'prev_date', 'varde': 'prev_varde'})
    # print(data_gdf)
    # print(lookup_gdf)
    # data_gdf = data_gdf.merge(lookup_gdf, on=['prev_date', 'Region_id'], how='left').drop('prev_date', axis=1)
    # print(data_gdf)

    # data_gdf['prev_date'] = data_gdf['graderingsdatum'] - pd.DateOffset(weeks=2)
    # lookup_gdf = data_gdf[['graderingsdatum', 'Region_id', 'varde']].rename(columns={'graderingsdatum': 'prev_date', 'varde': 'prev_prev_varde'})
    # print(data_gdf)
    # print(lookup_gdf)
    # data_gdf = data_gdf.merge(lookup_gdf, on=['prev_date', 'Region_id'], how='left').drop('prev_date', axis=1)
    # print(data_gdf)

    data_gdf = data_gdf.dropna()

    # print('3', data_gdf)
    y = data_gdf['target']
    X = data_gdf.drop(['geometry', 'target', 'groda', 'skadegorare', 'graderingsdatum', 'ekologisk'], axis=1)
    # print('features\n', X)
    # print('target\n', y)

    # print('Features:', X.columns)
    test_id = 1
    mask = X['Region_id'] == test_id

    X = X.drop(['Region_id'], axis=1)
    norm = ['utvecklingsstadium', 'Lufttemperatur_mean', 'Nederbördsmängd_sum', 'Solskenstid_sum', 'Daggpunktstemperatur_mean', 'Relativ Luftfuktighet_mean', 'Lufttryck reducerat havsytans nivå_mean', 'Långvågs-Irradians_mean'] # + lag_col_names

    X[norm] = (X[norm] - X[norm].min()) / (X[norm].max() - X[norm].min())
    
    field_y = y[mask]
    field_X = X[mask]

    print(field_X[['week', 'utvecklingsstadium']].join(field_y))

    y = y[~mask]
    X = X[~mask]

    # print(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit the model
    gb_reg = model(
        # n_estimators=100,
        # loss='squared_error',
        # criterion='friedman_mse',
        # alpha=0.9,
        # learning_rate=0.2,     
        # max_depth=3,
        # max_features=None,
        # # random_state=42
    )
    gb_reg = gb_reg.fit(X_train, y_train)

    # Define Neural Network model
    # mlp = model2(hidden_layer_sizes=(100, 50, 50), activation='relu', solver='adam', max_iter=10000, batch_size=32)
    # Train the model
    # mlp.fit(X_train, y_train)

    # Make predictions
    y_pred = pd.Series(gb_reg.predict(X_test), name='prediction')
    # y_pred2 = pd.Series(mlp.predict(X_test), name='prediction2')
    y_pred_field = pd.Series(gb_reg.predict(field_X), name='prediction_field')
    # y_pred_field2 = pd.Series(mlp.predict(field_X), name='prediction_field2')
    # print(type(y_pred))
    # print(y_pred)
    # print(y_test)

    print( 'TEST:  MAE:',mean_absolute_error(y_test, y_pred), 'MSE:', mean_squared_error(y_test, y_pred), 'R2:', r2_score(y_test, y_pred))
    # print( 'TEST2:  MAE:',mean_absolute_error(y_test, y_pred2), 'MSE:', mean_squared_error(y_test, y_pred2), 'R2:', r2_score(y_test, y_pred2))
    print( 'FIELD: MAE:',mean_absolute_error(field_y, y_pred_field), 'MSE:', mean_squared_error(field_y, y_pred_field), 'R2:', r2_score(field_y, y_pred_field))
    # print( 'FIELD2: MAE:',mean_absolute_error(field_y, y_pred_field2), 'MSE:', mean_squared_error(field_y, y_pred_field2), 'R2:', r2_score(field_y, y_pred_field2))

    plt.figure(figsize=(10, 8))

    plt.plot(field_X['week'], field_y, label='Actual', marker='o', linestyle='dashed')
    # plt.plot(field_X['week'], field_X.drop('week', axis=1), label=field_X.drop('week', axis=1).columns, marker='o', linestyle='dashed')

    plt.plot(field_X['week'], y_pred_field, label='Predicted', marker='s', linestyle='solid')
    # plt.plot(field_X['week'], y_pred_field2, label='Predicted2', marker='s', linestyle='solid')

    # Labels and title
    plt.xlabel('week')
    plt.ylabel('Value')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.show()

    # field_X_filtered = field_X.drop(['week'], axis=1)
    # plt.figure(figsize=(10, 8))

    # plt.plot(field_X['week'], field_X_filtered, label=field_X_filtered.columns, alpha=0.8, marker='o', linestyle='dashed')
    # plt.plot(field_X['week'], field_y, label=field_y.name, alpha=0.8, marker='o', linestyle='dashed')
    # plt.plot(field_X['week'], y_pred_field, label='Predicted', marker='s', linestyle='solid')

    # # Labels and title
    # plt.xlabel('week')
    # plt.ylabel('Value')
    # plt.title('Actual vs. Predicted Values')
    # plt.legend()
    # plt.show()


    # Calculate the correlation matrix
    corr_matrix = data_gdf.drop(['geometry', 'groda', 'skadegorare', 'graderingsdatum', 'ekologisk', 'Region_id', 'week'], axis=1).corr()

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
