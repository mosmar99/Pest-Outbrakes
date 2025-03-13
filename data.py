import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import data_processing.smhi_api as smhi_api
import data_processing.smhi_processing as smhi_processing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

if __name__ == '__main__':
    groda='höstvete'
    skadegorare = 'Svartpricksjuka'
    from_date = '2021-01-01' #'2016-01-07'
    to_date = '2024-01-01'

    data_json = jbv_api.get_gradings(from_date=from_date, to_date=to_date, groda=groda, skadegorare=skadegorare)
    print("---FETCHED JBV-DATA")

    wanted_features = ['graderingsdatum', 'utvecklingsstadium', 'varde', 'latitud', 'longitud', 'Series_id']
    data_df = jbv_process.feature_extraction(data_json, wanted_features) 
    print('JBV FEATURES:', data_df.shape)

    # Aggregate JBV data

    # agg_dict = {'ekologisk': 'first',
    #             'groda': 'first',
    #             'skadegorare': 'first',
    #             'utvecklingsstadium': 'mean',
    #             'varde': 'mean',
    #             'latitud': 'first',
    #             'longitud': 'first'}

    # data_df = jbv_process.aggregate_data_for_series(data_df, agg_dict, time_period='W-SUN')
    # print('JBV AGGREGATED:', data_df.shape)

    data_df = jbv_process.drop_rows_with_missing_values(data_df)
    data_df = jbv_process.drop_duplicates(data_df)
    print('DROPPED DUPLICATES AND NAN:', data_df.shape)

    # Filter out uncommon and short Week Spans
    data_df = jbv_process.filter_uncommon_or_short_span(data_df)
    print('DROPPED SHORT/UNCOMMON SERIES', data_df.shape) 

    # # Interpolate JBV data
    # data_df = jbv_process.interpolate_jbv_data(data_df, method='linear')
    # print('JBV INTERPOLATED:', data_df.shape)

    # Clean and convert coordinates
    data_df = jbv_process.clean_coordinate_format(data_df)
    data_gdf = jbv_process.sweref99tm_to_wgs84(data_df)
    data_gdf = jbv_process.remove_outside_sweden_coordinates(data_gdf)
    print('COORDS CLEANED:', data_gdf.shape)

    data_gdf = data_gdf.drop('geometry', axis=1)

    data_gdf['week'] = data_gdf['week'] - (data_gdf['week'].min()) + 1

    weeks = 1
    col = 'varde'
    data_gdf[f'{weeks}w_{col}'] = data_gdf.groupby('Series_id')[col].shift(weeks)
    data_gdf = data_gdf.dropna()

    data_gdf['year'] = data_gdf['graderingsdatum'].dt.year
    data_gdf = data_gdf.drop('graderingsdatum', axis=1)

    X = data_gdf.drop(['varde'], axis=1)
    y = data_gdf['varde']

    def normalize_features(df):
        return (df - df.min()) / (df.max() - df.min())

    # Normalize the entire dataset before splitting
    X_norm = normalize_features(X)
    y_norm = normalize_features(y)

    # Split based on the year
    largest_year = data_gdf['year'].max()
    X_train = X_norm[data_gdf['year'] < largest_year]
    X_test = X_norm[data_gdf['year'] == largest_year]
    y_train = y_norm[data_gdf['year'] < largest_year]
    y_test = y_norm[data_gdf['year'] == largest_year]

    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # # Get SMHI Data
    # TEMP = '2'
    # RAIN = '5'
    # SUN = '10'
    # DEWPOINT = '39'
    # HUMIDITY = '6'
    # AIR_PREASSUE = '9'
    # LONG_WAVE_IRR = '24'
    # # WIND_SPEED = '4'
    
    # params = [(TEMP, 'mean'),
    #           (RAIN, 'sum'),
    #           (SUN, 'sum'),
    #           (DEWPOINT, 'mean'),
    #           (HUMIDITY, 'mean'),
    #           (AIR_PREASSUE, 'mean'),
    #           (LONG_WAVE_IRR, 'mean')]

    # data_gdf = smhi_processing.gather_weather_data(
    #     data_gdf,
    #     params,
    #     smhi_api.get_stations_on_parameter_id,
    #     smhi_api.get_station_data_on_key_param,
    #     from_date,
    #     to_date,
    #     aggregation_rule='D')
    
    # print('SMHI GATHERED:', data_gdf.shape)

    # # Save as pickle
    # data_gdf.to_pickle("test_out.pkl")

    # Save data as test_out.pkl
    # can be read as follows

    # data_df = pd.read_pickle("test_out.pkl")
    # data_gdf = gpd.GeoDataFrame(data_df)
