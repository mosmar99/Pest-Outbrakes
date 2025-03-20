import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import data_processing.smhi_api as smhi_api
import data_processing.smhi_processing as smhi_processing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
 
pd.set_option('future.no_silent_downcasting', True)
 
if __name__ == '__main__':
    groda='Rågvete'
    # skadegorare = 'Svartpricksjuka'
    from_date = '2016-01-01' #'2016-01-07'
    to_date = '2024-01-01'
 
    data_json = jbv_api.get_gradings(from_date=from_date, to_date=to_date, groda=groda)
    print("---FETCHED JBV-DATA")

    data_df = jbv_process.feature_extraction(data_json)
    print('JBV FEATURES:', data_df.shape)
 
    # wanted_skadegorare = ['Svartpricksjuka', 'Vetets bladfläcksjuka', 'Bladfläcksvampar', 'Mjöldagg', 'Gulrost', 'Brunrost', 'Nederbörd', 'Havrebladlus', 'Sädesbladlus', 'Gräsbladlus']
    # wanted_skadegorare = ['Sköldfläcksjuka', 'Kornets bladfläcksjuka', 'Mjöldagg', 'Havrebladlus', 'Sädesbladlus', 'Kornrost', 'Gräsbladlus']
    wanted_skadegorare = ['Brunrost', 'Gulrost', 'Sköldfläcksjuka', 'Mjöldagg', 'Bladfläcksvampar']

    data_df = data_df[data_df['matmetod'].isin(['% ang blad 1–3', 'mm', 'antal/strå'])]
    data_df = data_df[data_df['skadegorare'].isin(wanted_skadegorare)]
    print('JBV Filtered matmetod and skadegorare:', data_df.shape)
    # print(data_df.iloc[80000:80040,:])
    # mask = data_df['Series_id'] == 512

    agg_dict = {'delomrade': 'first',
                'ekologisk': 'first',
                'forforfrukt': 'first',
                'forfrukt': 'first',
                'groda': 'first',
                'jordart': 'first',
                'lan': 'first',
                'latitud': 'first',
                'longitud': 'first',
                'plojt': 'first',
                'sadatum': 'first',
                'skordear': 'first',
                'sort': 'first',
                'broddbehandling': 'first',
                'utplaceringsdatumFalla': 'first',
                'graderingstyp':'first',
                'utvecklingsstadium': 'mean',
                'matmetod': 'first',
                # 'skadegorare': 'first',
                }

    data_df = data_df.groupby(['Series_id', 'graderingsdatum', 'skadegorare']).agg({**agg_dict, 'varde':'first'}).reset_index()
    pivot = data_df.pivot(index=['Series_id', 'graderingsdatum'], columns=['skadegorare'], values='varde').fillna(0)

    data_df = data_df.groupby(['Series_id', 'graderingsdatum']).agg({**agg_dict})
    data_df = data_df.join(pivot).reset_index()
    print('Aggregated data:', data_df.shape)

    # data_df = jbv_process.add_sensitivity(data_df, fill_na='median')
    # print('Added Sensitivity:', data_df.shape)
    # Aggregate JBV data
 
    data_df = jbv_process.drop_rows_with_missing_values(data_df)
    data_df = jbv_process.drop_duplicates(data_df)
    print('DROPPED DUPLICATES AND NAN:', data_df.shape)

    # Filter out uncommon and short Week Spans
    data_df = jbv_process.filter_uncommon_or_short_span(data_df, start_min=15, start_max=19, stop_min=26, stop_max=29)
    print('DROPPED SHORT/UNCOMMON SERIES', data_df.shape)
 
    # # Interpolate JBV data
    # data_df = jbv_process.interpolate_jbv_data(data_df, method='polynomial', order=2)
    # print('JBV INTERPOLATED:', data_df.shape)
 
    # Clean and convert coordinates
    data_df = jbv_process.clean_coordinate_format(data_df)
    data_gdf = jbv_process.sweref99tm_to_wgs84(data_df)
    data_gdf = jbv_process.remove_outside_sweden_coordinates(data_gdf)
    print('COORDS CLEANED:', data_gdf.shape)

    # data_gdf = jbv_process.get_surrounding_varde(data_gdf)
    print('COORDS CLEANED:', data_gdf.shape)
    # print(data_gdf.head(20))
    # data_gdf = data_gdf.dropna()
 
    # Get SMHI Data
    TEMP = '2'
    RAIN = '5'
    SUN = '10'
    DEWPOINT = '39'
    HUMIDITY = '6'
    AIR_PREASSUE = '9'
    LONG_WAVE_IRR = '24'
    WIND_SPEED = '4'
 
    params = [(TEMP, 'min'),
              (TEMP, 'max'),
              (RAIN, 'sum'),
              (RAIN, 'max'),
              (SUN, 'sum'),
              (DEWPOINT, 'mean'),
              (DEWPOINT, 'max'),
              (HUMIDITY, 'mean'),
              (LONG_WAVE_IRR, 'mean'),
              ]

    data_gdf = smhi_processing.gather_weather_data(
        data_gdf,
        params,
        smhi_api.get_stations_on_parameter_id,
        smhi_api.get_station_data_on_key_param,
        from_date,
        to_date,
        weekly=True,
        closest_n=3)
   
    print('SMHI GATHERED:', data_gdf.shape)
    print(data_gdf.head(40))
 
    # Save as pickle
    data_gdf.to_csv('more_pests_Rågvete.csv', index=False)
    data_gdf.to_pickle("more_pests_Rågvete.pkl")
 
    # Save data as test_out.pkl
    # can be read as follows
 
    # data_df = pd.read_pickle("test_out.pkl")
    # data_gdf = gpd.GeoDataFrame(data_df)
    # print(data_gdf.columns)