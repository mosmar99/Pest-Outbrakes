import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import geopandas as gpd
import visualize as viz
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    groda='höstvete'
    skadegorare = 'Svartpricksjuka'
    data_json = jbv_api.get_gradings(from_date="2015-08-04", to_date="2025-02-01", groda=groda, skadegorare=skadegorare)
    print("---FETCHED JBV-DATA")

    wanted_features = ['groda', 'skadegorare', 'graderingsdatum', 'utvecklingsstadium', 'varde', 'skadegorare', 'latitud', 'longitud']
    data_df = jbv_process.feature_extraction(data_json, wanted_features) 
    print('1', data_df.shape)
    data_df = jbv_process.drop_rows_with_missing_values(data_df)
    print('2', data_df.shape)
    data_df = jbv_process.drop_duplicate_rows(data_df)
    print('3', data_df.shape)
    data_df = jbv_process.clean_coordinate_format(data_df)
    print('4', data_df.shape)
    data_gdf = jbv_process.sweref99tm_to_wgs84(data_df)
    print('5', data_gdf.shape)
    data_gdf = jbv_process.remove_outside_sweden_coordinates(data_gdf)
    print('6', data_gdf.shape)
    data_gdf = jbv_process.aggregate_data_for_plantations(data_gdf, time_period='W-MON')
    print('7', data_gdf.shape)
    
    most_frequent_plantation_gdf = jbv_process.get_most_frequent_plantation(data_gdf)
    padd_most_frequent_plantation_gdf = jbv_process.introduce_nan_for_large_gaps(most_frequent_plantation_gdf)
    viz.lineplot(padd_most_frequent_plantation_gdf)