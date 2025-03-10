import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import data_processing.smhi_api as smhi_api
import data_processing.smhi_processing as smhi_processing
import geopandas as gpd
import visualize as viz
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    groda='höstvete'
    skadegorare = 'Svartpricksjuka'
    from_date = '2020-01-07'
    to_date = '2024-01-01'

    data_json = jbv_api.get_gradings(from_date=from_date, to_date=to_date, groda=groda, skadegorare=skadegorare)
    print("---FETCHED JBV-DATA")

    wanted_features = ['groda', 'skadegorare', 'graderingsdatum', 'utvecklingsstadium', 'varde', 'latitud', 'longitud', 'sort']
    data_df = jbv_process.feature_extraction(data_json, wanted_features) 
    print('1', data_df.shape)
    data_df = jbv_process.drop_rows_with_missing_values(data_df)
    print('2', data_df.shape)
    data_df = jbv_process.drop_duplicates(data_df)
    print('3', data_df.shape)
    data_df = jbv_process.clean_coordinate_format(data_df)
    print('4', data_df.shape)
    data_gdf = jbv_process.sweref99tm_to_wgs84(data_df)
    print('5', data_gdf.shape)
    data_gdf = jbv_process.remove_outside_sweden_coordinates(data_gdf)
    print('6', data_gdf.shape)
    data_gdf = jbv_process.aggregate_data_for_plantations(data_gdf)
    print('7', data_gdf.shape)
    data_gdf = jbv_process.add_sensitivity(data_gdf)
    print('8', data_gdf.shape)
    print(data_gdf)

    param_id = "2"
    param_id_2 = "5"
    params = [(param_id, 'mean'), (param_id_2, 'sum')]

    data_gdf = smhi_processing.gather_weather_data(
        data_gdf,
        params,
        smhi_api.get_stations_on_parameter_id,
        smhi_api.get_station_data_on_key_param,
        from_date,
        to_date,
        weekly=True)
    print('9', data_gdf.shape)
    
    most_frequent_plantation_gdf = jbv_process.get_most_frequent_plantation(data_gdf)
    padd_most_frequent_plantation_gdf = jbv_process.introduce_nan_for_large_gaps(most_frequent_plantation_gdf)
    viz.lineplot(padd_most_frequent_plantation_gdf)
    viz.lineplot_w_weather(padd_most_frequent_plantation_gdf)