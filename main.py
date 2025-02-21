import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import geopandas as gpd
import visualize as viz
import pandas as pd

if __name__ == "__main__":
    crop='höstvete'
    pest = 'Svartpricksjuka'
    data_df = jbv_api.get_gradings(from_date="2015-08-04", to_date="2025-02-01", crop=crop, pest=pest)
    print("---FETCHED JBV-DATA")

    data_df = jbv_process.feature_extraction(data_df)
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
    print('6', data_gdf.columns)
    data_gdf = jbv_process.aggregate_data_for_plantations(data_gdf, time_period='MS')
    print(data_gdf)

    # top_latitud, top_longitud = jbv_process.get_uniq_plantation_coord(data_df, selector=1)
    # plantation_df = jbv_process.get_plantation_by_coord(data_df, top_latitud, top_longitud)
    # plantation_df = jbv_process.transform_date(data_df)
    # plantation_df
    # # geopandas, geometri kolumn, aggregerat veckovis på datum, 
    # print(plantation_df.head())
    # print("---PROCESSED JBV-DATA")

    # print("---FETCHED SMHI-DATA")
    # print("---PROCESSED SMHI-DATA")

    # viz.lineplot(extracted_df, crop, pest, top_latitud, top_longitud)
    # print("Extracted Data:")
    # print(extracted_df.head())
    # print(extracted_df.shape)
    