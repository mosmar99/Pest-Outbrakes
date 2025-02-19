import pandas as pd
import geopandas as gpd

import os

import api

SMHI_URI = {
    "base_uri": "https://opendata-download-metobs.smhi.se/api/version/1.0",
    "all_parameters": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter.json",
    "all_stations": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter",
}

def get_stations_on_parameter_id(param_id="19"):
    """
    Get Dataframe for stations that have parameter id within timeframe.

    Args:
        jbv_gdf (GeoDataFrame): JBV reference data.
        param_id (String): Parameter id in SMHI API.
        
    Returns:
        GeoDataFrame: GeoDataFrame with SMHI Stations for timeframe and parameter id. crs="EPSG:4326"
    """
    stations_json = api.fetch_data(f'{SMHI_URI["all_stations"]}/{param_id}.json')["station"]

    df = pd.DataFrame(stations_json)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    return gdf

def get_station_data_on_key_param(station_key, param_id):
    """
    Get SMHI data, If dowloaded get from local copy else Download new SMHI data to local and read. Add station key col

    Args:
        station_key (String): Station key in SMHI API.
        param_id (String): Parameter id in SMHI API.

    Returns:
        DataFrame: DataFrame with SMHI Data for parameter and station.
    """
    directory = "./smhi_data"
    filename = f"{param_id}-{station_key}"
    filepath = f"{directory}/{filename}.csv"

    if os.path.exists(filepath):
        smhi_df = read_station_data_csv(filepath)
    else:
        endpoint = f"{SMHI_URI['base_uri']}/parameter/{param_id}/station/{station_key}/period/corrected-archive/data.csv"
        api.download_csv(endpoint, directory, filename)
        smhi_df = read_station_data_csv(filepath)
    smhi_df["station_key"] = station_key
    return smhi_df

def read_station_data_csv(filename):
    """
    Read SMHI station data from local data.csv.

    Args:
        filename (String): Relative path string to SMHI saved data, ex. ".\\smhi_data\\{param_id}-{station_key}.csv".
        
    Returns:
        DataFrame: DataFrame with SMHI Data for parameter and station.
    """
    data_start = 0
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            #Break on "Kvalitet, Exists in all parameters"
            if 'Kvalitet' in line:
                data_start = i
                break
    
    smhi_df = pd.read_csv(filename, skiprows=data_start, delimiter=';')

    return smhi_df

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # testing
    pass