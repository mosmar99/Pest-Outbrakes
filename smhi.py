import requests
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

parent_dir = os.path.relpath("../")

sys.path.append(parent_dir)

import api
import csv

import geopandas as gpd

SMHI_URI = {
    "all_parameters": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter.json",
    "all_stations": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter",
}
    
def fetch_data_on_key(endpoint, json_key):
    """
    Get Dataframe for stations that have parameter id within timeframe.

    Args:
        endpoint (String): Api endpoint string.
        json_key (String): Key for nested json in SMHI data.
        
    Returns:
        DataFrame: DataFrame with nested json contents,
    """
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[json_key])
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_stations_on_parameter_id(param_id="19", measuringStations=None, from_date=None, to_date=None):
    """
    Get Dataframe for stations that have parameter id within timeframe.

    Args:
        jbv_gdf (GeoDataFrame): JBV reference data.
        param_id (String): Parameter id in SMHI API.
        from_date (String): Fromdate, for checking avalibility in station.
        to_date (String): Fromdate, for checking avalibility in station.
        
    Returns:
        GeoDataFrame: GeoDataFrame with SMHI Stations for timeframe and parameter id.
    """
    df = fetch_data_on_key(f'{SMHI_URI["all_stations"]}/{param_id}.json', "station")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.drop(['latitude','longitude', 'summary', 'link', 'id'], axis=1)

    gdf['updated'] = pd.to_datetime(gdf['updated'], unit="ms")
    gdf['from'] = pd.to_datetime(gdf['from'], unit="ms")
    gdf['to'] = pd.to_datetime(gdf['to'], unit="ms")

    if measuringStations:
        measuringStations_mask = gdf["measuringStations"] == measuringStations
        gdf = gdf[measuringStations_mask]

    if from_date:
        from_mask = gdf['from'] < pd.to_datetime(from_date)
        gdf = gdf[from_mask]
    
    if to_date:
        to_mask = gdf['to'] > pd.to_datetime(to_date)
        gdf = gdf[to_mask]
    
    return gdf

def find_closest_stations(jbv_gdf, param_id, from_date, to_date):
    """
    Get SMHI data, If dowloaded get from local copy else Download new SMHI data to local and read.

    Args:
        jbv_gdf (GeoDataFrame): JBV reference data
        param_id (String): Parameter id in SMHI API.
        from_date (String): Fromdate, for checking avalibility in station
        to_date (String): Fromdate, for checking avalibility in station
        
    Returns:
        tuple(GeoDataFrame, List): GeoDataFrame with SMHI Station key and location appended, List of unique station keys's.
    """
    smhi_stations_gdf = get_stations_on_parameter_id(param_id=param_id, measuringStations="CORE", from_date=from_date, to_date=to_date)

    # Convert to estimated UTM CRS to achieve accuracy in distances
    utm_crs = jbv_gdf.estimate_utm_crs()
    jbv_gdf = jbv_gdf.to_crs(utm_crs)
    smhi_stations_gdf = smhi_stations_gdf.to_crs(utm_crs)
    # Keep station point for plotting
    smhi_stations_gdf["station_location"] = smhi_stations_gdf["geometry"]

    # Perform Spatial join by nearest neighbour
    nearest_gdf = jbv_gdf.sjoin_nearest(smhi_stations_gdf, how="left")[["key", "station_location"]]
    jbv_gdf["station_key"] = nearest_gdf["key"]
    jbv_gdf["station_location"] = nearest_gdf["station_location"].to_crs("EPSG:4326")
    
    return jbv_gdf.to_crs("EPSG:4326"), pd.unique(nearest_gdf["key"])

#region Getting and reading SMHI DATA

def get_station_data_on_key_param(station_key, param_id, from_date, to_date):
    """
    Get SMHI data, If dowloaded get from local copy else Download new SMHI data to local and read.

    Args:
        station_key (String): Station key in SMHI API.
        param_id (String): Parameter id in SMHI API.

    Returns:
        DataFrame: DataFrame with SMHI Data for parameter and station.
    """
    directory = "./smhi_data"
    filename = f"{directory}/{param_id}-{station_key}.csv"

    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filename):
        smhi_df = read_station_data_file_on_key_param(filename)
    else:
        download_csv_station_data_on_key_param(station_key, param_id, filename)
        smhi_df = read_station_data_file_on_key_param(filename)

    smhi_df = filter_smhi_data_on_date(smhi_df, from_date, to_date)
    return smhi_df

def download_csv_station_data_on_key_param(station_key, param_id, filename):
    """
    Download SMHI data to local data.csv.

    Args:
        station_key (String): Station key in SMHI API.
        param_id (String): Parameter id in SMHI API.
        filename (String): Relative path string to SMHI saved data, ex. ".\\smhi_data\\{param_id}-{station_key}.csv".
    """
    endpoint = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param_id}/station/{station_key}/period/corrected-archive/data.csv"
    with requests.get(endpoint, stream=True) as r:
        lines = (line.decode('utf-8') for line in r.iter_lines())
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in csv.reader(lines):
                writer.writerow(row)

def read_station_data_file_on_key_param(filename):
    """
    Read SMHI data from local data.csv.

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
    smhi_df = clean_station_data(smhi_df)

    return smhi_df

def clean_station_data(smhi_gdf):
    """
    Reformat SMHI Data in unified format and drop irrelevant rows.

    Args:
        smhi_gdf (GeoDataFrame): GeoDataFrame of SMHI data.

    Returns:
        DataFrame: SMHI Data in unified format without irrelevant rows.
    """
    # Dates comes in either ["Tid (UTC)", "Datum"] or ["Från Datum Tid (UTC)", "Till Datum Tid (UTC)"] format based on parameter,
    # consolidate both in same format with "DateTime (UTC)"
    if "Tid (UTC)" in smhi_gdf.columns:
        smhi_gdf["DateTime (UTC)"] = pd.to_datetime(smhi_gdf["Datum"] + " " + smhi_gdf["Tid (UTC)"])
    else:
        smhi_gdf["DateTime (UTC)"] = pd.to_datetime(smhi_gdf["Från Datum Tid (UTC)"])

    # Drop irrelevant columns if they exist
    drop_cols = ["Representativt dygn", "Tid (UTC)", "Datum", "Från Datum Tid (UTC)", "Till Datum Tid (UTC)", "Unnamed: 4", "Unnamed: 5", "Tidsutsnitt:"]
    for col in drop_cols:
        if col in smhi_gdf.columns:
            smhi_gdf = smhi_gdf.drop(col, axis=1)
    
    return smhi_gdf

def filter_smhi_data_on_date(smhi_df, from_date, to_date):
    """
    Filter on timeframe

    Args:
        smhi_df (GeoDataFrame): GeoDataFrame of SMHI data.
        from_date (string): Date to filter from
        to_date (string): Date to filter to

    Returns:
        DataFrame: SMHI Data within timeframe
    """
    if from_date and to_date:
        from_mask = (smhi_df["DateTime (UTC)"] > pd.to_datetime(from_date)) & (smhi_df["DateTime (UTC)"] < pd.to_datetime(to_date))
        smhi_df = smhi_df[from_mask]
    return smhi_df

#endregion

if __name__ == "__main__":
    from shapely.geometry import LineString
    # """
    # Example usage of module
    # """
    from_date = '2020-01-01'
    to_date = '2021-01-01'
    param_id = "19"

    # smhi_gdf = get_station_data_on_key_param("188790", "21", from_date, to_date)
    # # smhi_gdf = get_station_data_on_key_param("154860", "19", from_date, to_date)

    # print(smhi_gdf)

    gradings_df = api.get_gradings(from_date=str(from_date), to_date=str(to_date), crop="Höstvete", pest="svartpricksjuka")
    
    jbv_gdf = gpd.GeoDataFrame(gradings_df, geometry=gpd.points_from_xy(gradings_df.longitud, gradings_df.latitud), crs="EPSG:3006")
    jbv_gdf["geometry"] = jbv_gdf["geometry"].to_crs("EPSG:4326")

    # REMOVE ENTRIES NOT IN SWEDEN! Should probably be implemented somewhere :)
    countries = ".\\geodata\\ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp"
    world = gpd.read_file(countries)
    sweden_boundary = world[world['NAME'] == 'Sweden']
    sweden_boundary.to_crs("EPSG:4326")
    jbv_gdf = jbv_gdf[jbv_gdf.geometry.within(sweden_boundary.geometry.union_all())]

    jbv_gdf, unique_keys = find_closest_stations(jbv_gdf, param_id, from_date, to_date)
    print("JBV GDF with station key and location: \n", jbv_gdf)
    print("\nUnique station Keys Near Fields: \n", unique_keys)

    station_data = get_station_data_on_key_param(unique_keys[0],param_id, from_date, to_date)
    print("\nStation data for first station in Unique list: \n", station_data)

    def create_connection(row, l_key, r_key):
        if row[l_key].is_empty or row[r_key].is_empty:
            return None  # or return LineString() to create an empty LineString
        else:
            return LineString([row[l_key], row[r_key]])
    
    jbv_gdf['connection'] = jbv_gdf.apply(lambda row: create_connection(row, l_key='geometry', r_key='station_location'), axis=1)

    connections = gpd.GeoDataFrame(jbv_gdf['connection'], geometry='connection', crs="EPSG:4326")

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the original points
    jbv_gdf["geometry"].plot(ax=ax, color='green', label='Field', markersize=20)
    # smhi_stations_gdf["station_loc"].plot(ax=ax, color='gray', label='Unused Station', markersize=10)
    jbv_gdf["station_location"].plot(ax=ax, color='blue', label='Used Station', markersize=20)

    # Plot the connection lines
    connections.plot(ax=ax, color='gray', linewidth=0.8, linestyle='--')

    plt.legend()
    plt.title('Station selection')
    plt.show()

    # station_data = get_data_from_closest_station(unique_keys, "19", to_date)

    # fd = {key: get_date_interval_from_data(value, to_date, from_date) for key, value in station_data.items() if value is not None}
    # print(fd)

    # first_key = next(iter(fd))
    # head = fd[first_key]
