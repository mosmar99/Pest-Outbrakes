import requests
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

from io import StringIO

parent_dir = os.path.relpath("../")

sys.path.append(parent_dir)

import api

import datetime

import numpy as np
import csv
# import chardet

import geopandas as gpd

SMHI_URI = {
    "all_parameters": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter.json",
    "all_stations": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter",
}

@staticmethod
def fetch_data(endpoint):
    """
    Fetches data from SMHI endpoint
    
    Args:
        endpoint (string): endpoint for the SMHI api
    
    Returns:
        DataFrame: if call was successful otherwise None
    """

    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json() 

        if "station" in data: 
            df = pd.DataFrame(data["station"])
            return df
        else:
            print("Unexpected JSON structure:", data)
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
def fetch_data_on_key(endpoint, json_key):
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[json_key])
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def find_closest_stations(jbv_gdf, param_id, from_date, to_date):
    smhi_stations_gdf = get_stations_on_parameter_id(param_id=param_id, measuringStations="CORE", from_date=from_date, to_date=to_date)

    print(smhi_stations_gdf.crs)
    print(jbv_gdf.crs)
    # Convert to estimated UTM CRS to achieve accuracy in distances
    utm_crs = jbv_gdf.estimate_utm_crs()
    jbv_gdf = jbv_gdf.to_crs(utm_crs)
    smhi_stations_gdf = smhi_stations_gdf.to_crs(utm_crs)
    # Keep station point for plotting
    smhi_stations_gdf["station_loc"] = smhi_stations_gdf["geometry"]

    # Perform Spatial join by nearest neighbour
    jbv_gdf = jbv_gdf.sjoin_nearest(smhi_stations_gdf, how="left")
    return jbv_gdf, pd.unique(jbv_gdf["key"])


def get_stations_on_parameter_id(param_id="19", measuringStations=None, from_date=None, to_date=None):
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

@staticmethod
def fetch_station_data(station_key, parameter):
    """
    Fetch data from API

    Args:
        station_key (int): SMHI Station Key
        parameter (int): SMHI Parameter
    Returns:
    string: Response from SMHI if successful otherwise None
    """
    endpoint = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{parameter}/station/{station_key}/period/corrected-archive/data.csv"
    response = requests.get(endpoint)
    return response if response.status_code == 200 else None
    
@staticmethod
def save_csv_with_correct_encoding(response, file_path='station_data.csv'):
    """
    Detect encoding and save CSV file

    Args:
        response (string): Response from SMHI API Call
        file_path (string): location to store data at.
    
    Returns:
        string: The encoding format of SMHI data
    """
    raw_data = response.content
    encoding_info = chardet.detect(raw_data)
    decoded_text = raw_data.decode(encoding_info['encoding'], errors='replace')
    data = pd.read_csv(StringIO(decoded_text), on_bad_lines='skip')
    data.to_csv(file_path, index=False)
    return encoding_info['encoding']

@staticmethod
def read_and_clean_station_data(file_path, encoding):
    """ 
    Read and clean station data from CSV
    
    Args:
        file_path (string): File path to store data to.
        encoding (string): encoding information of data recieved from SMHI
    
    Return:
        DataFrame: SMHI data without header
        List(string): List of strings representing our SMHI data
    """
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
    
    lines = [line.strip() for line in lines]

    header_index = next((i for i, line in enumerate(lines) if 'Från Datum Tid (UTC)' in line), -1)
    if header_index == -1:
        return None
    
    head = lines[header_index].split(";")
    data = [line.split(';') for line in lines[header_index + 1:]]  # Remove header from data

    cleaned_data = []
    for d in data:
        idx = d[0].find(',')
        cleaned_line = d[0][idx+1:].split(';') 
        cleaned_data.append(cleaned_line + d[1:]) 
    
    df = pd.DataFrame(cleaned_data, columns=head)
    return df, lines

def get_station_data_on_key_param(station_key, parameter):
    filename = f"./smhi_data/{parameter}-{station_key}.csv"
    if os.path.exists(filename):
        smhi_df = read_station_data_file_on_key_param(filename)
    else:
        download_csv_station_data_on_key_param(station_key, parameter, filename)
        smhi_df = read_station_data_file_on_key_param(filename)

    return smhi_df

def download_csv_station_data_on_key_param(station_key, parameter, filename):
    endpoint = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{parameter}/station/{station_key}/period/corrected-archive/data.csv"
    with requests.get(endpoint, stream=True) as r:
        lines = (line.decode('utf-8') for line in r.iter_lines())
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in csv.reader(lines):
                writer.writerow(row)

def read_station_data_file_on_key_param(filename):
    data_start = 0
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if 'Kvalitet' in line:
                data_start = i
                break
    
    smhi_df = pd.read_csv(filename, skiprows=data_start, delimiter=';')
    return smhi_df

def get_for_station_for_parameter(station_key, parameter):
    """
    Main function to get and process data

    Args:
        station_key (integer): SMHI station key
        parameter (int): parameter we want to query for.
        to_date (string): the upperbound date format: yyyy:mm:dd

    Returns:
        DataFrame: data of some station, specified by key. None if station doesn't have data within interval
    """
    response = fetch_station_data(station_key, parameter)
    if not response:
        print(f"Error fetching data")
        return None

    encoding = save_csv_with_correct_encoding(response)
    df, lines = read_and_clean_station_data('station_data.csv', encoding)

    df.to_csv('station_data.csv', sep=';', index=False)
    return df

def get_data_stations(unique_keys, param_id):
    """
    Gets DataFrame with data from the closest station that has data within our time interval.

    Args:
        cl (List(float, float)): Sorted list of tuples with regards to the meter distance, tuple:(distance, station_key)
        param (int): parameter we want to query for.
        to_date (string): the upperbound date format: yyyy:mm:dd

    Returns:
        DataFrame: data of the closest station that has data wihtin our time interval. None if none of the stations had data within interval
    """
    return {key: get_for_station_for_parameter(key, param_id) for key in unique_keys}

def get_date_interval_from_data(data, to_date, from_date):
    """
    Removes data points not contained within specified time interval

    Args:
        data (DataFrame): The data to filter
        to_date (string): the upperbound date format: yyyy:mm:dd
        from_date (string): the lowerbound date format: yyyy:mm:dd

    Returns:
        DataFrame: filtered data with regards to specified bounds
    """
    
    data['Från Datum Tid (UTC)'] = pd.to_datetime(data['Från Datum Tid (UTC)'], errors='coerce')
    data['Till Datum Tid (UTC)'] = pd.to_datetime(data['Till Datum Tid (UTC)'], errors='coerce')

    start_range = pd.to_datetime(from_date)
    end_range = pd.to_datetime(to_date)

    filtered_data = data[(data['Från Datum Tid (UTC)'] >= start_range) & (data['Till Datum Tid (UTC)'] <= end_range)]
    return filtered_data

def get_parameter_data_on_JBV_gdf(jbv_gdf, param_id, from_date, to_date):
    jbv_gdf, unique_keys = find_closest_stations(jbv_gdf, param_id, from_date, to_date)
    station_data = get_data_stations(unique_keys, param_id)

    filtered_station_data = {key: get_date_interval_from_data(value, to_date, from_date) for key, value in station_data.items() if value is not None}
    return jbv_gdf, filtered_station_data


if __name__ == "__main__":
    # """
    # Example usage of module
    # """
    # from_date = '2020-01-01'
    # to_date = '2021-01-01'

    smhi_df = get_station_data_on_key_param("154860", "19")


    print(smhi_df)

    # gradings_df = api.get_gradings(from_date=str(from_date), to_date=str(to_date), crop="Höstvete", pest="svartpricksjuka")
    
    # jbv_gdf = gpd.GeoDataFrame(gradings_df, geometry=gpd.points_from_xy(gradings_df.longitud, gradings_df.latitud), crs="EPSG:3006")
    # jbv_gdf["geometry"] = jbv_gdf["geometry"].to_crs("EPSG:4326")

    # # REMOVE ENTRIES NOT IN SWEDEN!
    # countries = ".\\geodata\\ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp"
    # world = gpd.read_file(countries)
    # sweden_boundary = world[world['NAME'] == 'Sweden']
    # sweden_boundary.to_crs("EPSG:4326")
    # jbv_gdf = jbv_gdf[jbv_gdf.geometry.within(sweden_boundary.geometry.union_all())]
    # jbv_gdf = jbv_gdf[jbv_gdf["delomrade"] == "Västmanland"]
    # print(jbv_gdf)
    # jbv_gdf, filtered_station_data = get_parameter_data_on_JBV_gdf(jbv_gdf, "19", from_date, to_date)

    # jbv_gdf, unique_keys = find_closest_station(jbv_gdf, "19", from_date, to_date)
    # print(unique_keys)

    # station_data = get_data_from_closest_station(unique_keys, "19", to_date)

    # fd = {key: get_date_interval_from_data(value, to_date, from_date) for key, value in station_data.items() if value is not None}
    # print(fd)

    # first_key = next(iter(fd))
    # head = fd[first_key]
