import requests
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

from io import StringIO

parent_dir = os.path.relpath("../")

sys.path.append(parent_dir)

import api

from geopy.distance import geodesic

import datetime

import numpy as np

import chardet


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

@staticmethod
def get_meter_distance(point1, point2):
    """
    Takes two points as paramters and counts meter distance between the two.
    
    Args:
        point1 (float, float): 
        point2 (float, float):

    Returns:
        float: distance in meters.
    """
    return geodesic(point1[::-1], point2[::-1]).meters

def find_closest_station(df_stations, df_single_gradings):
    """ 
    Finds closest stations to JBV dataframe point
    
    Args:
        df_stations (DataFrame): DataFrame containing our SMHI Stations
        df_single_gradings (DataFrame): DataFrame Containing a single JBV datapoint
    
    Return:
        List(float, int): Sorted list of tuples with regards to the meter value distance, tuple:(distance, station_key)
    """

    station_keys = df_stations['key']
    to_find = float(df_single_gradings['longitud']), float(df_single_gradings['latitud'])
    long_lat = list(zip(df_stations['longitude'], df_stations['latitude']))

    cl = []
    for i, station in enumerate(station_keys):
        dist = get_meter_distance(to_find, long_lat[i])
        cl.append((dist, station))

    cl = sorted(cl, key=lambda x: x[0])
    return cl

def get_stations_per_parameter(param):
    """ 
    Returns all stations for some specified parameter
    
    Args:
        param (int): SMHI Parameter
    
    Return:
        DataFrame: Contaning available stations for specified parameter
    """

    df = None
    try:
        endpoint = f"https://opendata-download-metobs.smhi.se/api/version/latest/parameter/{param}.json"
        df = fetch_data(endpoint=endpoint)
    except:
        print("Invalid parameter")
    return df


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

@staticmethod
def is_data_up_to_date(lines, to_date):
    """
    Check if the last available data is up-to-date
    
    Args:
        lines (List(string)): The lines of our df object
        to_date (string): the upperbound date format: yyyy:mm:dd

    Returns:
        Bool: True if the ending_date is post our to_date, false if not.
    """
    try:
        ending_date = lines[-1].split(';')[1].split(" ")[0].split('-')
        ending_date = list(map(int, ending_date))
        td = list(map(int, to_date.split('-')))
        return datetime.date(ending_date[0], ending_date[1], ending_date[2]) >= datetime.date(td[0], td[1], td[2])
    except:
        return False

def get_for_station_for_parameter(station_key, parameter, to_date):
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

    if not is_data_up_to_date(lines, to_date):
        return None

    df.to_csv('station_data.csv', sep=';', index=False)
    return df



def get_data_from_closest_station(cl, param, to_date):
    """
    Gets DataFrame with data from the closest station that has data within our time interval.

    Args:
        cl (List(float, float)): Sorted list of tuples with regards to the meter distance, tuple:(distance, station_key)
        param (int): parameter we want to query for.
        to_date (string): the upperbound date format: yyyy:mm:dd

    Returns:
        DataFrame: data of the closest station that has data wihtin our time interval. None if none of the stations had data within interval
    """
    
    for index, (_ , station_key) in enumerate(cl):
        df = get_for_station_for_parameter(station_key, param, to_date)
        if df is not None:
            return df
    return None

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


if __name__ == "__main__":
    """
    Example usage of module
    """

    # Get all stations for parameter 19 (Lufttemperatur)
    df_stations = get_stations_per_parameter(19)

    # Specifications of dates, we are interested in
    from_date = '2020-01-01'
    to_date = '2021-01-01'

    # Get Gradings from JBV API
    gradings_df = api.get_gradings(from_date=from_date, to_date=to_date)
    gradings_df = api.sweref99tm_to_wgs84(gradings_df)

    # Extract the first row / data point
    first_row = gradings_df.iloc[0]

    cl = find_closest_station(df_stations, first_row)
    ret = get_data_from_closest_station(cl, 19, to_date)
    fd  = get_date_interval_from_data(ret, to_date, from_date)
