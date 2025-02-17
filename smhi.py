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

def find_closest_station(jbv_gdf, param_id, from_date, to_date):
    smhi_stations_gdf = get_all_stations_on_parameter_id(param_id=param_id, measuringStations="CORE", from_date=from_date, to_date=to_date)

    # Convert to estimated UTM CRS to achieve accuracy in distances
    utm_crs = jbv_gdf.estimate_utm_crs()
    jbv_gdf = jbv_gdf.to_crs(utm_crs)
    smhi_stations_gdf = smhi_stations_gdf.to_crs(utm_crs)
    # Keep station point for plotting
    smhi_stations_gdf["station_loc"] = smhi_stations_gdf["geometry"]

    # Perform Spatial join by nearest neighbour
    jbv_gdf = jbv_gdf.sjoin_nearest(smhi_stations_gdf, how="left")
    return jbv_gdf, pd.unique(jbv_gdf["key"])


def get_all_stations_on_parameter_id(param_id="19", measuringStations=None, from_date=None, to_date=None):
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



def get_data_from_closest_station(unique_keys, param_id, to_date):
    """
    Gets DataFrame with data from the closest station that has data within our time interval.

    Args:
        cl (List(float, float)): Sorted list of tuples with regards to the meter distance, tuple:(distance, station_key)
        param (int): parameter we want to query for.
        to_date (string): the upperbound date format: yyyy:mm:dd

    Returns:
        DataFrame: data of the closest station that has data wihtin our time interval. None if none of the stations had data within interval
    """
    return {key: get_for_station_for_parameter(key, param_id, to_date) for key in unique_keys}

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
    from_date = '2020-01-01'
    to_date = '2021-01-01'

    gradings_df = api.get_gradings(from_date=from_date, to_date=to_date)

    jbv_gdf = gpd.GeoDataFrame(gradings_df, geometry=gpd.points_from_xy(gradings_df.longitud, gradings_df.latitud), crs="EPSG:3006")
    jbv_gdf["geometry"] = jbv_gdf["geometry"].to_crs("EPSG:4326")

    jbv_gdf, unique_keys = find_closest_station(jbv_gdf, "19", from_date, to_date)
    print(unique_keys)

    station_data = get_data_from_closest_station(unique_keys, "19", to_date)

    fd = {key: get_date_interval_from_data(value, to_date, from_date) for key, value in station_data.items() if value is not None}
    print(fd)

    first_key = next(iter(fd))
    head = fd[first_key]
    print(head['Lufttemperatur'])

    def extract_pest(df: pd.DataFrame, pest: str) -> pd.DataFrame:
        """
        Look through df and extract data from 'graderingstillfalleList'.
        Extract records where 'skadegorare' == pest,
        Also adding other relevant information such as delomrade, lat-lon coordinates
        Needed since graderingslist is a list containing all measure instances for a given time period (year)
        """

        records = []

        for idx, row in df.iterrows():
            graderings_tillfallen = row.get('graderingstillfalleList', [])
            if not isinstance(graderings_tillfallen, list):
                continue

            #get area name, lat/long, etc
            delomrade_val = row.get('delomrade')
            lat_val = row.get('latitud')
            lon_val = row.get('longitud')
            crop = row.get('groda')
            sort = row.get('sort')

            for tillfalle in graderings_tillfallen:
                datum_str = tillfalle.get('graderingsdatum')
                datum_parsed = pd.to_datetime(datum_str, errors='coerce') if datum_str else None
                year = datum_parsed.year if pd.notnull(datum_parsed) else None

                graderingar = tillfalle.get('graderingList', [])
                for g in graderingar:
                    if g.get('skadegorare') == pest:
                        varde_str = g.get('varde')
                        value = pd.to_numeric(varde_str, errors='coerce') if varde_str is not None else None
                        measuring_method = g.get('matmetod')

                        records.append({
                            "index_in_main_df": idx,
                            "delomrade": delomrade_val,
                            "latitud": lat_val,
                            "longitud": lon_val,
                            "date": datum_parsed,
                            "year": year,
                            "crop": crop,
                            "type": sort,
                            "pest": pest,
                            "measuring_method": measuring_method,
                            "value": value
                        })

        return pd.DataFrame(records)
    

    df_bf = extract_pest(jbv_gdf.head(), 'Bladlus')
    df_bf = df_bf.sort_values(by='date')

    head['Lufttemperatur'] = pd.to_numeric(head['Lufttemperatur'], errors='coerce')

    vector_size = 11
    stddev = 2

    weights = np.exp(-0.5 * (np.linspace(-stddev, stddev, vector_size) ** 2))
    weights = weights / np.sum(weights)

    smoothed_temperature = np.convolve(head['Lufttemperatur'], weights, mode='same')
    head['Lufttemperatur'] = smoothed_temperature

    start_range = pd.to_datetime(from_date)
    end_range = pd.to_datetime(to_date)

    smoothed_temperature = np.convolve(df_bf['value'], weights, mode='same')
    df_bf['value'] = smoothed_temperature
    filtered_data = head[(head['Från Datum Tid (UTC)'] >= start_range) & (head['Till Datum Tid (UTC)'] <= end_range)]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Från Datum Tid (UTC)'], filtered_data['Lufttemperatur'], linestyle='-', color='b', label='Temperature (°C)')
    plt.plot(df_bf['date'], df_bf['value'], color='g', label='Bladlus (antal)')

    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Over Time (Filtered)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.show()
