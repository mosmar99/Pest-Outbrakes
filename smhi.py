"""
Imports
"""
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

"""
Fetches data from SMHI endpoint, similar to the one found in api.py
returns DataFrame containing data.
"""
@staticmethod
def fetch_data(endpoint):
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

"""
Takes two points as paramters and counts meter distance between the two.
"""
@staticmethod
def get_meter_distance(point1, point2):
    return geodesic(point1[::-1], point2[::-1]).meters

"""
Takes two DataFrames one contaning our stations and the second one being a DataFrame containing a single gradings
Returns a sorted list of tuples with regards to the distance. Tuple = (distance, station_key)
"""
def find_closest_station(df_stations, df_single_gradings):
    station_keys = df_stations['key']
    to_find = float(df_single_gradings['longitud']), float(df_single_gradings['latitud'])
    long_lat = list(zip(df_stations['longitude'], df_stations['latitude']))

    cl = []
    for i, station in enumerate(station_keys):
        dist = get_meter_distance(to_find, long_lat[i])
        cl.append((dist, station))

    cl = sorted(cl, key=lambda x: x[0])
    return cl

"""
Gets available stations for some parameter, see smhi documentation for available parameters
Returns DataFrame contaning the available stations.
"""
def get_stations_per_parameter(param):
    df = None
    try:
        endpoint = f"https://opendata-download-metobs.smhi.se/api/version/latest/parameter/{param}.json"
        df = fetch_data(endpoint=endpoint)
    except:
        print("Invalid parameter")
    return df

"""
"""
@staticmethod
def get_encoding(ret):
    raw_data = ret.content
    encoding_info = chardet.detect(raw_data)
    decoded_text = raw_data.decode(encoding_info['encoding'], errors='replace')
    data = pd.read_csv(StringIO(decoded_text), on_bad_lines='skip')
    data.to_csv('station_data.csv', index=False)
    return encoding_info

"""
"""
@staticmethod
def get_for_station_for_parameter(station_key, parameter):
    # Get the data for specified parameter from specified station
    endpoint = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{parameter}/station/{station_key}/period/corrected-archive/data.csv"
    ret = requests.get(endpoint)
    if ret.status_code == 200:
        # Get encoding and read data
        encoding_info = get_encoding(ret)
        
        with open('station_data.csv', 'r', encoding=encoding_info['encoding']) as file:
            lines = file.readlines()
        for line in lines:
            line.strip()

        # get final date in the dataframe object
        ending_date = lines[-1].split(';')[1].split(" ")[0]
        ending_date = ending_date.split('-')
        td = to_date.split('-')

        # convert dates from strings to integers
        try:
            for i in range(len(td)):
                td[i] = int(td[i])
                ending_date[i] = int(ending_date[i])
        except:
            return None

        # remove header, as well as split data into header & data
        index = -1
        for i, line in enumerate(lines):
            if 'Från Datum Tid (UTC)' in line:
                index = i
                break
        lines = lines[index:]
        head = lines[0].split(";")
        lines = lines[1:] 

        data = [line.split(';') for line in lines]  

        cleaned_data = []
        for d in data:
            idx = d[0].find(',')
            cleaned_line = d[0][idx+1:].split(';') 
            cleaned_data.append(cleaned_line + d[1:]) 
        
        data = cleaned_data

        df = pd.DataFrame(data, columns=head)  

        df.to_csv('station_data.csv', sep=';', index=False)


        if datetime.date(ending_date[0], ending_date[1], ending_date[2]) < datetime.date(td[0], td[1], td[2]):
            return None
        else:
            return df
 
    else:
        print(f"Error fetching data: {ret.status_code}")

"""
"""
def get_data_from_closest_station(cl, param):
    for index, (_ , station_key) in enumerate(cl):
        ret = get_for_station_for_parameter(station_key, param)
        if ret is not None:
            return ret



"""
For testing
"""
if __name__ == "__main__":
    df_stations = get_stations_per_parameter(19)
    from_date = '2020-01-01'
    to_date = '2021-01-01'
    pest = 'Bladfläcksvampar'
    gradings_df = api.get_gradings(from_date=from_date, to_date=to_date)
    gradings_df = api.sweref99tm_to_wgs84(gradings_df)
    first_row = gradings_df.iloc[0]

    cl = find_closest_station(df_stations, first_row)
    ret = get_data_from_closest_station(cl, 19)
    print(ret['Lufttemperatur'])
