import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pyproj import Transformer
from requests.auth import HTTPBasicAuth

URI = {
    "all_crops": "https://api.jordbruksverket.se/rest/povapi/groda",
    "all_pests": "https://api.jordbruksverket.se/rest/povapi/skadegorare",
    "pests_linked_to_gradings": "https://api.jordbruksverket.se/rest/povapi/skadegorare/alla",
    "crops_linked_to_gradings": "https://api.jordbruksverket.se/rest/povapi/groda/alla",
    "gradings": "https://api.jordbruksverket.se/rest/povapi/graderingar",
    "update_information": "https://api.jordbruksverket.se/rest/povapi/uppdateringsinformation",
    "smhi": "https://opendata-download-metobs.smhi.se/api/"
}

load_dotenv()
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")
print(USERNAME, PASSWORD)
def fetch_data(endpoint):
    response = requests.get(endpoint, auth=HTTPBasicAuth(USERNAME, PASSWORD))

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_all_pests():
    return fetch_data(URI["all_pests"])

def get_all_crops():
    return fetch_data(URI["all_crops"])

def get_crops_linked_to_gradings():
    return fetch_data(URI["crops_linked_to_gradings"])

def get_pests_linked_to_gradings():
    return fetch_data(URI["pests_linked_to_gradings"])

def get_gradings(from_date="2019-08-04", to_date="2019-11-22", crop="Potatis", pest="Bladlus"):
    endpoint = f"{URI['gradings']}?fran={from_date}&till={to_date}&groda={crop}&skadegorare={pest}"
    return fetch_data(endpoint)

def get_update_information():
    return fetch_data(URI["update_information"])

def sweref99tm_to_wgs84(df):
    sweref99tm = 'EPSG:3006'
    wgs84 = 'EPSG:4326'
    transformer = Transformer.from_crs(sweref99tm, wgs84, always_xy=False)
    df['latitud'], df['longitud'] = transformer.transform(df['latitud'].values, df['longitud'].values)
    return df