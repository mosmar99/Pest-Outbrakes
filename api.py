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
    """
    Retrieves data from an API endpoint using HTTP Basic Authentication.

    Args:
        endpoint (str): The API endpoint URL to fetch data from.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing the fetched data if successful, otherwise None.
    """
    response = requests.get(endpoint, auth=HTTPBasicAuth(USERNAME, PASSWORD))

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_all_pests():
    """
    Fetches and returns a dataframe of all pests from the API.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing all pest data if successful, otherwise None.
    """
    return fetch_data(URI["all_pests"])

def get_all_crops():
    """
    Fetches and returns a dataframe of all crops from the API.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing all crop data if successful, otherwise None.
    """
    return fetch_data(URI["all_crops"])

def get_crops_linked_to_gradings():
    """
    Retrieves a dataframe of crops associated with grading data.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing crops linked to grading data if successful, otherwise None.
    """
    return fetch_data(URI["crops_linked_to_gradings"])

def get_pests_linked_to_gradings():
    """
    Retrieves a dataframe of pests associated with grading data.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing pests linked to grading data if successful, otherwise None.
    """
    return fetch_data(URI["pests_linked_to_gradings"])

def get_gradings(from_date="2019-08-04", to_date="2019-11-22", crop="Potatis", pest="Bladlus"):
    """
    Fetches grading data for a date range, crop, and pest from the API.

    Args:
        from_date (str): The start date for filtering data (format YYYY-MM-DD).
        to_date (str): The end date for filtering data (format YYYY-MM-DD).
        crop (str): The crop name to retrieve grading data for.
        pest (str): The pest name to retrieve grading data for.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing grading data if successful, otherwise None.
    """
    endpoint = f"{URI['gradings']}?fran={from_date}&till={to_date}&groda={crop}&skadegorare={pest}"
    return fetch_data(endpoint)

def get_update_information():
    """
    Retrieves and returns the latest update information from the API.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing update information if successful, otherwise None.
    """
    return fetch_data(URI["update_information"])

def sweref99tm_to_wgs84(df):
    """
    Converts coordinates from SWEREF 99 TM (EPSG:3006) to WGS 84 (EPSG:4326) and updates the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing SWEREF 99 TM coordinate data.

    Returns:
        pd.DataFrame: The transformed dataframe with updated WGS 84 coordinates.
    """
    sweref99tm = 'EPSG:3006'
    wgs84 = 'EPSG:4326'
    transformer = Transformer.from_crs(sweref99tm, wgs84, always_xy=False)
    df['latitud'], df['longitud'] = transformer.transform(df['latitud'].values, df['longitud'].values)
    return df