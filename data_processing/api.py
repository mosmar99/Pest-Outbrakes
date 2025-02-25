
import requests
import pandas as pd
import geopandas as gpd
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

import os

import csv

def fetch_data(endpoint, auth=None):
    """
    Retrieves data from an API endpoint using HTTP Basic Authentication.

    Args:
        endpoint (str): The API endpoint URL to fetch data from.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing the fetched data if successful, otherwise None.
    """

    response = requests.get(endpoint, auth=auth)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def download_csv(endpoint, directory, filename):
    """
    Download csv data to local {filename}.csv. will create directory if it does not exist.

    Args:
        endpoint (String): Station key in SMHI API.
        directory (String): Relative path string to diretory, ex. ".\\smhi_data".
        filename (String): String of the name the file should be saved with excluding extension, ex. "data".
    
    Returns:
        String: Path to the downloaded file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, f"{filename}.csv")

    with requests.get(endpoint, stream=True) as r:
        lines = (line.decode('utf-8') for line in r.iter_lines())
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in csv.reader(lines):
                writer.writerow(row)
    
    return filepath