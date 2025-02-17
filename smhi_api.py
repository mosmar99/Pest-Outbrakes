import os
import requests
import pandas as pd
import geopandas as gpd

SMHI_URI = {
    "all_parameters": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter.json",
    "all_stations": "https://opendata-download-metobs.smhi.se/api/version/latest/parameter",
}

def fetch_data_on_key(endpoint, json_key):
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[json_key])
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_all_parameters():
    df = fetch_data_on_key(SMHI_URI["all_parameters"], "resource")
    df = df.drop(['link', 'geoBox'], axis=1)
    return df

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

if __name__ == "__main__":
    df = get_all_parameters()
    print(df)
    gdf = get_all_stations_on_parameter_id()
    print(gdf)