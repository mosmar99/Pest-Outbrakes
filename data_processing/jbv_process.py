import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def feature_extraction(data_df, wanted_features):
    """
    Extracts user specified feature from json table

    Args:
        data_df (json): Raw json api inputs, for a specific crop and pest.
        wanted_features (list of strings): Simply a list of the feature you want to extract

    Returns:
        pd.DataFrame: A cleaned and sorted dataframe containing extracted grading data.
    """

    data_df = pd.DataFrame(data_df)

    exploded_graderingstillfalleList = data_df.explode('graderingstillfalleList')
    normalized_graderingstillfalleList = pd.json_normalize(exploded_graderingstillfalleList['graderingstillfalleList'])

    normalized_graderingList = normalized_graderingstillfalleList.explode('graderingList')
    normalized_graderingList = pd.json_normalize(normalized_graderingList['graderingList'])

    combined_df = pd.concat([exploded_graderingstillfalleList.reset_index(drop=True), normalized_graderingstillfalleList.reset_index(drop=True)], axis=1)
    combined_df = pd.concat([combined_df.reset_index(drop=True), normalized_graderingList.reset_index(drop=True)], axis=1)
    combined_df = combined_df.drop(columns=['graderingstillfalleList', 'graderingList'])

    desired_features_df = combined_df[wanted_features]
    desired_features_df['graderingsdatum'] = pd.to_datetime(desired_features_df['graderingsdatum'])

    return desired_features_df

def remove_outside_sweden_coordinates(data_gdf):
    """Drops datapoints (rows) with coordinates located outside of Sweden
    
    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing location data with a geometry column.

    Returns:
        gpd.GeoDataFrame: The output GeoDataFrame exclusively containing Swedish location data.
    """
    # Assumes WGS84 Coordinate System
    min_lat, max_lat = 55, 70
    min_long, max_long = 10, 25

    # (x, y) = (longitude, latitude)
    valid_lat = (data_gdf.geometry.y >= min_lat) & (data_gdf.geometry.y <= max_lat)
    valid_long = (data_gdf.geometry.x >= min_long) & (data_gdf.geometry.x <= max_long)

    filtered_df = data_gdf[valid_long & valid_lat]
    return filtered_df

def drop_rows_with_missing_values(data_df):
    """Drops rows with null values in 'latitud' or 'longitud' columns.
    
    Args:
        data_df (pd.DataFrame): The input dataframe containing location data.

    Returns:
        data_df (pd.DataFrame): The ouput dataframe without containing missing location data.
    """
    return data_df.dropna(subset=['graderingsdatum', 'varde', 'utvecklingsstadium', 'latitud', 'longitud'])

def drop_duplicate_rows(data_df):
    """Drops duplicate rows from the dataframe.

    Args:
        data_df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with duplicate rows removed.
    """
    return data_df.drop_duplicates()


def introduce_nan_for_large_gaps(data_gdf, days=30):
    """
    Introduces NaN values for large gaps in the 'graderingsdatum' column to prevent line drawing in graphs.

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing date-related data.

    Returns:
        gpd.GeoDataFrame: The modified GeoDataFrame with NaN values introduced for large gaps.
    """
    # Define the threshold for large gaps
    threshold = pd.Timedelta(days)
    gap = data_gdf['graderingsdatum'].diff() > threshold
    data_gdf.loc[gap, ['varde', 'utvecklingsstadium']] = np.nan

    return data_gdf

import pandas as pd

def aggregate_data_for_plantations(data_gdf, time_period='week'):
    pass

def sweref99tm_to_wgs84(data_df):
    """
    Converts coordinates from SWEREF 99 TM (EPSG:3006) to WGS 84 (EPSG:4326) and updates the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing SWEREF 99 TM coordinate data.

    Returns:
        pd.DataFrame: The transformed dataframe with updated WGS 84 coordinates.
    """
    sweref99tm = 'EPSG:3006'
    wgs84 = 'EPSG:4326'

    # use points_from_xy to computationally effective manenr convert columns into Shapley Points
    data_df["geometry"] = gpd.points_from_xy(data_df['longitud'], data_df['latitud'])

    # wrap pd in gpd and specify current coordinate system
    g_df_old = gpd.GeoDataFrame(data_df, geometry="geometry", crs=sweref99tm)

    # specify coordinate system we want to convert to
    g_df_new = g_df_old.to_crs(wgs84) # to crs = to Coordinnate Reference System

    g_df = g_df_new.drop(columns=['latitud', 'longitud'])

    return g_df

def clean_coordinate_format(df):
    """
    Trims the 'latitud' and 'longitud' columns to a specified number of decimal places.
    The 'latitud' column is truncated to 7 characters, and the 'longitud' column is truncated to 6 characters.

    Args:
        df (pd.DataFrame): The input dataframe containing 'latitud' and 'longitud' columns.

    Returns:
        pd.DataFrame: The dataframe with updated 'latitud' and 'longitud' columns.
    """
    latitud_trimmed = df['latitud'].astype(str).str[:7]
    longitud_trimmed = df['longitud'].astype(str).str[:6]

    # try converting coordiantes to numbers, if conversion fails, add NaN
    latitud_trimmed = pd.to_numeric(latitud_trimmed, errors='coerce')
    longitud_trimmed = pd.to_numeric(longitud_trimmed, errors='coerce')

    # remove rows that failed conversion
    df = df[latitud_trimmed.notna() & longitud_trimmed.notna()]

    return df

def get_all_unique_coordinates(data_gdf):
    """
    Returns all unique geometry coordinates from the GeoDataFrame.

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing geometry data.

    Returns:
        pd.Dataframe: A list of unique geometry coordinates.
    """
    unique_coordinates = set()

    for geom in data_gdf.geometry:
        unique_coordinates.update(geom.coords)

    return pd.DataFrame(unique_coordinates, columns=['longitude', 'latitude'])


def get_uniq_plantation_coord(data_df, selector=1):
    """
    Retrieves unique latitude-longitude combinations, sorted by occurrence count.

    Args:
        data_df (pd.DataFrame): The input dataframe containing location data.
        selector (int): If 0, returns a sorted dataframe of descending coordinate frequencies; 
                        otherwise, returns frequency coordinates from a list of descending coordinate frequencies.
    Returns:
        Tuple[float, float]: Tuple of latitude and longitude
    """
    if selector == 0:
        combination_counts = data_df.groupby(['latitud', 'longitud']).size().reset_index(name='Count')
        sorted_combination_counts = combination_counts.sort_values(by='Count', ascending=False)
        return sorted_combination_counts
    else:
        sorted_combination_counts = get_uniq_plantation_coord(data_df, selector=0)
        top_combination = sorted_combination_counts.head(selector)
        top_latitud = top_combination['latitud'].values[0]
        top_longitud = top_combination['longitud'].values[0]

        return top_latitud, top_longitud
    
def get_plantation_by_coordinates(data_df, latitud, longitud):
    """
    Filters the dataframe for rows matching the given latitude and longitude.

    Args:
        data_df (pd.DataFrame): The input dataframe containing location data.
        latitud (float): The latitude to filter by.
        longitud (float): The longitude to filter by.

    Returns:
        pd.DataFrame: A dataframe containing only rows that match the specified coordinates.
    """
    filtered_rows_df = data_df[(data_df['latitud'] == latitud) & (data_df['longitud'] == longitud)]
    return filtered_rows_df