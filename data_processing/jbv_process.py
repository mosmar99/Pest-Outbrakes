import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

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

def drop_duplicates(data_df):
    """Drops duplicate rows & columns from the dataframe.

    Args:
        data_df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with duplicate rows & columns removed.
    """
    data_df = data_df.drop_duplicates()
    return data_df.loc[:, ~data_df.columns.duplicated()]


def introduce_nan_for_large_gaps(data_gdf, gap_days=30):
    """
    Introduces NaN values for large gaps in the 'graderingsdatum' column to prevent line drawing in graphs.

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing date-related data.

    Returns:
        gpd.GeoDataFrame: The modified GeoDataFrame with NaN values introduced for large gaps.
    """
    threshold = pd.Timedelta(days=gap_days)
    for geom, group in data_gdf.groupby('geometry'):
        gap = group['graderingsdatum'].diff() > threshold
        mask = data_gdf['geometry'] == geom
        data_gdf.loc[mask & gap, ['varde', 'utvecklingsstadium']] = np.nan
    data_gdf = data_gdf.sort_values(by='graderingsdatum').reset_index(drop=True)
    return data_gdf

def aggregate_data_for_plantations(data_gdf, time_period='W-MON'):
    """
    Aggregates the feature columns: varde & utvecklingsstadium with regards to specified timeframe

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing information
        time_period (string): String specifiying time_period, should be either 'W-MON', 'M' or 'Y-DEC'

    Returns:
        data_gdf (gpd.GeoDataFrame): Output GeoDataFrame containing aggregated values.
    """
    data_gdf['time_period'] = data_gdf['graderingsdatum'].dt.to_period(time_period)

    agg_dict = {
        'varde': 'mean',
        'utvecklingsstadium': 'mean',
        'groda': 'first',
    }
    aggregated_df = data_gdf.groupby(['time_period', 'geometry']).agg(agg_dict).reset_index()

    skadegorare_series = data_gdf.groupby(['time_period', 'geometry'])['skadegorare'].first().reset_index()
    aggregated_df = pd.merge(aggregated_df, skadegorare_series[['time_period', 'geometry', 'skadegorare']], on=['time_period', 'geometry'], how='left')


    aggregated_df['graderingsdatum'] = aggregated_df['time_period'].dt.start_time
    aggregated_df = aggregated_df.drop(columns=['time_period'])

    return aggregated_df

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
        data_gdf (gpd.GeoDataFrame): The dataframe with updated 'latitud' and 'longitud' columns.
    """
    latitud_trimmed = df['latitud'].astype(str).str[:7]
    longitud_trimmed = df['longitud'].astype(str).str[:6]

    # try converting coordiantes to numbers, if conversion fails, add NaN
    latitud_trimmed = pd.to_numeric(latitud_trimmed, errors='coerce')
    longitud_trimmed = pd.to_numeric(longitud_trimmed, errors='coerce')

    # remove rows that failed conversion
    df = df[latitud_trimmed.notna() & longitud_trimmed.notna()]

    return df

def get_most_frequent_plantation(data_gdf):
    """
    Retrieves all rows from the GeoDataFrame that correspond to the most frequently occurring geometry.

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing plantation data.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all rows that match the most frequent geometry.
    """
    grouped = data_gdf.groupby('geometry').size().reset_index(name='count')
    most_frequent_group = grouped.loc[grouped['count'].idxmax()]
    result = data_gdf[data_gdf['geometry'] == most_frequent_group['geometry']]
    return result
    
def get_plantation_by_coordinates(data_gdf, point):
    """
    Filters the dataframe for rows matching the given latitude and longitude.

    Args:
        data_gdf (pd.DataFrame): The input dataframe containing location data.
        point (Shapley Point): The latitude, lonitude to filter by.

    Returns:
        filtered_rows_gdf (gpd.GeoDataFrame): A dataframe containing only rows that match the specified coordinates.
    """
    latitud = point.y
    longitud = point.x
    
    filtered_rows_gdf = data_gdf[(data_gdf['latitud'] == latitud) & (data_gdf['longitud'] == longitud)]
    return filtered_rows_gdf