import pandas as pd
import numpy as np
from pyproj import Transformer

def feature_extraction(data_df, crop, pest):
    """
    Extracts grading data for a crop and pest from a nested dataframe structure.

    Args:
        data_df (pd.DataFrame): The input dataframe containing grading information.
        crop (str): The crop associated with the grading data.
        pest (str): The pest associated with the grading data.

    Returns:
        pd.DataFrame: A cleaned and sorted dataframe containing extracted grading data.
    """
    extracted_data = []
    for _, row in data_df.iterrows():
        graderingstillfalle_list = row['graderingstillfalleList']

        for entry in graderingstillfalle_list:
            for grading in entry['graderingList']:
                extracted_data.append({
                    'crop': crop,
                    'skadegorare': pest,  
                    'graderingsdatum': entry['graderingsdatum'],
                    'varde': grading['varde'],
                    'utvecklingsstadium': entry['utvecklingsstadium'],
                })

    return pd.DataFrame(extracted_data).sort_values(by='graderingsdatum')

def remove_outside_sweden_coords(data_df):
    """Drops datapoints (rows) with coordinates located outside of Sweden
    
    Args:
        data_df (pd.DataFrame): The input dataframe containing location data.

    Returns:
        data_df (pd.DataFrame): The ouput dataframe without containing missing location data.
    """
    # Assumes WGS84 Coordinate System
    min_lat, max_lat = 55, 70
    min_long, max_long = 10, 25
    valid_lat = (min_lat < data_df['latitud']) & (data_df['latitud'] < max_lat)
    valid_long = (min_long < data_df['longitud']) & (data_df['longitud'] < max_long)

    filtered_df = data_df[valid_long & valid_lat]
    return filtered_df

def drop_rows_no_coords(data_df):
    """Drops rows with null values in 'latitud' or 'longitud' columns.
    
    Args:
        data_df (pd.DataFrame): The input dataframe containing location data.

    Returns:
        data_df (pd.DataFrame): The ouput dataframe without containing missing location data.
    """
    return data_df.dropna(subset=['latitud', 'longitud'])

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
        cleaned_and_sorted_coords = remove_outside_sweden_coords(sorted_combination_counts)
        return cleaned_and_sorted_coords
    else:
        sorted_combination_counts = get_uniq_plantation_coord(data_df, selector=0)
        top_combination = sorted_combination_counts.head(selector)
        top_latitud = top_combination['latitud'].values[0]
        top_longitud = top_combination['longitud'].values[0]

        return top_latitud, top_longitud
    
def get_plantation_by_coord(data_df, latitud, longitud):
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

def transform_date(data_df):
    """
    Converts 'graderingsdatum' to datetime and introduces NaN values for large gaps.
    Averages data if there are multiple measurements during the same week.

    Args:
        data_df (pd.DataFrame): The input dataframe containing date-related data.

    Returns:
        pd.DataFrame: The modified dataframe with transformed date values and missing data handled.
    """
    # convert type
    data_df['graderingsdatum'] = pd.to_datetime(data_df['graderingsdatum'])

    # prune nonexisting dates
    threshold = pd.Timedelta(days=30)
    gap = data_df['graderingsdatum'].diff() > threshold
    data_df.loc[gap, ['varde', 'utvecklingsstadium']] = np.nan

    # average by week
    data_df.set_index('graderingsdatum', inplace=True)
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    data_df = data_df[numeric_cols].resample('W-MON').mean()
    data_df.reset_index(inplace=True)

    return data_df

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
    df.loc[:, 'latitud'], df.loc[:, 'longitud'] = transformer.transform(df['latitud'].values, df['longitud'].values)
    return df