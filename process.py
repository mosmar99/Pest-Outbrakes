import pandas as pd
import numpy as np

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

def get_uniq_plant_coord(data_df, selector=1):
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
        combination_counts = data_df.groupby(['latitud', 'longitud']).size().reset_index(name='Count')
        sorted_combination_counts = combination_counts.sort_values(by='Count', ascending=False)
        top_combination = sorted_combination_counts.head(selector)
        top_latitud = top_combination['latitud'].values[0]
        top_longitud = top_combination['longitud'].values[0]

        return top_latitud, top_longitud
    
def coordinate_rowfilter(data_df, latitud, longitud):
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

    Args:
        data_df (pd.DataFrame): The input dataframe containing date-related data.

    Returns:
        pd.DataFrame: The modified dataframe with transformed date values and missing data handled.
    """
    # convert type
    data_df['graderingsdatum'] = pd.to_datetime(data_df['graderingsdatum'])

    # prune nonexisting dates
    threshold = pd.Timedelta(days=14)
    gap = data_df['graderingsdatum'].diff() > threshold
    data_df.loc[gap, ['varde', 'utvecklingsstadium']] = np.nan

    return data_df