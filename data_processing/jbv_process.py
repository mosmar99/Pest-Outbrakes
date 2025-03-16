import pandas as pd
import numpy as np
import geopandas as gpd
import pandas as pd

pd.options.mode.chained_assignment = None

def feature_extraction(data_df, wanted_features=None):
    """
    Extracts user specified feature from json table

    Args:
        data_df (json): Raw json api inputs, for a specific crop and pest.
        wanted_features (list of strings): Simply a list of the feature you want to extract

    Returns:
        pd.DataFrame: A cleaned and sorted dataframe containing extracted grading data.
    """
    data_df = [{**d, 'Series_id':i} for i, d in enumerate(data_df)]
    df = pd.json_normalize(
        data_df,
        ['graderingstillfalleList', 'graderingList'],
        ['Series_id',
         'delomrade',
         'ekologisk',
         'forforfrukt',
         'forfrukt',
         'groda',
         'jordart',
         'lan',
         'latitud',
         'longitud',
         'plojt',
         'sadatum',
         'sort',
         'skordear',
         'broddbehandling',
         'utplaceringsdatumFalla',
         ['graderingstillfalleList', 'graderingsdatum'],
         ['graderingstillfalleList', 'graderingstyp'],
         ['graderingstillfalleList', 'utvecklingsstadium']
        ],
        errors='ignore',
    )
    df = df.rename(columns={'graderingstillfalleList.graderingsdatum': 'graderingsdatum',
                            'graderingstillfalleList.graderingstyp': 'graderingstyp', 
                            'graderingstillfalleList.utvecklingsstadium': 'utvecklingsstadium'})
    
    if wanted_features is not None:
        df = df[wanted_features]
    df['graderingsdatum'] = pd.to_datetime(df['graderingsdatum'])

    return df

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
    return data_df.dropna(subset=['graderingsdatum', 'utvecklingsstadium', 'latitud', 'longitud'])

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

def aggregate_data_for_plantations(data_gdf, time_period='W-SUN'):
    """
    Aggregates the feature columns: varde & utvecklingsstadium with regards to specified timeframe

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing information
        time_period (string): String specifiying time_period, should be either 'W-SUN', 'M' or 'Y-DEC'

    Returns:
        data_gdf (gpd.GeoDataFrame): Output GeoDataFrame containing aggregated values.
    """

    data_gdf['time_period'] = data_gdf['graderingsdatum'].dt.to_period(time_period)

    agg_dict = {
        'varde': 'mean',
        'utvecklingsstadium': 'mean',
        'groda': 'first',
        'sort': 'first',
    }
    aggregated_df = data_gdf.groupby(['time_period', 'geometry']).agg(agg_dict).reset_index()

    skadegorare_series = data_gdf.groupby(['time_period', 'geometry'])['skadegorare'].first().reset_index()
    aggregated_df = pd.merge(aggregated_df, skadegorare_series[['time_period', 'geometry', 'skadegorare']], on=['time_period', 'geometry'], how='left')
    aggregated_df = gpd.GeoDataFrame(aggregated_df, geometry='geometry')

    aggregated_df['graderingsdatum'] = aggregated_df['time_period'].dt.start_time

    aggregated_df = aggregated_df.drop(columns=['time_period'])

    return aggregated_df

def aggregate_data_for_series(data_gdf, agg_dict, time_period='W-SUN'):
    """
    Aggregates the feature columns from agg_dict with regards to specified timeframe and Series_id
    Effectively realigning data on mondays for merge with SMHI data

    Args:
        data_gdf (gpd.GeoDataFrame): The input GeoDataFrame containing information
        agg_dict (dict): Dictionary for aggreagation
        time_period (string): String specifiying time_period, should be either 'W-SUN', 'M' or 'Y-DEC'

    Returns:
        data_gdf (gpd.GeoDataFrame): Output GeoDataFrame containing aggregated values.
    """

    data_gdf['time_period'] = data_gdf['graderingsdatum'].dt.to_period(time_period)
    aggregated_df = data_gdf.groupby(['time_period', 'Series_id']).agg(agg_dict).reset_index()

    aggregated_df['graderingsdatum'] = aggregated_df['time_period'].dt.start_time

    aggregated_df = aggregated_df.drop('time_period', axis=1)

    return aggregated_df

def interpolate_jbv_data(data_df:pd.DataFrame, cols_to_interpolate=['varde', 'utvecklingsstadium'], freq='D', method='linear', order=None):
    """
    Introduces missing dates in the dataset, interpolates missing values for specified columns, and applies forward filling for the rest of the data within each group defined by 'Series_id'.
    
    This function operates on a DataFrame that contains multiple series (identified by the 'Series_id' column) and performs interpolation on specified columns for each group. It also fills any missing dates and forward-fills other columns that are not part of the interpolation.

    Parameters:
    - data_df (pd.DataFrame): The input pandas DataFrame containing the data. It must include a 'Series_id' column for grouping, a 'graderingsdatum' column for dates, and the columns that need to be interpolated.
    - cols_to_interpolate (list): A list of column names to be interpolated. Default is `['varde', 'utvecklingsstadium']`.
    - freq (str): Frequency for the date range when filling in missing dates. Default is `'D'` (daily). You can specify other frequencies, such as `'H'` for hourly or `'M'` for monthly.
    - method (str): Interpolation method to use. Default is `'linear'`. Other options are `'polynomial'`, `'spline'`.
    - order (int or None): The order of interpolation for methods like polynomial or spline interpolation.

    Returns:
    - pd.DataFrame: A DataFrame with interpolated values for the specified columns, other columns forward Filled, Appends column meassured_value which indicates if values are actual meassurement values.
    """
    result_dfs = []
    # Group by the Series_id column and process each group individually
    for _, group in data_df.groupby('Series_id'):
        # Create a full date range for the current group from the min to max date
        full_date_range = pd.date_range(start=group['graderingsdatum'].min(), end=group['graderingsdatum'].max(), freq=freq).to_series().rename('graderingsdatum')

        group = pd.merge(full_date_range, group, how='left', on='graderingsdatum', indicator='meassured_value')
        group['meassured_value'] = group['meassured_value'] == 'both'
        # print(group[cols_to_interpolate])

        group[cols_to_interpolate] = group[cols_to_interpolate].interpolate(method=method, limit_area='inside', order=order)
        group = group.ffill(limit_area='inside').infer_objects()
        result_dfs.append(group)
    
    # Concatenate all processed groups back into a single DataFrame
    return pd.concat(result_dfs).reset_index(drop=True)

def filter_uncommon_or_short_span(data_df, start_min=15, start_max=20, stop_min=26, stop_max=28):
    """
    Filters the dataframe to retain only series within a specified time span based on ISO calendar weeks.

    Args:
        data_df (pd.DataFrame): Input dataframe containing 'Series_id' and 'graderingsdatum' (datetime).
        start_min (int): Minimum allowed starting week.
        start_max (int): Maximum allowed starting week.
        stop_min (int): Minimum allowed stopping week.
        stop_max (int): Maximum allowed stopping week.

    Returns:
        pd.DataFrame: Filtered dataframe with only series spanning the specified week range.
    """
    data_df['week'] = data_df['graderingsdatum'].dt.isocalendar().week

    first_last_week = data_df.groupby(['Series_id']).agg({'week': ['first', 'last']})
    first_last_week.columns = ['week_first', 'week_last']
    first_last_week = first_last_week.reset_index()

    data_df = data_df.merge(first_last_week, on=['Series_id'])
    week_outlier_mask = (data_df['week_first'] >= start_min) & (data_df['week_first'] <= start_max) & (data_df['week_last'] >= stop_min) & (data_df['week_last'] <= stop_max)

    # print(f'keeping {np.sum(week_outlier_mask)} rows, {np.sum(week_outlier_mask) / week_outlier_mask.shape[0]}%')
    
    data_df = data_df[week_outlier_mask].drop(['week_last','week_first'], axis=1)
    return data_df

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

def get_surrounding_varde(data_gdf, n_closest=3):
    utm_crs = data_gdf.estimate_utm_crs()
    data_gdf = data_gdf.to_crs(utm_crs)

    result = []
    for (Series_id, graderingsdatum), group in data_gdf.groupby(['Series_id', 'graderingsdatum']):
        last_start = graderingsdatum - pd.DateOffset(weeks=1)
        allowed_mask = (data_gdf['graderingsdatum'] >= last_start) & (data_gdf['graderingsdatum'] <= graderingsdatum) & (data_gdf['Series_id'] != Series_id) # 12 sec
        allowed = data_gdf[allowed_mask]

        allowed['distance'] = allowed['geometry'].distance(group['geometry'].iloc[0]).rename('distance')
        group[f'varde_nearby_mean'] = allowed.sort_values(by=['distance']).head(n_closest)['varde'].mean()
        group[f'varde_nearby_min'] = allowed.sort_values(by=['distance']).head(n_closest)['varde'].min()
        group[f'varde_nearby_max'] = allowed.sort_values(by=['distance']).head(n_closest)['varde'].max()
        result.append(group)
    data_gdf = pd.concat(result)
    return data_gdf

def add_sensitivity(data_gdf, fill_na=None):
    winterwheat_sort_senitivities = {
        "Variety": [
            "Bright", "Brons", "Ceylon", "Etana", "Fenomen", "Festival", "Hallfreda", "Hereford", 
            "Informer", "Jonas", "Julius", "Kalmar", "Kask", "KWS Ahoi", "KWS Kerrin", "Lini", 
            "Linus", "Lykke", "Majken", "Marly", "Norin", "Praktik", "Prinz", "Pondus", "RGT Reform", 
            "RGT Stokes", "SU Joran", "Terence", "Valhal"
        ],
        "Brunrost": [
            4, 6, 4, 5, 4, 7, 6, 7, 4, 5, 5, 4, 4, 5, 4, 5, 4, 4, 7, 3, 4, 4, 4, 5, 4, 5, 4, 4, 4
        ],
        "Gulrost": [
            2, 3, 3, 3, 2, 3, 2, 4, 1, 5, 7, 8, 4, 6, 6, 2, 6, 4, 3, 3, 3, 3, 2, 1, 6, 4, 3, 2, 3
        ],
        "Mjöldagg": [
            3, 4, 4, 5, 3, 3, 3, 4, 2, 4, 5, 5, 3, 4, 3, 4, 4, 3, 4, 4, 4, 4, 3, 4, 5, 4, 4, 3, 4
        ],
        "Svartpricksjuka": [
            4, 5, 6, 7, 3, 4, 6, 6, 3, 5, 4, 4, 4, 5, 5, 5, 4, 5, 5, 6, 7, 7, 4, 2, 5, 4, 5, 5, 4
        ],
        "Vetets bladfläcksjuka": [
            5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5,np.nan,5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
        ]
    }

    df = pd.DataFrame(winterwheat_sort_senitivities)
    variety_df_melted = pd.melt(df, id_vars=["Variety"], var_name="skadegorare", value_name="sensitivity")
    merged_data = pd.merge(data_gdf, variety_df_melted, left_on=['sort', 'skadegorare'], right_on=['Variety', 'skadegorare'], how='left')
    merged_data = merged_data.drop(columns='Variety')
    
    # merged_data['sensitivity'] = merged_data['sensitivity'].fillna(merged_data.groupby('skadegorare')['sensitivity'].transform(fill_mode))
    if fill_na is not None:
        filled_dfs = []
        for skadegorare, group in merged_data.groupby('skadegorare'):
            if skadegorare not in df.columns:
                filled_dfs.append(group)
                continue
            group = group.fillna(df[skadegorare].agg(fill_na))
            filled_dfs.append(group)
        merged_data = pd.concat(filled_dfs).reset_index(drop=True)
    
    return merged_data