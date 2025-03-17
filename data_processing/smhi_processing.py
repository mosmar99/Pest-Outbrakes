import numpy as np
import pandas as pd
import geopandas as gpd
import pickle

def clean_stations(smhi_stations_gdf, dropcols=['latitude','longitude', 'summary', 'link', 'id', 'title', 'ownerCategory', 'owner', 'active', 'updated', 'height']):
    """
    Clean list of stations, drop irrelevant columns and convert dtypes.

    Args:
        smhi_gdf (GeoDataFrame): Smhi stations gdf.
        dropcols List(String): Columns to drop from data, default=['latitude','longitude', 'summary', 'link', 'id', 'title', 'ownerCategory', 'owner', 'active', 'updated', 'height']
        
    Returns:
        GeoDataFrame: GeoDataFrame with columns dropped and dtypes set.
    """
    smhi_stations_gdf = smhi_stations_gdf.drop(dropcols, axis=1)

    # smhi_stations_gdf['updated'] = pd.to_datetime(smhi_stations_gdf['updated'], unit="ms")
    smhi_stations_gdf['from'] = pd.to_datetime(smhi_stations_gdf['from'], unit="ms")
    smhi_stations_gdf['to'] = pd.to_datetime(smhi_stations_gdf['to'], unit="ms")

    return smhi_stations_gdf

def filter_stations_on_date(smhi_stations_gdf, from_date, to_date):
    """
    Get filter on stations that have data within timeframe.

    Args:
        gdf (GeoDataFrame): Smhi Stations gdf.
        from_date (String): Fromdate, for checking avalibility in station.
        to_date (String): Fromdate, for checking avalibility in station.
        
    Returns:
        GeoDataFrame: GeoDataFrame with SMHI Stations for timeframe.
    """
    from_mask = (smhi_stations_gdf['from'] < pd.to_datetime(from_date)) & (smhi_stations_gdf['to'] > pd.to_datetime(to_date))
    smhi_stations_gdf = smhi_stations_gdf[from_mask]
    
    return smhi_stations_gdf

def filter_stations_type(smhi_stations_gdf, measuringStations):
    """
    Filter stations on type ['CORE', 'ADDITIONAL']

    Args:
        gdf (GeoDataFrame): Smhi Stations gdf.
        from_date (String): Fromdate, for checking avalibility in station.
        to_date (String): Fromdate, for checking avalibility in station.
        
    Returns:
        GeoDataFrame: GeoDataFrame with SMHI Stations for timeframe and parameter id.
    """
    measuringStations_mask = smhi_stations_gdf["measuringStations"] == measuringStations
    smhi_stations_gdf = smhi_stations_gdf[measuringStations_mask]
    return smhi_stations_gdf

def process_stations(smhi_stations_gdf, from_date, to_date):
    """
    All in one Stations pre-processing, 

    Args:
        smhi_stations_gdf (GeoDataFrame): Smhi Stations gdf.
        from_date (String): Fromdate, for checking avalibility in station.
        to_date (String): Fromdate, for checking avalibility in station.
        
    Returns:
        GeoDataFrame: GeoDataFrame with SMHI Stations for timeframe and parameter id.
    """
    smhi_stations_gdf = clean_stations(smhi_stations_gdf)
    smhi_stations_gdf = filter_stations_type(smhi_stations_gdf, 'CORE')
    smhi_stations_gdf = filter_stations_on_date(smhi_stations_gdf, from_date, to_date)
    return smhi_stations_gdf

def clean_station_data(smhi_df):
    """
    Reformat SMHI Data in unified format and drop irrelevant rows.

    Args:
        smhi_df (GeoDataFrame): GeoDataFrame of SMHI data.

    Returns:
        DataFrame: SMHI Data in unified format without irrelevant rows.
    """
    # Dates comes in either ["Tid (UTC)", "Datum"] or ["Från Datum Tid (UTC)", "Till Datum Tid (UTC)"] format based on parameter,
    # consolidate both in same format with "DateTime (UTC)"
    if "Tid (UTC)" in smhi_df.columns:
        smhi_df["DateTime (UTC)"] = pd.to_datetime(smhi_df["Datum"] + " " + smhi_df["Tid (UTC)"])
    else:
        smhi_df["DateTime (UTC)"] = pd.to_datetime(smhi_df["Från Datum Tid (UTC)"])

    # Drop irrelevant columns if they exist
    drop_cols = ["Representativt dygn", "Tid (UTC)", "Datum", "Från Datum Tid (UTC)", "Till Datum Tid (UTC)", "Unnamed: 4", "Unnamed: 5", "Tidsutsnitt:"]
    for col in drop_cols:
        if col in smhi_df.columns:
            smhi_df = smhi_df.drop(col, axis=1)
    
    return smhi_df

def filter_station_data_on_date(smhi_df, from_date, to_date):
    """
    Filter on timeframe

    Args:
        smhi_df (DataFrame): DataFrame of SMHI data.
        from_date (string): Date to filter from
        to_date (string): Date to filter to

    Returns:
        DataFrame: SMHI Data within timeframe
    """
    from_mask = (smhi_df["DateTime (UTC)"] > pd.to_datetime(from_date)) & (smhi_df["DateTime (UTC)"] < pd.to_datetime(to_date))
    smhi_df = smhi_df[from_mask]

    numeric = smhi_df.select_dtypes(include=['float64', 'int64']).columns
    

    return smhi_df

def aggregate_smhi_data(smhi_df, how='mean'):
    """
    Data aggregation on datetime for smhi station data dataframe

    Args:
        smhi_df (DataFrame): Smhi Station data df.
        time_period (String): Aggregation rule, uses pandas aggregation rules
        
    Returns:
        DataFrame: DataFrame with SMHI weather data aggregated on datetime.
    """
    numeric = smhi_df.select_dtypes(include=['float64', 'int64']).columns
    aggregation = {**{column: how for column in numeric},
                   'station_key': 'first'}
    
    smhi_df['time_period'] = smhi_df['DateTime (UTC)'].dt.to_period('D')
    smhi_df = smhi_df.groupby('time_period').agg(aggregation).reset_index()
    smhi_df['DateTime (UTC)'] = smhi_df['time_period'].dt.start_time
    smhi_df = smhi_df.drop('time_period', axis=1)

    names = {column: f'{column}_{how}' for column in numeric}
    smhi_df = smhi_df.rename(columns=names)
    return smhi_df

def process_smhi_data(smhi_df, from_date, to_date, aggregaton_how='mean'):
    """
    All in one Station data pre-processing, 

    Args:
        smhi_df (DataFrame): Smhi Station data df.
        from_date (String): from date, for filtering in data.
        to_date (String): to date, for filtering in data.
        
    Returns:
        DataFrame: DataFrame with SMHI weather data with standard cleaning and in specified timeframe.
    """
    smhi_df = clean_station_data(smhi_df)
    smhi_df = filter_station_data_on_date(smhi_df, from_date, to_date)
    smhi_df = aggregate_smhi_data(smhi_df, aggregaton_how)
    return smhi_df

def find_closest_n_stations(gdf, stations_gdf:gpd.GeoDataFrame, max_dist=50000, closest_n=4):
    utm_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm_crs)
    stations_gdf = stations_gdf.to_crs(utm_crs)

    loc_to_stations = {}
    stations = pd.Series([], name='key')

    locations = gdf['geometry'].unique()
    for location in locations:
        # print(location)
        distances = stations_gdf['geometry'].distance(location).rename('distance')

        stations_gdf_dist = pd.concat([stations_gdf, distances], axis=1)
        filtered_stations_gdf = stations_gdf_dist[stations_gdf_dist['distance'] < max_dist]

        filtered_stations_gdf = filtered_stations_gdf.sort_values(by='distance', ascending=True)
        
        loc_to_stations[location] = filtered_stations_gdf[['key', 'distance']].head(closest_n)
        stations = pd.concat([stations, filtered_stations_gdf['key'].head(closest_n)], axis=0)
    
    return loc_to_stations, stations.unique(), utm_crs

def gather_weather_data(gdf, params, f_get_stations, f_get_station_data, from_date, to_date, weekly=False, max_dist=float('inf'), closest_n=4):
    all_parameter_data = {}
    for param_id, aggregaton_how in params:
        stations_gdf = f_get_stations(param_id)
        stations_gdf = process_stations(stations_gdf, from_date, to_date)
        loc_to_stations, unique_stations, utm_crs = find_closest_n_stations(gdf, stations_gdf, max_dist=max_dist, closest_n=closest_n)
        gdf = gdf.to_crs(utm_crs)
        
        # Gather all data for parameter
        all_station_data = pd.date_range(start=pd.to_datetime(from_date), end=pd.to_datetime(to_date), freq='D').to_series().rename('DateTime (UTC)')
        param_name = None
        for key in unique_stations:
            station_data_df = f_get_station_data(key, param_id)
            station_data_df = process_smhi_data(station_data_df, from_date, to_date, aggregaton_how)

            numeric = station_data_df.select_dtypes(include=['float64', 'int64']).columns
            if param_name == None:
                param_name = numeric[0]
            station_data_df = station_data_df.rename(columns={name: key for name in numeric})

            relevant_cols = station_data_df.select_dtypes(include=['float64', 'int64', 'datetime64']).columns
            all_station_data = pd.merge(all_station_data, station_data_df[relevant_cols], how='left', on='DateTime (UTC)')

            all_parameter_data[param_name] = all_station_data

        weather_data = []
        for location, stations in loc_to_stations.items():
            stations_to_use = stations['key'].to_list()
            vals = all_station_data[stations_to_use].mean(axis=1, skipna=True).rename(param_name)
            vals = pd.concat([all_station_data['DateTime (UTC)'], vals], axis=1)
            if weekly:
                vals = vals.rolling(window=7, center=False, on='DateTime (UTC)').agg(aggregaton_how)
            vals['geometry'] = location
            weather_data.append(vals)
        
        weather_data_df = pd.concat(weather_data, axis=0).rename(columns={'DateTime (UTC)': 'graderingsdatum'})
        gdf = pd.merge(gdf, weather_data_df, on=['graderingsdatum', 'geometry'], how='left')
        print('GOT SMHI PARAMETER:', param_name)

    return gdf.to_crs("EPSG:4326")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString, Point
    import smhi_api

    with open('wdata.pickle', 'rb') as handle:
        all_parameter_data:pd.DataFrame = pickle.load(handle)
    
    humidity_df = all_parameter_data['Relativ Luftfuktighet_mean'].set_index('DateTime (UTC)')
    rain_df = all_parameter_data['Nederbördsmängd_sum'].set_index('DateTime (UTC)')

    print(humidity_df.columns)
    print(humidity_df.columns.intersection(rain_df.columns))
    common = humidity_df.columns.intersection(rain_df.columns)

    for station in common:
        pass

