import pandas as pd
import geopandas as gpd

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
    return smhi_df

def aggregate_smhi_data(smhi_df, how='mean', rule='W-MON'):
    """
    Data aggregation on datetime for smhi station data dataframe

    Args:
        smhi_df (DataFrame): Smhi Station data df.
        rule (String): Aggregation rule, uses pandas aggregation rules
        
    Returns:
        DataFrame: DataFrame with SMHI weather data aggregated on datetime.
    """
    numeric = smhi_df.select_dtypes(include=['float64', 'int64']).columns
    aggregation = {**{column: how for column in numeric},
                   'station_key': 'first'}
    smhi_df = smhi_df.resample(rule, on='DateTime (UTC)').agg(aggregation)
    return smhi_df

def process_smhi_data(smhi_df, from_date, to_date, aggregaton_how=['mean'], aggregation_rule='W-MON'):
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
    smhi_df = aggregate_smhi_data(smhi_df, aggregaton_how, aggregation_rule)
    return smhi_df

def find_closest_stations(gdf, smhi_stations_gdf):
    """
    Finds closest stations for every row.

    Args:
        gdf (GeoDataFrame): GeoDataFrame for wich we want to find the nearest stations.
        smhi_stations_gdf (GeoDataFrame): Data on stations with geometry.
        
    Returns:
        Tuple(GeoDataFrame, List): original gdf with SMHI station key appended. List of unique nearest station keys
    """
    # Convert to estimated UTM CRS to achieve accuracy in distances
    utm_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm_crs)
    smhi_stations_gdf = smhi_stations_gdf.to_crs(utm_crs)

    # Perform Spatial join by nearest neighbour
    nearest_gdf = gdf.sjoin_nearest(smhi_stations_gdf, how="left")
    nearest_gdf = nearest_gdf[~nearest_gdf.index.duplicated(keep='first')]

    gdf["station_key"] = nearest_gdf["key"]
    return gdf.to_crs("EPSG:4326"), pd.unique(nearest_gdf["key"])

def gather_weather_data(gdf, params, f_get_stations, f_get_station_data, from_date, to_date, aggregation_rule='W-MON'):
    """
    Standard function for gathering closest weather data into a gdf

    Args:
        gdf (GeoDataFrame): GeoDataFrame for wich we want to find the nearest stations.
        params (List(Tuple(String, String))): List of tuple with parameter id to get, and aggregation type ex. 'mean', 'sum', 'max', 'min'
        f_get_stations (GeoDataFrame): Function for getting data on stations.
        f_get_station_data (GeoDataFrame): Function for getting weatherdata from stations.
        from_date (String): from date, for filtering in data.
        to_date (String): to date, for filtering in data.
        aggregation_rule (String): Aggregation rule, uses pandas aggregation rules
        
    Returns:
        Tuple(GeoDataFrame, List): original gdf with SMHI station key appended. List of unique nearest station keys
    """
    for param_id, aggregaton_how in params:
        stations_gdf = f_get_stations(param_id)
        stations_gdf = process_stations(stations_gdf, from_date, to_date)

        gdf, stations_to_get = find_closest_stations(gdf, stations_gdf)

        parameter_data = pd.DataFrame()
        for station_key in stations_to_get:
            station_data_df = f_get_station_data(station_key, param_id)
            station_data_df = process_smhi_data(station_data_df, from_date, to_date, aggregaton_how, aggregation_rule)
            parameter_data = pd.concat([parameter_data, station_data_df])

        gdf = gdf.merge(parameter_data,
                            how='left',
                            left_on=['graderingsdatum', 'station_key'],
                            right_on=['DateTime (UTC)', 'station_key'])
        gdf = gdf.drop('station_key', axis=1)

    return gdf

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString, Point
    import smhi_api
    # """
    # Example usage of module
    # """
    from_date = '2020-01-01'
    to_date = '2021-01-01'
    param_id = "2"
    param_id_2 = "5"
    param_id_3 = "10"
    params = [(param_id, 'mean'), (param_id_2, 'sum'), (param_id_3, 'sum')]

    # smhi_stations_gdf = smhi_api.get_stations_on_parameter_id(param_id)
    # smhi_stations_gdf = process_stations(smhi_stations_gdf, from_date, to_date)

    # print(smhi_stations_gdf)

    # smhi_df = smhi_api.get_station_data_on_key_param("188790", param_id)
    # smhi_df = process_smhi_data(smhi_df, from_date, to_date)
    # smhi_df = aggregate_smhi_data(smhi_df)
    # print(smhi_df)

    # Define three example points in Sweden
    data = {
        "Name": ["Stockholm", "Stockholm", "Stockholm",
                "Gothenburg", "Gothenburg", "Gothenburg",
                "Malmö", "Malmö", "Malmö"],
        "geometry": [
            Point(18.0686, 59.3293), Point(18.0686, 59.3293), Point(18.0686, 59.3293),
            Point(11.9746, 57.7089), Point(11.9746, 57.7089), Point(11.9746, 57.7089),
            Point(13.0038, 55.6049), Point(13.0038, 55.6049), Point(13.0038, 55.6049)
        ],
        "graderingsdatum": [
            pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-20"),
            pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-20"),
            pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-20")
        ]
    }

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    gdf = gather_weather_data(  gdf,
                                params,
                                smhi_api.get_stations_on_parameter_id,
                                smhi_api.get_station_data_on_key_param,
                                from_date,
                                to_date)

    print(gdf)
    # smhi_gdf = get_station_data_on_key_param("188790", "21", from_date, to_date)
    # # smhi_gdf = get_station_data_on_key_param("154860", "19", from_date, to_date)

    # print(smhi_gdf)