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

def aggregate_smhi_data(smhi_df:pd.DataFrame, rule='W-MON'):
    numeric = smhi_df.select_dtypes(include="float64").columns
    aggregation = {**{column: 'mean' for column in numeric},
                   'station_key': 'first'}
    smhi_df = smhi_df.resample(rule, on='DateTime (UTC)').agg(aggregation)
    return smhi_df

def process_smhi_data(smhi_df, from_date, to_date):
    """
    All in one Station data pre-processing, 

    Args:
        smhi_df (DataFrame): Smhi Station data df.
        from_date (String): Fromdate, for filtering in data.
        to_date (String): Fromdate, , for filtering in data.
        
    Returns:
        DataFrame: DataFrame with SMHI weather data with standard cleaning and in specified timeframe.
    """
    smhi_df = clean_station_data(smhi_df)
    smhi_df = filter_station_data_on_date(smhi_df, from_date, to_date)
    return smhi_df

def find_closest_stations(jbv_gdf, smhi_stations_gdf, param_id):
    # Convert to estimated UTM CRS to achieve accuracy in distances
    utm_crs = jbv_gdf.estimate_utm_crs()
    jbv_gdf = jbv_gdf.to_crs(utm_crs)
    smhi_stations_gdf = smhi_stations_gdf.to_crs(utm_crs)
    # Keep station point for plotting
    smhi_stations_gdf["station_location"] = smhi_stations_gdf["geometry"]

    # Perform Spatial join by nearest neighbour
    nearest_gdf = jbv_gdf.sjoin_nearest(smhi_stations_gdf, how="left")[["key", "station_location"]]
    jbv_gdf["station_key"] = nearest_gdf["key"]
    jbv_gdf["station_location"] = nearest_gdf["station_location"].to_crs("EPSG:4326")
    
    return jbv_gdf.to_crs("EPSG:4326"), pd.unique(nearest_gdf["key"])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    import smhi_api
    # """
    # Example usage of module
    # """
    from_date = '2020-01-01'
    to_date = '2021-01-01'
    param_id = "19"

    smhi_stations_gdf = smhi_api.get_stations_on_parameter_id(param_id)
    smhi_stations_gdf = process_stations(smhi_stations_gdf, from_date, to_date)

    print(smhi_stations_gdf)

    smhi_df = smhi_api.get_station_data_on_key_param("188790", param_id)
    smhi_df = process_smhi_data(smhi_df, from_date, to_date)
    smhi_df = aggregate_smhi_data(smhi_df)
    print(smhi_df)

    # Kan man ju göra, ???, det kan ochså läggas in 
    # går ha separata funktioner med, kan bara kalla den första i den andra

    # smhi_gdf = get_station_data_on_key_param("188790", "21", from_date, to_date)
    # # smhi_gdf = get_station_data_on_key_param("154860", "19", from_date, to_date)

    # print(smhi_gdf)