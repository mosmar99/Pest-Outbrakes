import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import data_processing.smhi_api as smhi_api
import data_processing.smhi_processing as smhi_processing
import features as F
import geopandas as gpd
import visualize as viz
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from tools.train_net import do_train

if __name__ == "__main__":
    groda='höstvete'
    skadegorare = 'Svartpricksjuka'
    from_date = '2020-01-01'
    to_date = '2024-01-01'

    data_json = jbv_api.get_gradings(from_date=from_date, to_date=to_date, groda=groda, skadegorare=skadegorare)
    print("---FETCHED JBV-DATA")

    wanted_features = ['groda', 'sort', 'skadegorare', 'graderingsdatum', 'matmetod', 'utvecklingsstadium', 'varde', 'latitud', 'longitud', 'sadatum', 'utplaceringsdatumFalla']
    data_df = jbv_process.feature_extraction(data_json, wanted_features)
    #data_df['varde'] = data_df['varde'] * data_df['utvecklingsstadium']
    print('1', data_df.shape)
    data_df = jbv_process.drop_rows_with_missing_values(data_df)
    print('2', data_df.shape)
    data_df = jbv_process.drop_duplicates(data_df)
    print('3', data_df.shape)
    data_df = jbv_process.clean_coordinate_format(data_df)
    print('4', data_df.shape)
    data_gdf = jbv_process.sweref99tm_to_wgs84(data_df)
    print('5', data_gdf.shape)
    data_gdf = jbv_process.remove_outside_sweden_coordinates(data_gdf)
    print('6', data_gdf.shape)
    data_gdf = jbv_process.aggregate_data_for_plantations(data_gdf)
    print('7', data_gdf.shape)
    data_gdf = jbv_process.add_sensitivity(data_gdf)
    print('8', data_gdf.shape)
    print(data_gdf)

    param_id = "2"
    param_id_2 = "5"
    params = [(param_id, 'mean'), (param_id_2, 'sum')]

    data_gdf = smhi_processing.gather_weather_data(
        data_gdf,
        params,
        smhi_api.get_stations_on_parameter_id,
        smhi_api.get_station_data_on_key_param,
        from_date,
        to_date,
        weekly=True)
    print('9', data_gdf.shape)

    #most_frequent_plantation_gdf = jbv_process.get_most_frequent_plantation(data_gdf)
    #padd_most_frequent_plantation_gdf = jbv_process.introduce_nan_for_large_gaps(data_gdf)
    #viz.lineplot(padd_most_frequent_plantation_gdf)
    #viz.lineplot_w_weather(padd_most_frequent_plantation_gdf)
    
    # feature engineering below
    #new_df = F.days_since_sowing(data_gdf)
    new_df = F.days_since_last_measurement(data_gdf)
    #new_df = F.days_since_last_rain(new_df)
    #enw_df = F.days_since_trap_placement(new_df)
    new_df = F.utvecklingsstadium_progression(new_df)
    new_df = F.varde_progression(new_df)
    df = F.add_month_week_day(new_df)
    print(df.shape)
    print(df)
    print(df.columns.values.tolist())
    df.to_csv("enh_data.csv")
    df = pd.read_csv("enh_data.csv")
    
    # data cleaning and preparation
    print("Missing values per column:")
    print(df.isnull().sum())

    # data cleaning
    #df = df.drop(columns=['Unnamed: 0','sadatum', 'varde_progression', 'utplaceringsdatumFalla']) use when reading csv dataframe
    df = df.drop(columns=['varde_progression'])
    print("Missing values per column:")
    print(df.isnull().sum())

    #df[["lon", "lat"]] = df["geometry"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)")

    #df["lon"] = df["lon"].astype(float)
    #df["lat"] = df["lat"].astype(float)
    #df["lat_lon_product"] = df['lat'] * df['lon']

    target_col = 'varde'

    exclude = {target_col, 'graderingsdatum'}
    candidate_cols = [col for col in df.columns if col not in exclude]

    df['varde'] = df.groupby('geometry')['varde'].transform(lambda x: x.ffill().bfill())

    num_cols = df[candidate_cols].select_dtypes(include=['number']).columns.tolist()
    cat_cols = df[candidate_cols].select_dtypes(include=['object']).columns.tolist()

    cat_cols = [col for col in cat_cols if col != 'geometry']

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    print("Missing values per column:")
    print(df.isnull().sum())

    df = df.dropna(subset=cat_cols)
    print("Missing values per column:")
    print(df.isnull().sum())

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
        print("One-hot encoded columns:", cat_cols)

    df['graderingsdatum'] = pd.to_datetime(df['graderingsdatum'])
    df = df.sort_values(['geometry', 'graderingsdatum'])

    df['varde_lag1'] = df.groupby('geometry')['varde'].shift(1)
    df['varde_lag2'] = df.groupby('geometry')['varde'].shift(2)
    df['varde_rollmean3'] = df.groupby('geometry')['varde'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    df['varde_lag1'] = df.groupby('geometry')['varde_lag1'].transform(lambda x: x.fillna(x.mean()))
    df['varde_lag2'] = df.groupby('geometry')['varde_lag2'].transform(lambda x: x.fillna(x.mean()))
    df['varde_rollmean3'] = df.groupby('geometry')['varde_rollmean3'].transform(lambda x: x.fillna(x.mean()))

    print(df.head())

    # do training
    do_train(df)