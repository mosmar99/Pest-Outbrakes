import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import data_processing.smhi_api as smhi_api
import data_processing.smhi_processing as smhi_processing
import geopandas as gpd
import visualize as viz
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


def sarimax_prediction(df):
    """
    Tar en preprocessorad DataFrame (df) som innehåller:
      - 'graderingsdatum': datetime-kolumn
      - 'varde': målvariabel (skadedjursnivå)
      - Andra variabler, t.ex. 'utvecklingsstadium'
    Funktionen konverterar datum, sätter index, skapar lag-funktioner, delar upp datan i train och test,
    och passar en SARIMAX-modell med exogena regressorer.
    """
    import numpy as np
    
    # 1. Se till att 'graderingsdatum' är datetime och sortera
    df["graderingsdatum"] = pd.to_datetime(df["graderingsdatum"], errors='coerce')
    df = df.sort_values("graderingsdatum")
    
    # 2. Sätt datumkolumnen som index
    df_ts = df.set_index("graderingsdatum")
    
    # 3. Skapa lag-funktioner för 'varde'
    #df_ts["lag1_varde"] = df_ts["varde"].shift(1)
    #df_ts["lag2_varde"] = df_ts["varde"].shift(2)
    #df_ts = df_ts.dropna(subset=["lag1_varde", "lag2_varde"])
    
    # 4. Dela upp i train och test baserat på datum
    train_df = df_ts.loc[:'2017-12-31'].copy()
    test_df  = df_ts.loc['2018-01-01':].copy()
    
    y_train = train_df["varde"]
    y_test = test_df["varde"]
    
    # 5. Definiera exogena regressorer (t.ex. 'utvecklingsstadium' och lag-funktionerna)


    exog_vars = ["utvecklingsstadium", "Lufttemperatur",  "Nederbördsmängd"]
    X_train_exog = train_df[exog_vars]
    X_test_exog = test_df[exog_vars]
    
    # Kontrollera att inga NaN finns kvar
    if X_train_exog.isnull().any().any() or X_test_exog.isnull().any().any():
        raise ValueError("Exogena variabler innehåller fortfarande NaN.")
    
    # 6. Passa SARIMAX-modellen (exempelparametrar, justera efter din data)
    model = sm.tsa.statespace.SARIMAX(
        y_train,
        exog=X_train_exog,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # 7. Förutsäg på testperioden med heltalsindex istället för datum
    start = len(train_df)
    end = start + len(test_df) - 1
    pred_res = results.get_prediction(start=start, end=end, exog=X_test_exog)
    y_pred = pred_res.predicted_mean
    # Återassignera testets datumindex
    y_pred.index = test_df.index
    
    # 8. Utvärdera modellen
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return results, y_pred, mse, r2, test_df, y_test

if __name__ == "__main__":
    groda='höstvete'
    skadegorare = 'Svartpricksjuka'
    from_date = '2011-01-07'
    to_date = '2021-01-01'

    data_json = jbv_api.get_gradings(from_date=from_date, to_date=to_date, groda=groda, skadegorare=skadegorare)
    print("---FETCHED JBV-DATA")
    #print(data_json)

    #well, almost all features :^)
    all_features = ['groda', 'latitud', 'longitud', 'plojt', 'skordear', 'sort', 'skadegorare', 'varde', 'graderingsdatum', 'utvecklingsstadium']

    #not using: matmetod, lan, graderingstyp, jordart, sadatum

    #wanted_features = ['groda', 'skadegorare', 'graderingsdatum', 'utvecklingsstadium', 'varde', 'latitud', 'longitud']
    data_df = jbv_process.feature_extraction(data_json, all_features)
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
    #data_gdf = jbv_process.aggregate_data_for_plantations(data_gdf, time_period='W-MON')
    #print('7', data_gdf.shape)

    print(data_df)

    param_id = "2"
    param_id_2 = "5"
    params = [(param_id, 'mean'), (param_id_2, 'sum')]

    data_gdf = smhi_processing.gather_weather_data(
        data_gdf,
        params,
        smhi_api.get_stations_on_parameter_id,
        smhi_api.get_station_data_on_key_param,
        from_date,
        to_date)
    print('8', data_gdf.shape)
    
    most_frequent_plantation_gdf = jbv_process.get_most_frequent_plantation(data_gdf)
    padd_most_frequent_plantation_gdf = jbv_process.introduce_nan_for_large_gaps(most_frequent_plantation_gdf)
    print("null count: \n", padd_most_frequent_plantation_gdf.isnull().sum())
    viz.lineplot(padd_most_frequent_plantation_gdf)
    viz.lineplot_w_weather(padd_most_frequent_plantation_gdf)
    
    #padd_most_frequent_plantation_gdf = padd_most_frequent_plantation_gdf.dropna(subset=['varde', 'utvecklingsstadium', 'Lufttemperatur', 'Nederbördsmängd'])

    print(padd_most_frequent_plantation_gdf)

    print(padd_most_frequent_plantation_gdf['geometry'])

    #dataset to predict on needs some preprocessing, will drop columns with same values as they make no difference
    df = padd_most_frequent_plantation_gdf.drop(columns=['geometry', 'groda', 'skadegorare'])

    #print("null count: \n", padd_most_frequent_plantation_gdf.isnull().sum())
    #print(padd_most_frequent_plantation_gdf.dtypes)
    df = df.dropna(subset=['varde', 'utvecklingsstadium']) #there are a few missing values for these features, drop those rows, found by out-commented code above

    categorical_columns = ['plojt', 'sort']

    onehotencoder = OneHotEncoder(sparse_output=False)
    categorical_encoded = onehotencoder.fit_transform(df[categorical_columns])

    encoded_df = pd.DataFrame(
        categorical_encoded,
        columns=onehotencoder.get_feature_names_out(categorical_columns),
        index=df.index  # ensure same index to facilitate a clean join
    )

    #concat one-hot-encoded and original df
    df = pd.concat([df, encoded_df], axis=1)

    #drop the old ones
    df.drop(columns=categorical_columns, inplace=True)
    #pd.set_option("display.max_columns", None)
    #print(df.head(20))

    df = df.sort_values("graderingsdatum")

    # Create lag features for "varde" (for example, lag of 1 period and 2 periods)
    df["lag1_varde"] = df["varde"].shift(1)
    df["lag2_varde"] = df["varde"].shift(2)

    # Depending on your modeling strategy, you might drop rows with missing lag values:
    df = df.dropna(subset=["lag1_varde", "lag2_varde"])


    #convert datetime64[ns] to year, month, day etc
    df["year"] = df["graderingsdatum"].dt.year
    df["month"] = df["graderingsdatum"].dt.month
    df["day"] = df["graderingsdatum"].dt.day


    #as we have time series, we dont create a random train/test split
    train_df = df[df["graderingsdatum"] < "2018-01-01"]
    test_df = df[df["graderingsdatum"] >= "2018-01-01"]
    
    test_df_for_plot = test_df.copy()

    #now remove it as we have converted it into numerical values
    train_df.drop(columns=['graderingsdatum'], inplace=True)
    test_df.drop(columns=['graderingsdatum'], inplace=True)
    
    features = [col for col in train_df.columns if col != "varde"]

    feat_train = train_df[features]
    label_train = train_df["varde"]

    feat_test = test_df[features]
    label_test = test_df["varde"]

    #grid search
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
    }


    #rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_grid, 
                           scoring='neg_mean_squared_error', 
                           cv=3, 
                           n_jobs=-1, 
                           verbose=2)

    grid_search.fit(feat_train, label_train)

    #rf.fit(feat_train, label_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    best_rf = grid_search.best_estimator_
    
    val_pred = best_rf.predict(feat_test)

    test_df_for_plot["predicted_varde"] = val_pred

    mse = mean_squared_error(label_test, val_pred)
    r2 = r2_score(label_test, val_pred)

    plt.plot(
    test_df_for_plot["graderingsdatum"],
    test_df_for_plot["varde"],
    label='Actual varde'
    )
    plt.plot(
        test_df_for_plot["graderingsdatum"],
        test_df_for_plot["predicted_varde"],
        label='Predicted varde'
    )
    
    plt.title(f"Predictions of varde with a MSE: {mse:.2f} and R²: {r2:.2f}")
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.legend()

    plt.show()

    print("MSE:", mse)
    print("R^2:", r2)


