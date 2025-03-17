import pandas as pd
import numpy as np
import geopandas as gpd
import os

from sklearn.preprocessing import MinMaxScaler

class datamodule:
    VARKORN = 'Varkorn'
    HOSTVETE = 'Hostvete'
    RAGVETE = 'Ragvete'

    DISEASES = {HOSTVETE: ['Bladfläcksvampar', 'Brunrost', 'Svartpricksjuka','Gulrost', 'Mjöldagg', 'Vetets bladfläcksjuka', 'Gräsbladlus', 'Sädesbladlus', 'Havrebladlus',  'Nederbörd'],
                VARKORN: ['Sköldfläcksjuka', 'Kornets bladfläcksjuka', 'Mjöldagg', 'Havrebladlus', 'Sädesbladlus', 'Kornrost', 'Gräsbladlus'],
                RAGVETE: ['Brunrost', 'Gulrost', 'Sköldfläcksjuka', 'Mjöldagg', 'Bladfläcksvampar']}
    
    def __init__(self, crop):
        path = os.path.join('datasets', f"{crop}.pkl")
        data_df = pd.read_pickle(path)
        self.data_gdf = gpd.GeoDataFrame(data_df)
        self.data_gdf['utvecklingsstadium'] = self.data_gdf['utvecklingsstadium'].astype(np.float64)
        self.data_gdf['year'] = self.data_gdf['graderingsdatum'].dt.year
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.dependent = datamodule.DISEASES[crop]
        print('Possible targets', self.dependent)
        self.target = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def set_target(self, target):
        self.target = target
        self.data_gdf['target'] = self.data_gdf[self.target] 

    def add_cumulative(self):
        agg_year = {'Nederbördsmängd_sum': 'sum',
                'Daggpunktstemperatur_mean': 'mean',
                'Relativ Luftfuktighet_mean': 'mean',
                'Lufttemperatur_max': 'mean',
                'Solskenstid_sum': 'sum'}
        
        self.data_gdf = self.data_gdf.sort_values(by=['Series_id', 'graderingsdatum'])
        cumulative = self.data_gdf.groupby(['year', 'Series_id']).rolling(window=52, on='graderingsdatum', min_periods=1).agg(agg_year).reset_index()
        col_names = {key: f'{key}_year_cumulative'for key in agg_year.keys()}
        cumulative = cumulative.rename(columns=col_names)
        self.data_gdf[list(col_names.values())] = cumulative[list(col_names.values())]

    def drop_standard_weather(self):
        self.data_gdf = self.data_gdf.drop(['Lufttemperatur_min', 'Lufttemperatur_max',
            'Nederbördsmängd_sum', 'Nederbördsmängd_max', 'Solskenstid_sum',
            'Daggpunktstemperatur_mean', 'Daggpunktstemperatur_max',
            'Relativ Luftfuktighet_mean', 'Långvågs-Irradians_mean'], axis=1)

    def lag_dependent(self, num_weeks=2, additional=[]):
        lagged_dependent = {f'{days}w_{col}': self.data_gdf.groupby('Series_id')[col].shift(days) for days in range(1,num_weeks) for col in self.dependent + additional}

        lagged_dependent_df = pd.DataFrame(lagged_dependent)
        self.data_gdf = pd.concat([self.data_gdf, lagged_dependent_df], axis=1)

    def onehot_encoding(self, columns):
        self.data_gdf = pd.get_dummies(self.data_gdf, columns=columns)

    def drop_objects(self):
        objects = self.data_gdf.select_dtypes(include=['object']).columns
        self.data_gdf = self.data_gdf.drop(objects, axis=1)

    def drop_na(self):
        self.data_gdf = self.data_gdf.dropna()

    def X_y_split(self, additional_to_drop=[]):
        numeric = self.data_gdf.select_dtypes(include=['float64', 'int64', 'UInt32', 'bool']).columns
        self.X = self.data_gdf[numeric].drop(['target', 'utvecklingsstadium'] + self.dependent + additional_to_drop, axis=1)
        self.y = self.data_gdf[['target']]

    def normalize_X_y(self):
        self.X = pd.DataFrame(self.scaler_X.fit_transform(self.X), columns=self.X.columns, index=self.X.index)
        self.y = pd.DataFrame(self.scaler_y.fit_transform(self.y), index=self.y.index)

    def test_train_split(self, test_size=0.2):
        series = self.data_gdf['Series_id'].unique()
        sampled_series = np.random.choice(series, size=int(test_size * len(series)), replace=False)
        test_mask = self.data_gdf['Series_id'].isin(sampled_series)

        train_mask = ~(test_mask)
        # print('Training on:',sum(train_mask)/len(train_mask))
        # print('testing on:', sum(test_mask)/len(test_mask))

        self.X_train, self.X_test = self.X[train_mask], self.X[test_mask]
        self.y_train, self.y_test = self.y[train_mask], self.y[test_mask]
    
    def CV_test_train_split(self, n_folds=10):
        splits = []
        series_left = self.data_gdf['Series_id'].unique()
        test_size = int((1/n_folds) * len(series_left))
        for fold in range(n_folds):
            if fold == n_folds:
                test_mask = self.data_gdf['Series_id'].isin(series_left)
            else:
                choice = np.random.choice(series_left, size=test_size, replace=False)
                series_left = np.setdiff1d(series_left, choice)
                test_mask = self.data_gdf['Series_id'].isin(choice)

            train_mask = ~(test_mask)
            X_train, X_test = self.X[train_mask], self.X[test_mask]
            y_train, y_test = self.y[train_mask], self.y[test_mask]

            splits.append((X_train, X_test, y_train, y_test))
        return splits
    
    def test_train_split_year(self, test_year=2022):
        test_mask = self.data_gdf['graderingsdatum'].dt.year == test_year

        train_mask = ~(test_mask)
        print('Training on:',sum(train_mask)/len(train_mask))
        print('testing on:', sum(test_mask)/len(test_mask))

        self.X_train, self.X_test = self.X[train_mask], self.X[test_mask]
        self.y_train, self.y_test = self.y[train_mask], self.y[test_mask]
    
    def get_test_train(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def inverse_scale(self, y):
        return pd.DataFrame(self.scaler_y.inverse_transform(y), index=y.index)
    
    def default_process(self, target):
        self.set_target(target)
        self.add_cumulative()
        self.drop_standard_weather()
        self.lag_dependent(num_weeks=2)
        # self.onehot_encoding(self, columns)
        self.drop_objects()
        self.drop_na()
        self.X_y_split()
        self.normalize_X_y()
        self.test_train_split(test_size=0.2)
        # self.test_train_split_year(self, test_year=2022)

        # Get test train for training, 
        # self.get_test_train(self)
        # self.inverse_scale(self, y_pred)
        # self.inverse_scale(self, y_test)
