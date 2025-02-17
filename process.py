import pandas as pd
import numpy as np

def feature_extraction(data_df, crop, pest):
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
    filtered_rows_df = data_df[(data_df['latitud'] == latitud) & (data_df['longitud'] == longitud)]
    return filtered_rows_df

def transform_date(data_df):
    # convert type
    data_df['graderingsdatum'] = pd.to_datetime(data_df['graderingsdatum'])

    # prune nonexisting dates
    threshold = pd.Timedelta(days=14)
    data_df['gap'] = data_df['graderingsdatum'].diff() > threshold
    data_df.loc[data_df['gap'], ['varde', 'utvecklingsstadium']] = np.nan

    return data_df

def filter_on_condition(data_df, feature, condition):
    """
    Returns rows from df where the 'crop' column matches the given crop name.
    """
    return data_df[data_df[feature] == condition]