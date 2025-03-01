# general file for feature engineering
# implement functions for feature engineering on the format param: pd.DataFrame, return: altered pd.DataFrame with new feature column/s

import pandas as pd

# general time based features
def days_since_sowing(data_df):
    """
    Adds the number of days since sadatum as a feature.
    
    Args:
        pd.DataFrame: The dataset.

    Returns:
        pd.DataFrame: The new dataset with the additional feature.
    """
    if 'sadatum' not in data_df.columns:
        print("sadatum column not found in data_df.")
        return data_df

    new_data_df = data_df.copy()

    new_data_df['graderingsdatum'] = pd.to_datetime(new_data_df['graderingsdatum'])

    sowing_date = pd.to_datetime(new_data_df['sadatum'])
    new_data_df['days_since_sowing'] = (new_data_df['graderingsdatum'] - sowing_date).dt.days

    return new_data_df

def days_since_trap_placement(data_df):
    """
    Adds the number of days since utplaceringsdatumFalla as a feature.
    
    Args:
        pd.DataFrame: The dataset.

    Returns:
        pd.DataFrame: The new dataset with the additional feature.
    """
    if 'utplaceringsdatumFalla' not in data_df.columns:
        print("utplaceringsdatumFalla column not found in data_df.")
        return data_df

    new_data_df = data_df.copy()

    new_data_df['graderingsdatum'] = pd.to_datetime(new_data_df['graderingsdatum'])

    trap_placement_day = pd.to_datetime(new_data_df['utplaceringsdatumFalla'])
    new_data_df['days_since_trap_placement'] = (new_data_df['graderingsdatum'] - trap_placement_day).dt.days

    return new_data_df

def days_since_last_measurement(data_df):
    """
    Adds the number of days since the last measurement as a feature.
    
    Args:
        pd.DataFrame: The dataset.

    Returns:
        pd.DataFrame: The new dataset with the additional feature.
    """
    if 'graderingsdatum' not in data_df.columns:
        print("graderingsdatum column not found in data_df.")
        return data_df

    new_data_df = data_df.copy()

    new_data_df['graderingsdatum'] = pd.to_datetime(new_data_df['graderingsdatum'])

    new_data_df['days_since_last_graderingsdatum'] = new_data_df.groupby(['geometry', 'groda', 'skadegorare'])['graderingsdatum'].diff().dt.days

    return new_data_df

def utvecklingsstadium_progression(data_df):
    """
    Adds the difference between the previous utvecklingsstadium and this as a feature.
    
    Args:
        pd.DataFrame: The dataset.

    Returns:
        pd.DataFrame: The new dataset with the additional feature.
    """
    if 'utvecklingsstadium' not in data_df.columns:
        print("utvecklingsstadium column not found in data_df.")
        return data_df

    new_data_df = data_df.copy()

    new_data_df['utvecklingsstadium_progression'] = new_data_df.groupby(['geometry', 'groda', 'skadegorare'])['utvecklingsstadium'].diff()

    return new_data_df

def varde_progression(data_df):
    """
    Adds the difference between the previous varde and this as a feature.
    
    Args:
        pd.DataFrame: The dataset.

    Returns:
        pd.DataFrame: The new dataset with the additional feature.
    """
    if 'varde' not in data_df.columns:
        print("varde column not found in data_df.")
        return data_df

    new_data_df = data_df.copy()

    new_data_df['varde_progression'] = new_data_df.groupby(['geometry', 'groda', 'skadegorare'])['varde'].diff()

    return new_data_df

def days_since_last_rain(data_df):
    """
    
    """

def MA_weekly(data_df):
    """
    
    """
    new_data_df = data_df.copy()
    
def add_month_week_day(data_df):
    """
    Add month number, week number, and day number to as features.
    
    Args:
        pd.DataFrame: The dataset.

    Returns:
        pd.DataFrame: The new dataset with the additional features.
    """
    new_data_df = data_df.copy()

    new_data_df['graderingsdatum'] = pd.to_datetime(new_data_df['graderingsdatum'])

    new_data_df['month'] = new_data_df['graderingsdatum'].dt.month

    new_data_df['week_number'] = new_data_df['graderingsdatum'].dt.isocalendar().week.astype(int)

    new_data_df['day_number'] = new_data_df['graderingsdatum'].dt.dayofyear

    return new_data_df

