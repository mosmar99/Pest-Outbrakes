import matplotlib.pyplot as plt
import pandas as pd

def lineplot(data_gdf, moving_avg=None):
    """
    Generates a line plot of pest and crop values over time, with an optional moving average filter.
    Args:
        data_gdf (pd.DataFrame): The input dataframe containing crop and pest data.
        crop (str): The name of the crop being analyzed.
        pest (str): The name of the pest affecting the crop.
        latitud (float): Latitude coordinate of the data location.
        longitud (float): Longitude coordinate of the data location.
        moving_avg (Optional[str]): Specifies the level of moving average ("low", "mid", "high"), or None.
    Returns:
        None
    """

    window_sizes = {
        "low" : max(3,  len(data_gdf) // 50),   
        "mid" : max(5,  len(data_gdf) // 45),   
        "high": max(7, len(data_gdf) // 40)  
    }

    crop = data_gdf['groda'].iloc[0]
    pest = data_gdf['skadegorare'].iloc[0]
    coordinate = data_gdf['geometry'].iloc[0]
    longitude, latitude = coordinate.x, coordinate.y


    plt.figure(figsize=(12, 6))
    if moving_avg == None:        
        plt.plot(data_gdf['graderingsdatum'], data_gdf['varde'], color='#d62728', label=str(pest).capitalize(), lw=1.5, zorder=1)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['utvecklingsstadium'], color='#2ca02c', label=str(crop).capitalize(), lw=2, zorder=2)
    elif moving_avg in window_sizes:
        window = window_sizes[moving_avg]
        data_gdf['varde_smooth'] = data_gdf['varde'].rolling(window=window, min_periods=1).mean()
        data_gdf['utvecklingsstadium_smooth'] = data_gdf['utvecklingsstadium'].rolling(window=window, min_periods=1).mean()

        data_gdf.loc[data_gdf['varde'].isna(), 'varde_smooth'] = None
        data_gdf.loc[data_gdf['utvecklingsstadium'].isna(), 'utvecklingsstadium_smooth'] = None        

        plt.plot(data_gdf['graderingsdatum'], data_gdf['varde_smooth'], color='#d62728', label=f"{str(pest).capitalize()} (Smoothed)", lw=1.5)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['utvecklingsstadium_smooth'], color='#2ca02c', label=f"{str(crop).capitalize()} (Smoothed)", lw=1.5)
    else:
        raise ValueError("Invalid moving_avg parameter. Choose from: None, 'low', 'mid', or 'high'.")
    
    plt.xlabel('DATUM')
    plt.ylabel('VÄRDEN')
    plt.title(f'{str(crop).capitalize()} & {str(pest).capitalize()} | Location. Lat={round(latitude, 5)}, Long: {round(longitude, 5)}')
    
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

def lineplot_w_weather(data_gdf, moving_avg=None):
    """
    Generates a line plot of pest and crop values over time, with an optional moving average filter.
    Args:
        data_gdf (pd.DataFrame): The input dataframe containing crop and pest data.
        crop (str): The name of the crop being analyzed.
        pest (str): The name of the pest affecting the crop.
        latitud (float): Latitude coordinate of the data location.
        longitud (float): Longitude coordinate of the data location.
        moving_avg (Optional[str]): Specifies the level of moving average ("low", "mid", "high"), or None.
    Returns:
        None
    """

    window_sizes = {
        "low" : max(3,  len(data_gdf) // 50),   
        "mid" : max(5,  len(data_gdf) // 45),   
        "high": max(7, len(data_gdf) // 40)  
    }

    crop = data_gdf['groda'].iloc[0]
    pest = data_gdf['skadegorare'].iloc[0]
    temp = data_gdf['Lufttemperatur'].iloc[0]
    coordinate = data_gdf['geometry'].iloc[0]
    longitude, latitude = coordinate.x, coordinate.y


    plt.figure(figsize=(12, 6))
    if moving_avg == None:        
        plt.plot(data_gdf['graderingsdatum'], data_gdf['varde'], color='#d62728', label=str(pest).capitalize(), lw=1.5, zorder=1)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['utvecklingsstadium'], color='#2ca02c', label=str(crop).capitalize(), lw=2, zorder=2)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['Lufttemperatur'], color='tab:gray', label='Lufttemperatur', lw=1.5, zorder=1)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['Nederbördsmängd'], color='tab:blue', label='Nederbördsmängd', lw=1.5, zorder=1)
    elif moving_avg in window_sizes:
        window = window_sizes[moving_avg]
        data_gdf['varde_smooth'] = data_gdf['varde'].rolling(window=window, min_periods=1).mean()
        data_gdf['utvecklingsstadium_smooth'] = data_gdf['utvecklingsstadium'].rolling(window=window, min_periods=1).mean()
        data_gdf['Lufttemperatur_smooth'] = data_gdf['Lufttemperatur'].rolling(window=window, min_periods=1).mean()
        data_gdf['Nederbördsmängd_smooth'] = data_gdf['Nederbördsmängd'].rolling(window=window, min_periods=1).mean()

        data_gdf.loc[data_gdf['varde'].isna(), 'varde_smooth'] = None
        data_gdf.loc[data_gdf['utvecklingsstadium'].isna(), 'utvecklingsstadium_smooth'] = None
        data_gdf.loc[data_gdf['Lufttemperatur'].isna(), 'Lufttemperatur_smooth'] = None    
        data_gdf.loc[data_gdf['Nederbördsmängd'].isna(), 'Nederbördsmängd_smooth'] = None   

        plt.plot(data_gdf['graderingsdatum'], data_gdf['varde_smooth'], color='#d62728', label=f"{str(pest).capitalize()} (Smoothed)", lw=1.5)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['utvecklingsstadium_smooth'], color='#2ca02c', label=f"{str(crop).capitalize()} (Smoothed)", lw=1.5)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['Lufttemperatur_smooth'], color='tab:gray', label=f"{'Lufttemperatur'} (Smoothed)", lw=1.5)
        plt.plot(data_gdf['graderingsdatum'], data_gdf['Nederbördsmängd_smooth'], color='tab:blue', label=f"{'Nederbördsmängd'} (Smoothed)", lw=1.5)
    else:
        raise ValueError("Invalid moving_avg parameter. Choose from: None, 'low', 'mid', or 'high'.")
    
    plt.xlabel('DATUM')
    plt.ylabel('VÄRDEN')
    plt.title(f'{str(crop).capitalize()} & {str(pest).capitalize()} | Location. Lat={round(latitude, 5)}, Long: {round(longitude, 5)}')
    
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()