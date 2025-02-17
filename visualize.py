import matplotlib.pyplot as plt
import pandas as pd

def lineplot(data_df, crop, pest, latitud, longitud, moving_avg=None):
    if moving_avg == None:        
        plt.figure(figsize=(12, 6))
        plt.plot(data_df['graderingsdatum'], data_df['varde'], color='tab:blue', label=str(pest).capitalize(), lw=1.5)
        plt.plot(data_df['graderingsdatum'], data_df['utvecklingsstadium'], color='tab:orange', label=str(crop).capitalize(), lw=1.5)

        plt.xlabel('DATUM')
        plt.ylabel('VÄRDEN')
        plt.title(f'{str(crop).capitalize()} & {str(pest).capitalize()} | Location. Lat={round(latitud, 5)}, Long: {round(longitud, 5)}')

        plt.legend()
        plt.grid()
        plt.show()
    elif moving_avg == "low":
        pass
    elif moving_avg == "mid":
        pass
    elif moving_avg == "high":
        pass
    else:
        raise ValueError("Wrong parameter values.")