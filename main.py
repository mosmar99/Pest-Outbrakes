import api
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import process
import visualize as viz

if __name__ == "__main__":
    crop='höstvete'
    pest = 'Svartpricksjuka'
    gradings_df = api.get_gradings(from_date="2015-08-04", to_date="2025-02-01", crop=crop, pest=pest)
    gradings_df = api.sweref99tm_to_wgs84(gradings_df)

    top_latitud, top_longitud = process.get_uniq_plant_coord(gradings_df, selector=1)
    filtered_rows_df = process.coordinate_rowfilter(gradings_df, top_latitud, top_longitud)
    extracted_df = process.transform_date(process.feature_extraction(filtered_rows_df, crop, pest))

    viz.lineplot(extracted_df, crop, pest, top_latitud, top_longitud)

    print("Extracted Data:")
    print(extracted_df.head())
    print(extracted_df.shape)
    