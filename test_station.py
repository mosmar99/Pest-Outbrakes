import api
import smhi_api
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString


if __name__ == "__main__":
    gradings_df = api.get_gradings(from_date="2019-03-01", crop="Höstvete", pest="svartpricksjuka")
    # gradings_df = api.sweref99tm_to_wgs84(gradings_df)
    jbv_gdf = gpd.GeoDataFrame(gradings_df, geometry=gpd.points_from_xy(gradings_df.longitud, gradings_df.latitud), crs="EPSG:3006")
    jbv_gdf["geometry"] = jbv_gdf["geometry"].to_crs("EPSG:4326")

    smhi_stations_gdf = smhi_api.get_all_stations_on_parameter_id()


    utm_crs = jbv_gdf.estimate_utm_crs()
    jbv_gdf = jbv_gdf.to_crs(utm_crs)
    smhi_stations_gdf = smhi_stations_gdf.to_crs(utm_crs)
    smhi_stations_gdf["station_loc"] = smhi_stations_gdf["geometry"]

    jbv_gdf = jbv_gdf.sjoin_nearest(smhi_stations_gdf, how="left")

    # print(f"JBV CRS: {jbv_gdf.crs}, SMHI CRS: {smhi_stations_gdf.crs}")
    # print("\nGradings DataFrame:")
    # print(jbv_gdf.head(7))
    print("\nStations DataFrame:")
    print(smhi_stations_gdf.head(7))

    jbv_gdf['connection'] = jbv_gdf.apply(
    lambda row: LineString([row['geometry'], row['station_loc']]), axis=1)

    connections = gpd.GeoDataFrame(jbv_gdf['connection'], geometry='connection', crs=utm_crs)


    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the original points
    jbv_gdf["geometry"].plot(ax=ax, color='green', label='Field', markersize=20)
    smhi_stations_gdf["station_loc"].plot(ax=ax, color='gray', label='Unused Station', markersize=10)
    jbv_gdf["station_loc"].plot(ax=ax, color='blue', label='Used Station', markersize=20)

    # Plot the connection lines
    connections.plot(ax=ax, color='gray', linewidth=0.8, linestyle='--')

    plt.legend()
    plt.title('Station selection')
    plt.show()