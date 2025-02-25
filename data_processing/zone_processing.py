import geopandas as gpd

def assign_growth_zone(gdf):
    CRS = 'EPSG:4326'
    zones = gpd.read_file(".\\geodata\\hardiness\\zones.shp")
    zones['zone'] = zones['name']
    zones = zones.drop(['id', 'name'], axis=1)
    gdf_joined = gpd.sjoin(gdf.to_crs(CRS), zones.to_crs(CRS), how="left", predicate="within").drop('index_right', axis=1).to_crs(CRS)
    return gdf_joined