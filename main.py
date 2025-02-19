import data_processing.jbv_api as jbv_api
import data_processing.jbv_process as jbv_process
import visualize as viz

if __name__ == "__main__":
    crop='höstvete'
    pest = 'Svartpricksjuka'
    gradings_df = jbv_api.get_gradings(from_date="2015-08-04", to_date="2025-02-01", crop=crop, pest=pest)
    print("---FETCHED JBV-DATA")

    gradings_df = jbv_process.sweref99tm_to_wgs84(jbv_process.drop_rows_no_coords(gradings_df))
    top_latitud, top_longitud = jbv_process.get_uniq_plantation_coord(gradings_df, selector=1)
    plantation_df = jbv_process.get_plantation_by_coord(gradings_df, top_latitud, top_longitud)
    plantation_df = jbv_process.transform_date(jbv_process.feature_extraction(plantation_df, crop, pest))
    print("---PROCESSED JBV-DATA")

    # print("---FETCHED JBV-DATA")
    # print("---PROCESSED SMHI-DATA")

    # viz.lineplot(extracted_df, crop, pest, top_latitud, top_longitud)
    # print("Extracted Data:")
    # print(extracted_df.head())
    # print(extracted_df.shape)
    