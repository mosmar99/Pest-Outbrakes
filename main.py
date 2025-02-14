import api

if __name__ == "__main__":
    gradings_df = api.get_gradings()
    gradings_df = api.sweref99tm_to_wgs84(gradings_df)

    print("\nGradings DataFrame:")
    print(gradings_df.head(7))

