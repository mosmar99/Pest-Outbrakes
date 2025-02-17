import pandas as pd

def flatten_data(df):
    """
    Flattens the inital dataframe from a api call by extracting the items in 'graderingstillfalleList' structure into a flat DataFrame.
    
    Args:
        df (pd.Dataframe): For now only tested with to work with a similar API call as this:
        http://api.jordbruksverket.se/rest/povapi/graderingar?fran=2010-06-04&till=2019-11-22&groda=Höstvete&skadegorare=alla
    
    Returns:
        pd.Dataframe: This produces a new dataframe with one output row per pest (per original row) from graderingstillfalleList. 
        New features of the datasets added per row:
        - pest: the pest name.
        - measuring_methods: a list of the measuring methods.
        - timestamps: a list of datetime objects for the grading event dates for the pest.
        - measurement_values: a list of numeric measurement values for that pest.
        - All features are translated to english.
    """
    records = []

    for _, row in df.iterrows():
        area                = row.get('delomrade')
        county              = row.get('lan')
        organic             = row.get('ekologisk')
        pre_pre_crop        = row.get('forforfrukt')
        pre_crop            = row.get('forfrukt')
        crop                = row.get('groda')
        crop_type           = row.get('sort')
        soil_type           = row.get('jordart')
        lat_val             = row.get('latitud')
        lon_val             = row.get('longitud')
        plowed              = row.get('plojt')
        sowing_date         = row.get('sadatum')
        harvest_year        = row.get('skordear')
        seedling_treatment  = row.get('broddbehandling')

        graderings_tillfallen = row.get('graderingstillfalleList', [])
        if not isinstance(graderings_tillfallen, list):
            continue
        pest_groups = {}

        for tillfalle in graderings_tillfallen:
            datum_str   = tillfalle.get('graderingsdatum')
            date_parsed = pd.to_datetime(datum_str, errors='coerce') if datum_str else None
            development_stage = tillfalle.get('utvecklingsstadium')

            graderingar = tillfalle.get('graderingList', [])
            for g in graderingar:
                pest_name = g.get('skadegorare')
                value = g.get('varde')
                measuring_method = g.get('matmetod')

                if pest_name not in pest_groups:
                    pest_groups[pest_name] = {
                        "timestamps": [],
                        "measurement_values": [],
                        "measuring_methods": set(),
                        "development_stages": []
                    }

                pest_groups[pest_name]["timestamps"].append(date_parsed)
                pest_groups[pest_name]["measurement_values"].append(value)
                pest_groups[pest_name]["measuring_methods"].add(measuring_method)
                pest_groups[pest_name]["development_stages"].append(development_stage)

        for pest_name, measurements in pest_groups.items():
            records.append({
                "area": area,
                "county": county,
                "organic": organic,
                "preprecrop": pre_pre_crop,
                "pre_crop": pre_crop,
                "crop": crop,
                "crop_type": crop_type,
                "soil_type": soil_type,
                "latitud": lat_val,
                "longitud": lon_val,
                "plowed": plowed,
                "sowing_date": sowing_date,
                "harvest_year": harvest_year,
                "seedling_treatment": seedling_treatment,
                "pest": pest_name,
                "measuring_methods": list(measurements["measuring_methods"]),
                "timestamps": measurements["timestamps"],
                "measurement_values": measurements["measurement_values"],
                "development_stages": measurements["development_stages"]
            })

    return pd.DataFrame(records)

#The rest of the filtering methods are dependent on flattening the data recieved from the API

def filter_crop(df, crop_name):
    """
    Returns rows from df where the 'crop' column matches the given crop name.

    Args:
        df (pd.DataFrame): A flattened dataframe (from flatten_data).
        crop_name(str): The name of the crop.
    
    Returns:
        pd.Dataframe: Rows where crop_name matches df['crop'].
    """
    return df[df['crop'] == crop_name]

def filter_area(df, area_name):
    """
    Returns rows from df where the 'area' column matches the given area name.
    Args:
        df (pd.DataFrame): A flattened dataframe (from flatten_data).
        area_name(str): The name of the area.
    
    Returns:
        pd.Dataframe: Rows where area_name matches df['area'].
    """
    return df[df['area'] == area_name]

def filter_by_pest(df, pest_name):
    """
    Returns rows from df where the 'pest column matches the given pest name.
    Args:
        df (pd.DataFrame): A flattened dataframe (from flatten_data).
        pest_name(str): The name of the pest.
    
    Returns:
        pd.Dataframe: Rows where pest_name matches df['pest'].
    """
    return df[df['pest'] == pest_name]

# TODO: Add more filtering methods depending on needs.