import os
import data_processing.api as api
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

URI = {
    "all_crops": "https://api.jordbruksverket.se/rest/povapi/groda",
    "all_pests": "https://api.jordbruksverket.se/rest/povapi/skadegorare",
    "pests_linked_to_gradings": "https://api.jordbruksverket.se/rest/povapi/skadegorare/alla",
    "crops_linked_to_gradings": "https://api.jordbruksverket.se/rest/povapi/groda/alla",
    "gradings": "https://api.jordbruksverket.se/rest/povapi/graderingar",
    "update_information": "https://api.jordbruksverket.se/rest/povapi/uppdateringsinformation",
}

load_dotenv()
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")
auth_details = HTTPBasicAuth(USERNAME, PASSWORD)
print(USERNAME, PASSWORD)

def get_all_pests():
    """
    Fetches and returns a dataframe of all pests from the API.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing all pest data if successful, otherwise None.
    """
    return api.fetch_data(URI["all_pests"], auth=auth_details, return_format='pandas')

def get_all_crops():
    """
    Fetches and returns a dataframe of all crops from the API.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing all crop data if successful, otherwise None.
    """
    return api.fetch_data(URI["all_crops"], auth=auth_details, return_format='pandas')

def get_crops_linked_to_gradings():
    """
    Retrieves a dataframe of crops associated with grading data.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing crops linked to grading data if successful, otherwise None.
    """
    return api.fetch_data(URI["crops_linked_to_gradings"], auth=auth_details, return_format='pandas')

def get_pests_linked_to_gradings():
    """
    Retrieves a dataframe of pests associated with grading data.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing pests linked to grading data if successful, otherwise None.
    """
    return api.fetch_data(URI["pests_linked_to_gradings"], auth=auth_details, return_format='pandas')

def get_gradings(from_date="2019-08-04", to_date="2019-11-22", crop="Potatis", pest="Bladlus"):
    """
    Fetches grading data for a date range, crop, and pest from the API.

    Args:
        from_date (str): The start date for filtering data (format YYYY-MM-DD).
        to_date (str): The end date for filtering data (format YYYY-MM-DD).
        crop (str): The crop name to retrieve grading data for.
        pest (str): The pest name to retrieve grading data for.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing grading data if successful, otherwise None.
    """
    endpoint = f"{URI['gradings']}?fran={from_date}&till={to_date}&groda={crop}&skadegorare={pest}"
    return api.fetch_data(endpoint, auth=auth_details, return_format='pandas')

def get_update_information():
    """
    Retrieves and returns the latest update information from the API.

    Returns:
        Optional[pd.DataFrame]: A dataframe containing update information if successful, otherwise None.
    """
    return api.fetch_data(URI["update_information"], auth=auth_details, return_format='pandas')
