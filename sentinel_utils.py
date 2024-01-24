import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from creds import *

#from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, Geometry)

def get_sentinel_cataloge(aoi, start_date, end_date, data_collection):
    json = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}) and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z").json()
    df = pd.DataFrame.from_dict(json['value'])
    print(df.head(5))
    return df["Id"].iloc[0]


def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                          data=data,
                          )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]


keycloak_token = get_keycloak(username, password)

start_date = "2023-03-10"
end_date = "2023-03-17"
data_collection = "SENTINEL-2"
#
aoi = "POLYGON((-121.68932106749568 50.0810305019437, -121.68932106749568 50.01438028639592, -121.56892190985448 50.01438028639592, -121.56892190985448 50.0810305019437, -121.68932106749568 50.0810305019437))'"

product_id = get_sentinel_cataloge(aoi, start_date, end_date, data_collection)

def download_as_zip(keycloak_token, product_id):
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {keycloak_token}'})
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    response = session.get(url, allow_redirects=False)
    while response.status_code in (301, 302, 303, 307):
        url = response.headers['Location']
        response = session.get(url, allow_redirects=False)

    file = session.get(url, verify=False, allow_redirects=True)

    with open(f"2023-03-17.zip", 'wb') as p:
        p.write(file.content)

download_as_zip(keycloak_token, product_id)