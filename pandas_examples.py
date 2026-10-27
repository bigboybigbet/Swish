import os

import numpy as np
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup

file_path = "nba.csv"

# Cac cach doc / ghi du lieu voi pandas:
# df = pd.read_csv(file_path)      # doc file CSV
# df["col"]                        # lay 1 cot
# df.to_csv("out.csv")             # ghi ra file CSV
# pd.read_excel("file.xlsx")       # doc file Excel
# pd.read_json("file.json")        # doc file JSON

person = {"name": "khanh", "age": 21}  # this is a dictionary in python


def read_csv(file_path):
    if not file_path or not os.path.exists(file_path):
        return "no file"
    ext = os.path.splitext(file_path)[1].lower()  # phan duoi sau dau "."
    if ext not in [".csv", ".xlsx", ".json"]:
        return "wrong format"
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext == ".xlsx":
        return pd.read_excel(file_path)
    return pd.read_json(file_path)


# GET, POST
# GET:  get data from api, mostly included in url (ex: google, web, ...)
# POST: get data from api, by giving additional information called payload

# tim du lieu
