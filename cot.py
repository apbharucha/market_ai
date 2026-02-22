import pandas as pd

COT_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2024.zip"

def load_cot():
    df = pd.read_csv(COT_URL, compression="zip")
    return df
