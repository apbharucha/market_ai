import pandas as pd

def correlation_matrix(price_series_dict):
    df = pd.concat(price_series_dict, axis=1)
    return df.pct_change().corr()
