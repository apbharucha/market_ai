
import pandas as pd
from data_sources import get_stock

def check_vix_historical():
    df = get_stock("^VIX", period="10d")
    print("VIX Historical Data (Last 5 days):")
    print(df.tail(5))
    
    if '2026-02-09' in df.index.strftime('%Y-%m-%d'):
        row_2026_02_09 = df[df.index.strftime('%Y-%m-%d') == '2026-02-09']
        print("\nData for 2026-02-09:")
        print(row_2026_02_09)
        
        # Check previous day
        idx = df.index.get_loc(row_2026_02_09.index[0])
        if idx > 0:
            prev_row = df.iloc[idx-1]
            print("\nPrevious day data:")
            print(prev_row)
            
            change = (row_2026_02_09['Close'].values[0] / prev_row['Close'] - 1) * 100
            print(f"\nCalculated Change for 2026-02-09: {change:.2f}%")

if __name__ == "__main__":
    check_vix_historical()
