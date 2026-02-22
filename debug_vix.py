
import pandas as pd
from data_sources import get_stock

def check_vix():
    symbol = "^VIX"
    print(f"Fetching {symbol}...")
    try:
        df = get_stock(symbol, period="5d")
        if not df.empty and "Close" in df.columns:
            close_col = df["Close"]
            if isinstance(close_col, pd.DataFrame):
                print("Close is DataFrame, flattening...")
                close_col = close_col.iloc[:, 0]
            
            print("\nRecent Data:")
            print(close_col.tail())
            
            current = float(close_col.iloc[-1])
            prev_close = float(close_col.iloc[-2]) if len(close_col) > 1 else current
            change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0
            change_pts = current - prev_close
            
            print(f"\nCurrent: {current}")
            print(f"Prev:    {prev_close}")
            print(f"Change Pts: {change_pts:.2f}")
            print(f"Change %:   {change_pct:.2f}%")
        else:
            print("DF empty or no Close col")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_vix()
