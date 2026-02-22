import ta
import pandas as pd
import numpy as np

def add_indicators(df):
    if df.empty:
        return df
        
    # Check if we have enough data for indicators (need at least 20 rows for typical EMAs/BB)
    if len(df) < 20:
        return df
    
    # Ensure Close is a 1D Series
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    elif not isinstance(close_series, pd.Series):
        close_series = pd.Series(close_series)
    
    # Ensure the series is 1D and has a proper index
    if close_series.ndim > 1:
        close_series = close_series.squeeze()
    
    df = df.copy()
    df["ema20"] = ta.trend.ema_indicator(close_series, 20)
    df["ema50"] = ta.trend.ema_indicator(close_series, 50)
    df["rsi"] = ta.momentum.rsi(close_series, 14)
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], close_series, window=14)
    
    # Add Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # Add MACD
    macd = ta.trend.MACD(close=close_series)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    return df.dropna()

def detect_candlestick_patterns(df):
    """
    Detects common candlestick patterns in the DataFrame.
    Returns a dictionary of detected patterns for the latest candle(s).
    """
    if df is None or df.empty or len(df) < 5:
        return {}
    
    # Get last few rows for pattern recognition
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    patterns = []
    
    # Calculate body and shadows
    body = abs(last['Close'] - last['Open'])
    upper_shadow = last['High'] - max(last['Close'], last['Open'])
    lower_shadow = min(last['Close'], last['Open']) - last['Low']
    avg_body = abs(df['Close'] - df['Open']).rolling(10).mean().iloc[-1]
    
    # 1. Doji
    if body <= avg_body * 0.1:
        patterns.append("Doji (Indecision)")
    
    # 2. Hammer (Bullish) / Hanging Man (Bearish)
    if lower_shadow > body * 2 and upper_shadow < body * 0.5:
        if last['Close'] < prev['Close']: # Downtrend context roughly
            patterns.append("Hammer (Potential Bullish Reversal)")
        else:
            patterns.append("Hanging Man (Potential Bearish Reversal)")
            
    # 3. Shooting Star (Bearish) / Inverted Hammer (Bullish)
    if upper_shadow > body * 2 and lower_shadow < body * 0.5:
        if last['Close'] > prev['Close']:
            patterns.append("Shooting Star (Potential Bearish Reversal)")
        else:
            patterns.append("Inverted Hammer (Potential Bullish Reversal)")
            
    # 4. Bullish Engulfing
    if (prev['Close'] < prev['Open'] and # Prev was red
        last['Close'] > last['Open'] and # Current is green
        last['Close'] > prev['Open'] and 
        last['Open'] < prev['Close']):
        patterns.append("Bullish Engulfing (Strong Bullish)")
        
    # 5. Bearish Engulfing
    if (prev['Close'] > prev['Open'] and # Prev was green
        last['Close'] < last['Open'] and # Current is red
        last['Open'] > prev['Close'] and 
        last['Close'] < prev['Open']):
        patterns.append("Bearish Engulfing (Strong Bearish)")
        
    # 6. Morning Star (3-candle pattern)
    if (prev2['Close'] < prev2['Open'] and # First red
        abs(prev['Close'] - prev['Open']) < avg_body * 0.5 and # Second small body
        last['Close'] > last['Open'] and # Third green
        last['Close'] > (prev2['Open'] + prev2['Close'])/2): # Closes above midpoint of first
        patterns.append("Morning Star (Bullish Reversal)")
        
    # 7. Evening Star (3-candle pattern)
    if (prev2['Close'] > prev2['Open'] and # First green
        abs(prev['Close'] - prev['Open']) < avg_body * 0.5 and # Second small body
        last['Close'] < last['Open'] and # Third red
        last['Close'] < (prev2['Open'] + prev2['Close'])/2): # Closes below midpoint of first
        patterns.append("Evening Star (Bearish Reversal)")

    return patterns

