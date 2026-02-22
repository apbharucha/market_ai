import numpy as np
import pandas as pd
from data_sources import get_stock, get_fx, get_futures_proxy

def _get_asset_data(symbol):
    """Get data for any asset type"""
    # Try stock first
    df = get_stock(symbol)
    if not df.empty:
        return df
    
    # Try futures
    if "=F" in symbol or symbol.startswith("ES") or symbol.startswith("NQ"):
        df = get_futures_proxy(symbol)
        if not df.empty:
            return df
    
    # Try FX (handle standard notations like USD/JPY, USD-JPY, USD_JPY)
    if any(x in symbol for x in ["=X", "_", "-", "/"]):
        # Normalize to Oanda format (USD_JPY)
        fx_symbol_oanda = symbol.replace("=X", "").replace("-", "_").replace("/", "_")
        try:
            df = get_fx(fx_symbol_oanda)
            if not df.empty:
                return df
        except Exception:
            pass # Fallback to Yahoo
            
        # Fallback to Yahoo format (USDJPY=X or JPY=X)
        # Try constructing Yahoo symbol: USD/JPY -> USDJPY=X
        yf_symbol = symbol.replace("/", "").replace("-", "").replace("_", "") + "=X"
        df = get_stock(yf_symbol)
        if not df.empty:
            return df
            
    # Try as stock again (for crypto like BTC-USD)
    return get_stock(symbol)

def correlation_matrix(symbols):
    prices = {}

    for s in symbols:
        try:
            df = _get_asset_data(s)
            if not df.empty:
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                # Ensure index is datetime and sorted
                close_col.index = pd.to_datetime(close_col.index)
                prices[s] = close_col.sort_index()
        except Exception as e:
            print(f"Error fetching data for {s} in correlation: {e}")

    if not prices:
        return pd.DataFrame()

    # Create combined DataFrame
    df = pd.DataFrame(prices)
    
    # Handle missing data more robustly
    # 1. Forward fill (limit to 3 days to avoid stale data)
    df = df.ffill(limit=3)
    # 2. Backward fill for the very start
    df = df.bfill(limit=3)
    # 3. Only then drop remaining NaNs
    df = df.dropna()
    
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    
    returns = df.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()
    
    return returns.corr()

def portfolio_var(symbols, weights, confidence=0.95):
    prices = {}
    valid_weights = []
    
    # Ensure symbols and weights match length initially
    if len(symbols) != len(weights):
        weights = [1.0/len(symbols)] * len(symbols)

    for i, s in enumerate(symbols):
        try:
            df = _get_asset_data(s)
            if not df.empty:
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_col.index = pd.to_datetime(close_col.index)
                prices[s] = close_col.sort_index()
                valid_weights.append(weights[i])
        except Exception:
            continue

    if not prices:
        return 0, 0

    # Normalize weights for valid assets
    total_weight = sum(valid_weights)
    if total_weight > 0:
        valid_weights = [w / total_weight for w in valid_weights]
    else:
        valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)

    df = pd.DataFrame(prices)
    df = df.ffill(limit=3).bfill(limit=3).dropna()
    
    if df.empty or len(df) < 10:
        return 0, 0
    
    returns = df.pct_change().dropna()
    if returns.empty or len(returns) < 5:
        return 0, 0

    cov = returns.cov()
    if cov.empty:
        return 0, 0
    
    w_array = np.array(valid_weights)
    
    # Variance calculation: w^T * Cov * w
    portfolio_vol = np.sqrt(np.dot(w_array, np.dot(cov, w_array)))

    # Use Monte Carlo or historical percentile for VaR
    portfolio_returns = returns.dot(w_array)
    if len(portfolio_returns) == 0:
        return 0, 0
    
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    # Annualize volatility (assuming daily returns)
    annual_vol = portfolio_vol * np.sqrt(252)
    
    return round(var * 100, 2), round(annual_vol * 100, 2)

def position_size(account_size, risk_pct, stop_pct):
    risk_amount = account_size * risk_pct
    size = risk_amount / stop_pct
    return round(size, 2)
