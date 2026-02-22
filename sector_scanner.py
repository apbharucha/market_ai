import pandas as pd
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_sources import get_stock

SECTOR_MAP = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "INTC", "CRM", "ADBE", "CSCO", 
        "ORCL", "IBM", "TXN", "QCOM", "AVGO", "AMAT", "MU", "LRCX", "ADI", "NOW", 
        "SNOW", "PLTR", "UBER", "ABNB", "SQ", "SHOP", "ZM", "DOCU", "NET", "CRWD"
    ],
    "Financials": [
        "JPM", "GS", "BAC", "WFC", "C", "MS", "BLK", "AXP", "V", "MA", 
        "PYPL", "COIN", "SCHW", "USB", "PNC", "TFC", "BK", "STT", "KKR", "BX",
        "AIG", "ALL", "CB", "MMC", "AON", "PGR", "TRV", "HIG", "CME", "ICE"
    ],
    "Energy": [
        "XOM", "CVX", "SLB", "COP", "EOG", "OXY", "PXD", "MPC", "VLO", "PSX", 
        "HAL", "BKR", "KMI", "WMB", "OKE", "HES", "DVN", "FANG", "MRO", "CTRA",
        "APA", "EQT", "TRGP", "LNG", "EPD", "MPLX", "ET", "PAA", "BP", "SHEL"
    ],
    "Healthcare": [
        "JNJ", "PFE", "LLY", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", 
        "AMGN", "GILD", "CVS", "CI", "ELV", "HCA", "SYK", "ISRG", "EW", "REGN",
        "VRTX", "ZTS", "BDX", "BSX", "HUM", "MCK", "CNC", "IQV", "A", "MTD"
    ],
    "Consumer": [
        "AMZN", "TSLA", "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", 
        "DIS", "SBUX", "TGT", "LOW", "TJX", "LULU", "CMG", "MAR", "HLT", "BKNG",
        "CL", "KMB", "EL", "MNST", "STZ", "GIS", "K", "MO", "PM", "WBA"
    ],
    "Industrials": [
        "CAT", "DE", "HON", "GE", "MMM", "UPS", "FDX", "RTX", "LMT", "BA",
        "UNP", "CSX", "NSC", "WM", "RSG", "ETN", "ITW", "EMR", "PH", "CMI",
        "GD", "NOC", "LHX", "TXT", "HII", "TDG", "HEI", "ROK", "AME", "DOV"
    ],
    "Materials": [
        "LIN", "SHW", "APD", "FCX", "NEM", "SCCO", "DOW", "DD", "CTVA", "ECL",
        "PPG", "VMC", "MLM", "NUE", "STLD", "ALB", "FMC", "MOS", "CF", "LYB"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "WEC",
        "ES", "ED", "DTE", "ETR", "FE", "PPL", "AEE", "CMS", "CNP", "ATO"
    ],
    "Real Estate": [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR", "VICI",
        "AVB", "EQR", "CBRE", "CSGP", "EXR", "MAA", "SUI", "ESS", "UDR", "INVH"
    ]
}

def _fetch_symbol_score(sym, lookback):
    """Helper to fetch score for a single symbol."""
    try:
        df = get_stock(sym)
        if df is None or df.empty or len(df) < lookback:
            return None
        
        close_col = df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        
        current = float(close_col.iloc[-1])
        prev = float(close_col.iloc[-lookback])
        
        # Avoid division by zero
        if prev <= 0:
            return None
            
        return ((current / prev) - 1) * 100
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def scan_sectors(lookback=21):
    rows = []
    
    # 1. Collect all unique symbols
    all_symbols = []
    for symbols in SECTOR_MAP.values():
        all_symbols.extend(symbols)
    all_symbols = list(set(all_symbols))
    
    # 2. Bulk download data (much faster than loop)
    # We download 3mo to ensure we have enough data for lookback
    try:
        data = yf.download(all_symbols, period="3mo", group_by='ticker', progress=False, threads=True)
    except Exception:
        # Fallback to empty if download fails completely
        return pd.DataFrame(columns=["Sector", "TrendScore", "AssetsUsed"])
        
    # 3. Process each sector
    for sector, symbols in SECTOR_MAP.items():
        scores = []
        for sym in symbols:
            try:
                # Handle yfinance multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.levels[0]:
                        continue
                    df = data[sym].dropna()
                else:
                    # Single symbol case (unlikely given the list, but good safety)
                    if sym != all_symbols[0]: 
                         # Should not happen if multiple symbols requested
                         continue 
                    df = data.dropna()

                if df.empty or len(df) < lookback:
                    continue
                
                # Get close prices
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                
                current = float(close_col.iloc[-1])
                prev = float(close_col.iloc[-lookback])
                
                if prev > 0:
                    ret = ((current / prev) - 1) * 100
                    scores.append(ret)
            except Exception:
                continue
        
        if len(scores) > 0:
            rows.append({
                "Sector": sector,
                "TrendScore": round(sum(scores) / len(scores), 2),
                "AssetsUsed": len(scores)
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=["Sector", "TrendScore", "AssetsUsed"])

    return pd.DataFrame(rows).sort_values("TrendScore", ascending=False)
