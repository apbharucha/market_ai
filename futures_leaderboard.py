import pandas as pd
import streamlit as st
import yfinance as yf

FUTURES = {
    "ES (S&P 500)": "ES=F",
    "NQ (Nasdaq)": "NQ=F",
    "CL (Crude Oil)": "CL=F",
    "GC (Gold)": "GC=F",
    "SI (Silver)": "SI=F",
    "ZB (Bonds)": "ZB=F"
}

@st.cache_data(ttl=3600, show_spinner=False)
def futures_rank(lookback=21):
    rows = []
    
    tickers = list(FUTURES.values())
    
    try:
        # Bulk download
        data = yf.download(tickers, period="3mo", group_by='ticker', progress=False, threads=True)
        
        for name, ticker in FUTURES.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.levels[0]:
                        continue
                    df = data[ticker].dropna()
                else:
                    if ticker != tickers[0]: continue
                    df = data.dropna()
                    
                if df.empty or len(df) < lookback:
                    continue

                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                
                current = float(close_col.iloc[-1])
                prev = float(close_col.iloc[-lookback])
                ret = ((current / prev) - 1) * 100 if prev > 0 else 0
                rows.append({"Contract": name, "TrendScore": round(ret, 2)})
            except Exception:
                continue
    except Exception:
         return pd.DataFrame(columns=["Contract", "TrendScore"])

    if len(rows) == 0:
        return pd.DataFrame(columns=["Contract", "TrendScore"])

    return pd.DataFrame(rows).sort_values("TrendScore", ascending=False)

def futures_data_health_check(lookback=21):
    """Quick diagnostic to verify futures data availability and basic inputs."""
    status = {"ok": True, "checked": [], "errors": []}
    tickers = list(FUTURES.values())
    try:
        data = yf.download(tickers, period="3mo", group_by="ticker", progress=False, threads=True)
        for name, ticker in FUTURES.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.levels[0]:
                        raise ValueError("missing_ticker_data")
                    df = data[ticker].dropna()
                else:
                    if ticker != tickers[0]:
                        continue
                    df = data.dropna()

                if df.empty or len(df) < lookback:
                    raise ValueError("insufficient_lookback")

                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]

                current = float(close_col.iloc[-1])
                prev = float(close_col.iloc[-lookback])
                if prev <= 0:
                    raise ValueError("invalid_previous_close")

                status["checked"].append({"contract": name, "current": current, "prev": prev})
            except Exception as e:
                status["ok"] = False
                status["errors"].append({"contract": name, "error": str(e)})
    except Exception as e:
        status["ok"] = False
        status["errors"].append({"contract": "ALL", "error": str(e)})

    return status
