import pandas as pd
import streamlit as st

try:
    from data_sources import get_fx

    HAS_GET_FX = True
except ImportError:
    HAS_GET_FX = False

# Canonical FX pair registry: display name → 6-char base code
FX_PAIRS = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "AUD_USD": "AUDUSD",
    "USD_CAD": "USDCAD",
    "USD_CHF": "USDCHF",
    "NZD_USD": "NZDUSD",
    "EUR_JPY": "EURJPY",
    "GBP_JPY": "GBPJPY",
    "EUR_GBP": "EURGBP",
    "EUR_CHF": "EURCHF",
    "EUR_AUD": "EURAUD",
    "AUD_JPY": "AUDJPY",
    "GBP_CHF": "GBPCHF",
    "CAD_JPY": "CADJPY",
    "NZD_JPY": "NZDJPY",
    "USD_TRY": "USDTRY",
    "USD_ZAR": "USDZAR",
    "USD_MXN": "USDMXN",
    "USD_NOK": "USDNOK",
}

# Subset used for quick dashboard scans
CORE_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "USD_CHF",
    "NZD_USD",
    "EUR_JPY",
    "EUR_GBP",
    "EUR_CHF",
    "EUR_AUD",
    "AUD_JPY",
    "GBP_JPY",
]


def _fetch_pair_df(clean6: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch OHLC data for a 6-char FX code (e.g. 'EURUSD') via get_fx,
    with a direct Yahoo fallback using the canonical =X format.
    Returns an empty DataFrame on failure.
    """
    if HAS_GET_FX:
        try:
            df = get_fx(clean6, period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    # Fallback: yfinance direct (=X format)
    try:
        import yfinance as yf

        yf_ticker = clean6 + "=X"
        df = yf.download(yf_ticker, period=period, progress=False, threads=False)
        if df is not None and not df.empty:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def scan_fx(lookback: int = 21, pairs: list = None) -> pd.DataFrame:
    """
    Scan FX pairs and return a DataFrame with trend/momentum metrics.

    Parameters
    ----------
    lookback : int
        Number of trading days used for return calculation.
    pairs : list[str] | None
        List of display-name keys from FX_PAIRS to scan.
        Defaults to CORE_PAIRS.

    Returns
    -------
    pd.DataFrame
        Indexed by pair display name with columns:
        TrendScore, Price, Change1D%, Change5D%, RSI14, AboveMA20
    """
    if pairs is None:
        pairs = CORE_PAIRS

    rows = []
    for pair_key in pairs:
        clean6 = FX_PAIRS.get(pair_key)
        if not clean6:
            continue

        try:
            df = _fetch_pair_df(clean6, period="6mo")
            if df is None or df.empty:
                continue

            # Resolve Close column
            close = df.get("Close", df.get("close"))
            if close is None:
                continue
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()

            if len(close) < max(lookback, 20):
                continue

            price = float(close.iloc[-1])
            prev_lb = (
                float(close.iloc[-lookback])
                if len(close) > lookback
                else float(close.iloc[0])
            )
            prev_1d = float(close.iloc[-2]) if len(close) >= 2 else price
            prev_5d = float(close.iloc[-5]) if len(close) >= 5 else float(close.iloc[0])

            trend_score = (
                round(((price / prev_lb) - 1) * 100, 3) if prev_lb > 0 else 0.0
            )
            change_1d = round(((price / prev_1d) - 1) * 100, 3) if prev_1d > 0 else 0.0
            change_5d = round(((price / prev_5d) - 1) * 100, 3) if prev_5d > 0 else 0.0

            # RSI-14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = (100 - 100 / (1 + rs)).iloc[-1]
            rsi14 = round(float(rsi), 1) if pd.notna(rsi) else 50.0

            # 20-period SMA
            sma20 = float(close.rolling(20).mean().iloc[-1])
            above_ma20 = price > sma20

            rows.append(
                {
                    "Pair": pair_key,
                    "Price": round(price, 5),
                    "TrendScore": trend_score,
                    "Change1D%": change_1d,
                    "Change5D%": change_5d,
                    "RSI14": rsi14,
                    "AboveMA20": above_ma20,
                }
            )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "Pair",
                "Price",
                "TrendScore",
                "Change1D%",
                "Change5D%",
                "RSI14",
                "AboveMA20",
            ]
        )

    result = (
        pd.DataFrame(rows).set_index("Pair").sort_values("TrendScore", ascending=False)
    )
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def get_fx_quote(pair_key: str) -> dict:
    """
    Return a quick quote dict for a single FX pair.

    Returns
    -------
    dict with keys: pair, price, change_1d_pct, change_5d_pct, rsi14
    """
    clean6 = FX_PAIRS.get(pair_key, pair_key.replace("/", "").replace("_", "").upper())
    try:
        df = _fetch_pair_df(clean6, period="1mo")
        if df is None or df.empty:
            return {}
        close = df.get("Close", df.get("close"))
        if close is None:
            return {}
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < 2:
            return {}

        price = float(close.iloc[-1])
        prev_1d = float(close.iloc[-2])
        prev_5d = float(close.iloc[-5]) if len(close) >= 5 else float(close.iloc[0])

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - 100 / (1 + rs)).iloc[-1]

        return {
            "pair": pair_key,
            "price": round(price, 5),
            "change_1d_pct": round(((price / prev_1d) - 1) * 100, 3)
            if prev_1d > 0
            else 0.0,
            "change_5d_pct": round(((price / prev_5d) - 1) * 100, 3)
            if prev_5d > 0
            else 0.0,
            "rsi14": round(float(rsi), 1) if pd.notna(rsi) else 50.0,
        }
    except Exception:
        return {}


def get_all_fx_pairs() -> list:
    """Return all registered FX pair display names."""
    return list(FX_PAIRS.keys())


def get_core_fx_pairs() -> list:
    """Return core FX pair display names used in quick scans."""
    return list(CORE_PAIRS)
