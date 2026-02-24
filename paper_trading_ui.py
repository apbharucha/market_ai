"""
Octavian Paper Trading UI — Live Charts, Multi-Asset Support, Chart Customization
Supports stocks, forex (USD/JPY, EUR/USD), futures (ES=F), crypto (BTC-USD).
No emojis. Live fragment-based chart updates without full page reload.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
from typing import Optional, Dict, List

from data_sources import get_stock
from trade_signal_overlay import generate_model_trades, add_trade_markers_to_fig, get_trade_summary

try:
    from data_sources import get_realtime_price as _ds_get_realtime_price
    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False
    _ds_get_realtime_price = None

try:
    from data_sources import get_fx, get_futures_proxy
    HAS_MULTI_ASSET = True
except ImportError:
    HAS_MULTI_ASSET = False

try:
    from octavian_theme import COLORS
except ImportError:
    COLORS = {"gold": "#D4AF37", "lavender": "#C4B5E0", "navy": "#0d1117",
              "navy_light": "#161b22", "white_soft": "#e6e6e6",
              "text_primary": "#ffffff", "text_secondary": "#8b949e",
              "border": "#30363d", "danger": "#f85149"}

_DEFAULT_CAPITAL = 100000.0

# 
# CHART TYPE CONSTANTS
# 

CHART_TYPES = {
    "Candlestick": "candlestick",
    "Heikin-Ashi": "heikin_ashi",
    "Line": "line",
    "Area": "area",
    "OHLC Bar": "ohlc",
}

AVAILABLE_INDICATORS = {
    "SMA 20": "sma20",
    "SMA 50": "sma50",
    "SMA 200": "sma200",
    "EMA 12": "ema12",
    "EMA 26": "ema26",
    "Bollinger Bands": "bbands",
    "VWAP": "vwap",
    "RSI (14)": "rsi",
    "MACD": "macd",
    "Volume": "volume",
    "ATR (14)": "atr",
    "Stochastic": "stoch",
}

INDICATOR_COLORS = {
    "sma20": "#42a5f5",
    "sma50": "#ef5350",
    "sma200": "#ff9800",
    "ema12": "#26c6da",
    "ema26": "#ab47bc",
    "bbands_mid": "#78909c",
    "bbands_upper": "#78909c",
    "bbands_lower": "#78909c",
    "vwap": "#ffeb3b",
    "rsi": "#ce93d8",
    "macd_line": "#42a5f5",
    "macd_signal": "#ef5350",
    "macd_hist_pos": "#26a69a",
    "macd_hist_neg": "#ef5350",
    "atr": "#ff7043",
    "stoch_k": "#42a5f5",
    "stoch_d": "#ef5350",
}


# 
# MULTI-ASSET DATA FETCHING
# 

def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns and ensure timezone-aware datetime index."""
    if df is None or df.empty:
        return df
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # Ensure datetime index is timezone-aware (use UTC for consistency)
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
    
    # Remove duplicate columns
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _detect_asset_type(symbol: str) -> str:
    """Detect the asset type from symbol format."""
    sym = symbol.strip().upper()
    if '/' in sym and '=' not in sym:
        return "forex"
    if sym.endswith('=X'):
        return "forex"
    if '=F' in sym:
        return "futures"
    if sym.endswith('-USD') and len(sym) <= 10:
        return "crypto"
    return "stock"


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_data(symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch data for any asset type — stocks, forex, futures, crypto. Cached 60s."""
    sym = symbol.strip().upper()
    asset_type = _detect_asset_type(sym)

    df = None

    if asset_type == "forex":
        # Handle EUR/USD, USD/JPY, EURUSD=X formats
        if '/' in sym:
            pair = sym.replace('/', '')
            yf_sym = pair + '=X'
            fx_key = sym.replace('/', '_')
        elif sym.endswith('=X'):
            yf_sym = sym
            base = sym.replace('=X', '')
            fx_key = base[:3] + '_' + base[3:] if len(base) == 6 else base
        else:
            yf_sym = sym
            fx_key = sym

        # For intraday intervals, go straight to yfinance (get_fx only has daily)
        if interval != "1d" and interval not in ("1wk", "1mo", "3mo"):
            try:
                import yfinance as yf
                df = yf.Ticker(yf_sym).history(period=period, interval=interval)
                if df is not None and not df.empty:
                    return _flatten_df(df)
            except Exception:
                pass

        # Try data_sources.get_fx first (daily only)
        if interval in ("1d", "1wk", "1mo", "3mo") and HAS_MULTI_ASSET:
            for key in [fx_key, fx_key.upper(), fx_key.lower()]:
                try:
                    df = get_fx(key)
                    if df is not None and not df.empty:
                        return _flatten_df(df)
                except Exception:
                    pass

        # Fallback to yfinance via get_stock or direct
        df = get_stock(yf_sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return _flatten_df(df)

        # Direct yfinance fallback
        try:
            import yfinance as yf
            df = yf.Ticker(yf_sym).history(period=period, interval=interval)
            if df is not None and not df.empty:
                return _flatten_df(df)
        except Exception:
            pass

        return None

    elif asset_type == "futures":
        if HAS_MULTI_ASSET:
            try:
                df = get_futures_proxy(sym, period=period, interval=interval)
                if df is not None and not df.empty:
                    return _flatten_df(df)
            except Exception:
                pass
        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return _flatten_df(df)
        return None

    else:
        # Stock or crypto
        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return _flatten_df(df)
        try:
            import yfinance as yf
            df = yf.Ticker(sym).history(period=period, interval=interval)
            if df is not None and not df.empty:
                return _flatten_df(df)
        except Exception:
            pass
        return None


def _get_live_price(symbol: str) -> float:
    """Get the most accurate current live price for any asset type."""
    sym = symbol.strip().upper()
    
    # Use centralized function with multiple fallbacks (includes Polygon)
    if HAS_REALTIME and _ds_get_realtime_price:
        try:
            price, _ = _ds_get_realtime_price(sym)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
    
    # Fallback to direct yfinance
    import yfinance as yf
    
    # Normalize forex symbols for yfinance
    if '/' in sym and '=' not in sym:
        yf_sym = sym.replace('/', '') + '=X'
    else:
        yf_sym = sym

    try:
        tk = yf.Ticker(yf_sym)
        
        # Method 1: Try intraday 1-minute data for most accurate price
        try:
            df_1m = tk.history(period="1d", interval="1m")
            if df_1m is not None and not df_1m.empty:
                if isinstance(df_1m.columns, pd.MultiIndex):
                    df_1m.columns = df_1m.columns.get_level_values(0)
                if "Close" in df_1m.columns:
                    close_col = df_1m["Close"]
                    if isinstance(close_col, pd.DataFrame):
                        close_col = close_col.iloc[:, 0]
                    close_col = close_col.dropna()
                    if len(close_col) > 0:
                        return float(close_col.iloc[-1])
        except Exception:
            pass
        
        # Method 2: Try fast_info
        try:
            info = tk.fast_info
            price = getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', None)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        
        # Method 3: Fallback to recent daily data
        df = tk.history(period="5d")
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" in df.columns:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna().astype(float)
                if len(close) > 0:
                    return float(close.iloc[-1])
    except Exception:
        pass

    # Final fallback: use _fetch_data
    df = _fetch_data(sym, period="5d")
    if df is not None and not df.empty and "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)
        if len(close) > 0:
            return float(close.iloc[-1])

    return 0.0


def _safe_close(df):
    """Extract close series safely."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)


def _safe_col(df, col):
    """Extract a column safely."""
    if df is None or col not in df.columns:
        return None
    c = df[col]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)


# 
# HEIKIN-ASHI CALCULATION
# 

def _compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLC data to Heikin-Ashi candles."""
    ha = pd.DataFrame(index=df.index)
    open_s = _safe_col(df, "Open")
    high_s = _safe_col(df, "High")
    low_s = _safe_col(df, "Low")
    close_s = _safe_col(df, "Close")
    if open_s is None or close_s is None:
        return df

    ha_close = (open_s + high_s + low_s + close_s) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (float(open_s.iloc[0]) + float(close_s.iloc[0])) / 2
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    ha["Open"] = ha_open
    ha["Close"] = ha_close
    ha["High"] = pd.concat([high_s, ha_open, ha_close], axis=1).max(axis=1)
    ha["Low"] = pd.concat([low_s, ha_open, ha_close], axis=1).min(axis=1)
    if "Volume" in df.columns:
        ha["Volume"] = df["Volume"]
    return ha


# 
# INDICATOR CALCULATIONS
# 

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _calc_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_bbands(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def _calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    high = _safe_col(df, "High")
    low = _safe_col(df, "Low")
    close = _safe_close(df)
    if high is None or low is None or close is None:
        return None
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    high = _safe_col(df, "High")
    low = _safe_col(df, "Low")
    close = _safe_close(df)
    if high is None or low is None or close is None:
        return None, None
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def _calc_vwap(df: pd.DataFrame) -> Optional[pd.Series]:
    close = _safe_close(df)
    vol = _safe_col(df, "Volume")
    high = _safe_col(df, "High")
    low = _safe_col(df, "Low")
    if close is None or vol is None or high is None or low is None:
        return None
    if vol.sum() == 0 or vol.max() < 1:
        return None
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    return cum_tp_vol / (cum_vol + 1e-10)


# 
# PORTFOLIO MANAGEMENT
# 

def _get_portfolio() -> dict:
    if "pt_portfolio" not in st.session_state:
        st.session_state["pt_portfolio"] = {
            "cash": _DEFAULT_CAPITAL,
            "positions": {},
            "options": [],
            "history": [],
            "initial_capital": _DEFAULT_CAPITAL,
        }
    return st.session_state["pt_portfolio"]


# 
# OPTIONS PRICING
# 

def _estimate_option_price(underlying: float, strike: float, side: str,
                           dte: int = 30, vol: float = 0.30) -> float:
    T = max(dte / 365.0, 0.001)
    r = 0.05
    sigma = max(vol, 0.10)
    d1 = (math.log(underlying / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    def _ncdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if side == "CALL":
        premium = underlying * _ncdf(d1) - strike * math.exp(-r * T) * _ncdf(d2)
    else:
        premium = strike * math.exp(-r * T) * _ncdf(-d2) - underlying * _ncdf(-d1)
    return max(0.01, round(premium, 2))


def _generate_strikes(price: float, n: int = 11) -> list:
    if price <= 0:
        return [100.0]
    if price < 25:
        interval = 1.0
    elif price < 100:
        interval = 2.5
    elif price < 250:
        interval = 5.0
    elif price < 500:
        interval = 10.0
    else:
        interval = 25.0
    base = round(price / interval) * interval
    half = n // 2
    return [s for s in [base + (i - half) * interval for i in range(n)] if s > 0]


# 
# CHART CUSTOMIZATION PANEL
# 

TIMEFRAME_OPTIONS = {
    "1 Day (1m)": {"period": "1d", "interval": "1m"},
    "5 Days (5m)": {"period": "5d", "interval": "5m"},
    "1 Month (30m)": {"period": "1mo", "interval": "30m"},
    "3 Months (Daily)": {"period": "3mo", "interval": "1d"},
    "6 Months (Daily)": {"period": "6mo", "interval": "1d"},
    "1 Year (Daily)": {"period": "1y", "interval": "1d"},
    "5 Years (Weekly)": {"period": "5y", "interval": "1wk"},
    "All Time": {"period": "max", "interval": "1wk"},
}


def _render_chart_controls():
    """Render chart customization controls. Returns settings dict."""
    with st.expander("Chart Settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            chart_type = st.selectbox(
                "Chart Type", list(CHART_TYPES.keys()),
                index=0, key="pt_chart_type",
            )
            tf_label = st.selectbox(
                "Timeframe", list(TIMEFRAME_OPTIONS.keys()),
                index=6, key="pt_chart_tf",
            )
            tf_config = TIMEFRAME_OPTIONS[tf_label]
            timeframe = tf_config["period"]
            interval = tf_config["interval"]
        with c2:
            show_trades = st.checkbox("Show Trade History on Chart", value=True, key="pt_show_trades")
            show_positions = st.checkbox("Show Open Positions", value=True, key="pt_show_positions")
            show_model_trades = st.checkbox("Show Hypothetical Model Trades", value=False, key="pt_show_model_trades")
            if show_model_trades:
                model_strategy = st.selectbox(
                    "Strategy", ["combined", "rsi", "sma_cross", "macd"],
                    index=0, key="pt_model_strategy",
                    format_func=lambda x: {"combined": "Combined (RSI+SMA+MACD)",
                                           "rsi": "RSI Crossover",
                                           "sma_cross": "SMA 20/50 Cross",
                                           "macd": "MACD Cross"}[x],
                )
            else:
                model_strategy = "combined"

        st.markdown("**Indicators**")
        # Group indicators into overlay vs panel
        overlay_indicators = ["SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26", "Bollinger Bands", "VWAP"]
        panel_indicators = ["RSI (14)", "MACD", "Volume", "ATR (14)", "Stochastic"]

        ic1, ic2 = st.columns(2)
        with ic1:
            st.caption("Overlay (on price)")
            selected_overlay = []
            for ind in overlay_indicators:
                if st.checkbox(ind, value=(ind in ["SMA 20", "SMA 50"]), key=f"pt_ind_{ind}"):
                    selected_overlay.append(AVAILABLE_INDICATORS[ind])
        with ic2:
            st.caption("Panel (below price)")
            selected_panel = []
            for ind in panel_indicators:
                default_on = ind == "Volume"
                if st.checkbox(ind, value=default_on, key=f"pt_ind_{ind}"):
                    selected_panel.append(AVAILABLE_INDICATORS[ind])

    return {
        "chart_type": CHART_TYPES[chart_type],
        "timeframe": timeframe,
        "interval": interval,
        "tf_label": tf_label,
        "overlay_indicators": selected_overlay,
        "panel_indicators": selected_panel,
        "show_trades": show_trades,
        "show_positions": show_positions,
        "show_model_trades": show_model_trades,
        "model_strategy": model_strategy,
    }


# 
# LIVE CHART (fragment-based, customizable)
# 

def _render_live_chart(symbol: str, portfolio: dict, chart_settings: dict):
    """Render chart inside a fragment container for live updates."""

    @st.fragment(run_every=60)
    def _chart_fragment():
        tf = chart_settings.get("timeframe", "6mo")
        iv = chart_settings.get("interval", "1d")
        
        # Override for specific keys if needed, but dict lookup should work
        # Map nice labels to actual period/interval if passed as label
        if tf in TIMEFRAME_OPTIONS:
            cfg = TIMEFRAME_OPTIONS[tf]
            tf = cfg["period"]
            iv = cfg["interval"]

        df = _fetch_data(symbol, period=tf, interval=iv)
        if df is None or df.empty:
            st.warning(f"No chart data available for {symbol}. Check the symbol format.")
            st.caption("Supported: AAPL, EUR/USD, USDJPY=X, ES=F, BTC-USD")
            return

        close = _safe_close(df)
        if close is None or len(close) < 5:
            st.warning(f"Insufficient data for {symbol}.")
            return

        # Get FRESH live price for real-time display
        live_price = _get_live_price(symbol)
        
        # Get current time for synchronization
        current_time = pd.Timestamp.now(tz='UTC')
        local_time = pd.Timestamp.now()
        
        # Use live price if available and more recent than last candle
        current = live_price if live_price > 0 else float(close.iloc[-1])

        ct = chart_settings.get("chart_type", "candlestick")
        overlay_inds = chart_settings.get("overlay_indicators", [])
        panel_inds = chart_settings.get("panel_indicators", [])

        # Determine subplot layout based on selected panel indicators
        panel_count = 0
        panel_map = {}  # indicator -> row number
        if "volume" in panel_inds:
            panel_count += 1
            panel_map["volume"] = 1 + panel_count
        if "rsi" in panel_inds:
            panel_count += 1
            panel_map["rsi"] = 1 + panel_count
        if "macd" in panel_inds:
            panel_count += 1
            panel_map["macd"] = 1 + panel_count
        if "atr" in panel_inds:
            panel_count += 1
            panel_map["atr"] = 1 + panel_count
        if "stoch" in panel_inds:
            panel_count += 1
            panel_map["stoch"] = 1 + panel_count

        total_rows = 1 + panel_count
        row_heights = [0.55] + [round(0.45 / max(panel_count, 1), 2)] * panel_count if panel_count > 0 else [1.0]

        fig = make_subplots(
            rows=total_rows, cols=1, shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.025,
        )

        #  Price chart (row 1) 
        open_s = _safe_col(df, "Open")
        high_s = _safe_col(df, "High")
        low_s = _safe_col(df, "Low")
        has_ohlc = open_s is not None and high_s is not None and low_s is not None

        if ct == "heikin_ashi" and has_ohlc:
            ha_df = _compute_heikin_ashi(df)
            ha_open = _safe_col(ha_df, "Open")
            ha_high = _safe_col(ha_df, "High")
            ha_low = _safe_col(ha_df, "Low")
            ha_close = _safe_col(ha_df, "Close")
            if ha_open is not None and ha_close is not None:
                fig.add_trace(go.Candlestick(
                    x=ha_df.index, open=ha_open, high=ha_high, low=ha_low, close=ha_close,
                    name="Heikin-Ashi", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Price",
                                         line=dict(color="#26a69a", width=2)), row=1, col=1)
        elif ct == "candlestick" and has_ohlc:
            fig.add_trace(go.Candlestick(
                x=df.index, open=open_s, high=high_s, low=low_s, close=close,
                name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            ), row=1, col=1)
        elif ct == "ohlc" and has_ohlc:
            fig.add_trace(go.Ohlc(
                x=df.index, open=open_s, high=high_s, low=low_s, close=close,
                name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            ), row=1, col=1)
        elif ct == "area":
            fig.add_trace(go.Scatter(
                x=close.index, y=close.values, mode="lines", name="Price",
                fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
                line=dict(color="#26a69a", width=2),
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=close.index, y=close.values, mode="lines", name="Price",
                line=dict(color="#26a69a", width=2),
            ), row=1, col=1)

        #  Overlay indicators 
        if "sma20" in overlay_inds and len(close) >= 20:
            sma = close.rolling(20).mean()
            fig.add_trace(go.Scatter(x=close.index, y=sma, mode="lines", name="SMA 20",
                                     line=dict(color=INDICATOR_COLORS["sma20"], width=1.2)), row=1, col=1)
        if "sma50" in overlay_inds and len(close) >= 50:
            sma = close.rolling(50).mean()
            fig.add_trace(go.Scatter(x=close.index, y=sma, mode="lines", name="SMA 50",
                                     line=dict(color=INDICATOR_COLORS["sma50"], width=1.2)), row=1, col=1)
        if "sma200" in overlay_inds and len(close) >= 200:
            sma = close.rolling(200).mean()
            fig.add_trace(go.Scatter(x=close.index, y=sma, mode="lines", name="SMA 200",
                                     line=dict(color=INDICATOR_COLORS["sma200"], width=1.2)), row=1, col=1)
        if "ema12" in overlay_inds and len(close) >= 12:
            ema = close.ewm(span=12, adjust=False).mean()
            fig.add_trace(go.Scatter(x=close.index, y=ema, mode="lines", name="EMA 12",
                                     line=dict(color=INDICATOR_COLORS["ema12"], width=1.2)), row=1, col=1)
        if "ema26" in overlay_inds and len(close) >= 26:
            ema = close.ewm(span=26, adjust=False).mean()
            fig.add_trace(go.Scatter(x=close.index, y=ema, mode="lines", name="EMA 26",
                                     line=dict(color=INDICATOR_COLORS["ema26"], width=1.2)), row=1, col=1)
        if "bbands" in overlay_inds and len(close) >= 20:
            bb_upper, bb_mid, bb_lower = _calc_bbands(close)
            fig.add_trace(go.Scatter(x=close.index, y=bb_upper, mode="lines", name="BB Upper",
                                     line=dict(color=INDICATOR_COLORS["bbands_upper"], width=0.8, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=close.index, y=bb_mid, mode="lines", name="BB Mid",
                                     line=dict(color=INDICATOR_COLORS["bbands_mid"], width=0.8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=close.index, y=bb_lower, mode="lines", name="BB Lower",
                                     fill="tonexty", fillcolor="rgba(120,144,156,0.08)",
                                     line=dict(color=INDICATOR_COLORS["bbands_lower"], width=0.8, dash="dot")), row=1, col=1)
        if "vwap" in overlay_inds:
            vwap = _calc_vwap(df)
            if vwap is not None:
                fig.add_trace(go.Scatter(x=vwap.index, y=vwap, mode="lines", name="VWAP",
                                         line=dict(color=INDICATOR_COLORS["vwap"], width=1.3, dash="dash")), row=1, col=1)

        #  Position markers 
        if chart_settings.get("show_positions", True):
            pos = portfolio.get("positions", {}).get(symbol)
            if pos:
                entry = pos.get("avg_price", 0)
                tp = pos.get("take_profit")
                sl = pos.get("stop_loss")
                if entry > 0:
                    fig.add_hline(y=entry, line_dash="dash", line_color="cyan",
                                  annotation_text=f"Entry ${entry:,.4f}", row=1, col=1)
                if tp and tp > 0:
                    fig.add_hline(y=tp, line_dash="dot", line_color="lime",
                                  annotation_text=f"TP ${tp:,.4f}", row=1, col=1)
                if sl and sl > 0:
                    fig.add_hline(y=sl, line_dash="dot", line_color="red",
                                  annotation_text=f"SL ${sl:,.4f}", row=1, col=1)

        #  Trade history markers 
        if chart_settings.get("show_trades", True):
            history = portfolio.get("history", [])
            buy_dates, buy_prices = [], []
            sell_dates, sell_prices = [], []
            for trade in history:
                if trade.get("Symbol") != symbol:
                    continue
                try:
                    t_time = pd.Timestamp(trade.get("Time", ""))
                    t_price_str = str(trade.get("Price", "0")).replace("$", "").replace(",", "")
                    t_price = float(t_price_str)
                    side = trade.get("Side", "")
                    if t_price > 0 and pd.notna(t_time):
                        if "BUY" in side or "LONG" in side or "COVER" in side:
                            buy_dates.append(t_time)
                            buy_prices.append(t_price)
                        elif "SELL" in side or "SHORT" in side or "CLOSE" in side:
                            sell_dates.append(t_time)
                            sell_prices.append(t_price)
                except Exception:
                    pass
            if buy_dates:
                fig.add_trace(go.Scatter(
                    x=buy_dates, y=buy_prices, mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(width=1, color="white")),
                ), row=1, col=1)
            if sell_dates:
                fig.add_trace(go.Scatter(
                    x=sell_dates, y=sell_prices, mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=1, color="white")),
                ), row=1, col=1)

        #  Model trade signals (hypothetical) 
        model_markers = []
        if chart_settings.get("show_model_trades", False) and close is not None and len(close) >= 30:
            strategy = chart_settings.get("model_strategy", "combined")
            model_markers = generate_model_trades(close, df, strategy)
            if model_markers:
                add_trade_markers_to_fig(fig, model_markers, row=1, col=1)

        #  Panel indicators 
        vol_s = _safe_col(df, "Volume")
        has_volume = vol_s is not None and len(vol_s) > 0 and vol_s.sum() > 0 and vol_s.max() > 1

        if "volume" in panel_map and has_volume:
            r = panel_map["volume"]
            if has_ohlc:
                colors = ["#26a69a" if c >= o else "#ef5350"
                          for c, o in zip(close.values[-len(vol_s):], open_s.values[-len(vol_s):])]
            else:
                colors = "#42a5f5"
            fig.add_trace(go.Bar(x=vol_s.index, y=vol_s.values, name="Volume",
                                 marker_color=colors, opacity=0.5), row=r, col=1)
            fig.update_yaxes(title_text="Vol", row=r, col=1)

        if "rsi" in panel_map and len(close) >= 14:
            r = panel_map["rsi"]
            rsi = _calc_rsi(close)
            fig.add_trace(go.Scatter(x=close.index, y=rsi, mode="lines", name="RSI",
                                     line=dict(color=INDICATOR_COLORS["rsi"], width=1.5)), row=r, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", line_width=0.8, row=r, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", line_width=0.8, row=r, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)

        if "macd" in panel_map and len(close) >= 27:
            r = panel_map["macd"]
            macd_l, sig_l, hist = _calc_macd(close)
            fig.add_trace(go.Scatter(x=close.index, y=macd_l, mode="lines", name="MACD",
                                     line=dict(color=INDICATOR_COLORS["macd_line"], width=1.2)), row=r, col=1)
            fig.add_trace(go.Scatter(x=close.index, y=sig_l, mode="lines", name="Signal",
                                     line=dict(color=INDICATOR_COLORS["macd_signal"], width=1.2)), row=r, col=1)
            hist_colors = [INDICATOR_COLORS["macd_hist_pos"] if v >= 0 else INDICATOR_COLORS["macd_hist_neg"]
                           for v in hist.values]
            fig.add_trace(go.Bar(x=close.index, y=hist.values, name="Histogram",
                                 marker_color=hist_colors, opacity=0.6), row=r, col=1)
            fig.update_yaxes(title_text="MACD", row=r, col=1)

        if "atr" in panel_map:
            r = panel_map["atr"]
            atr = _calc_atr(df)
            if atr is not None:
                fig.add_trace(go.Scatter(x=atr.index, y=atr, mode="lines", name="ATR",
                                         line=dict(color=INDICATOR_COLORS["atr"], width=1.2)), row=r, col=1)
                fig.update_yaxes(title_text="ATR", row=r, col=1)

        if "stoch" in panel_map:
            r = panel_map["stoch"]
            k, d = _calc_stochastic(df)
            if k is not None:
                fig.add_trace(go.Scatter(x=k.index, y=k, mode="lines", name="%K",
                                         line=dict(color=INDICATOR_COLORS["stoch_k"], width=1.2)), row=r, col=1)
                fig.add_trace(go.Scatter(x=d.index, y=d, mode="lines", name="%D",
                                         line=dict(color=INDICATOR_COLORS["stoch_d"], width=1.2)), row=r, col=1)
                fig.add_hline(y=80, line_dash="dot", line_color="red", line_width=0.8, row=r, col=1)
                fig.add_hline(y=20, line_dash="dot", line_color="green", line_width=0.8, row=r, col=1)
                fig.update_yaxes(title_text="Stoch", range=[0, 100], row=r, col=1)

        #  Layout 
        # Use live_price (current) from earlier for display
        asset_type = _detect_asset_type(symbol)
        price_fmt = f"${current:,.4f}" if asset_type == "forex" else f"${current:,.2f}"
        chart_label = ct.replace("_", "-").title()
        chart_height = 380 + panel_count * 120
        
        # Add live price marker on chart
        fig.add_hline(
            y=current, line_dash="dash", line_color="#00ff88", line_width=1.5,
            annotation_text=f"LIVE: {price_fmt}",
            annotation_position="right",
            annotation_font_color="#00ff88",
            row=1, col=1
        )

        fig.update_layout(
            title=dict(
                text=f"{symbol}  --  {chart_label}  |  LIVE: {price_fmt}  |  {local_time.strftime('%H:%M:%S')}",
                x=0.01, font=dict(size=13, color="white"),
                pad=dict(t=5, b=5),
            ),
            height=chart_height,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=9),
            ),
            xaxis=dict(
                type='date',
                showgrid=True,
                gridcolor='#30363d',
                gridwidth=1,
            ),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)

        # Use timestamp in key to force re-render on each fragment update
        chart_key = f"pt_chart_{symbol}_{tf}_{ct}_{local_time.strftime('%H%M%S')}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # Live price bar - use true previous trading day close from fast_info
        prev_close = None
        if HAS_REALTIME and _ds_get_realtime_price:
            try:
                _, _prev = _ds_get_realtime_price(symbol)
                if _prev and _prev > 0:
                    prev_close = float(_prev)
            except Exception:
                pass
        if prev_close is None:
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
        change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0
        period_return = ((current / float(close.iloc[0])) - 1) * 100 if float(close.iloc[0]) > 0 else 0
        high_s_last = _safe_col(df, "High")
        low_s_last = _safe_col(df, "Low")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Live Price", price_fmt, f"{change_pct:+.2f}%")
        with mc2:
            st.metric("Day High", f"${float(high_s_last.iloc[-1]):,.4f}" if high_s_last is not None and len(high_s_last) > 0 else "--")
        with mc3:
            st.metric("Day Low", f"${float(low_s_last.iloc[-1]):,.4f}" if low_s_last is not None and len(low_s_last) > 0 else "--")
        with mc4:
            tf_lbl = chart_settings.get("tf_label", tf)
            st.metric(f"Return ({tf_lbl})", f"{period_return:+.2f}%")

        #  Model trade summary (when enabled) 
        if chart_settings.get("show_model_trades", False) and model_markers:
            summary = get_trade_summary(model_markers)
            if summary["total_trades"] > 0:
                st.markdown("---")
                st.markdown("**Hypothetical Model Trade Summary**")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                with sc1: st.metric("Trades", summary.get("total_trades", 0))
                with sc2: st.metric("Win Rate", f"{summary.get('win_rate', 0):.0%}")
                with sc3: st.metric("Avg Win", f"{summary.get('avg_win', 0):+.2%}")
                with sc4: st.metric("Avg Loss", f"{summary.get('avg_loss', 0):+.2%}")
                with sc5: st.metric("Total Return", f"{summary.get('total_return', 0):+.2%}")
                sc6, sc7, sc8 = st.columns(3)
                with sc6: st.metric("Best Trade", f"{summary.get('best_trade', 0):+.2%}")
                with sc7: st.metric("Worst Trade", f"{summary.get('worst_trade', 0):+.2%}")
                with sc8:
                    pf = summary["profit_factor"]
                    st.metric("Profit Factor", f"{pf:.2f}" if pf < 100 else "Inf")
                with st.expander("Trade Log"):
                    log_rows = []
                    for m in model_markers:
                        row_data = {
                            "Date": str(m.timestamp)[:10],
                            "Side": m.side,
                            "Price": f"${m.price:,.2f}",
                            "Reason": m.reason,
                            "P&L": f"{m.pnl_pct:+.2%} (${m.pnl_dollar:+,.2f})" if m.pnl_pct is not None else "--",
                        }
                        log_rows.append(row_data)
                    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

    _chart_fragment()


# 
# TRADE EXECUTION (unchanged logic)
# 

def _execute_equity_trade(portfolio, symbol, side, qty, price, tp, sl):
    cost = qty * price
    if side == "BUY":
        if cost > portfolio["cash"]:
            st.error(f"Insufficient cash. Need ${cost:,.2f}, have ${portfolio['cash']:,.2f}.")
            return False
        portfolio["cash"] -= cost
        if symbol in portfolio["positions"]:
            pos = portfolio["positions"][symbol]
            total_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / total_qty
            pos["qty"] = total_qty
        else:
            portfolio["positions"][symbol] = {"qty": qty, "avg_price": price, "side": "LONG", "take_profit": tp, "stop_loss": sl}
        if tp: portfolio["positions"][symbol]["take_profit"] = tp
        if sl: portfolio["positions"][symbol]["stop_loss"] = sl
    elif side == "SELL":
        if symbol not in portfolio["positions"] or portfolio["positions"][symbol]["qty"] < qty:
            st.error("Insufficient shares to sell.")
            return False
        pos = portfolio["positions"][symbol]
        portfolio["cash"] += cost
        pos["qty"] -= qty
        if pos["qty"] <= 0:
            del portfolio["positions"][symbol]
    elif side == "SHORT":
        portfolio["cash"] += cost
        if symbol in portfolio["positions"] and portfolio["positions"][symbol].get("side") == "SHORT":
            pos = portfolio["positions"][symbol]
            total_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / total_qty
            pos["qty"] = total_qty
        else:
            portfolio["positions"][symbol] = {"qty": qty, "avg_price": price, "side": "SHORT", "take_profit": tp, "stop_loss": sl}
        if tp: portfolio["positions"][symbol]["take_profit"] = tp
        if sl: portfolio["positions"][symbol]["stop_loss"] = sl
    portfolio["history"].append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol, "Side": side, "Qty": qty,
        "Price": f"${price:,.4f}", "Cost": f"${cost:,.2f}",
        "TP": f"${tp:,.4f}" if tp else "--", "SL": f"${sl:,.4f}" if sl else "--",
        "Type": "Equity",
    })
    st.success(f"{side} {qty}x {symbol} @ ${price:,.4f}")
    return True


def _close_position(portfolio, symbol):
    if symbol not in portfolio["positions"]:
        st.error(f"No open position for {symbol}.")
        return
    pos = portfolio["positions"][symbol]
    qty = pos["qty"]; entry = pos["avg_price"]; side = pos.get("side", "LONG")
    current_price = _get_live_price(symbol)
    if current_price <= 0:
        st.error(f"Cannot fetch current price for {symbol}.")
        return
    if side == "LONG":
        pnl = (current_price - entry) * qty; portfolio["cash"] += qty * current_price
    else:
        pnl = (entry - current_price) * qty; portfolio["cash"] -= qty * current_price
    del portfolio["positions"][symbol]
    portfolio["history"].append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol, "Side": f"CLOSE {side}", "Qty": qty,
        "Price": f"${current_price:,.4f}", "Cost": f"${qty * current_price:,.2f}",
        "TP": "--", "SL": "--", "Type": "Equity", "P&L": f"${pnl:+,.2f}",
    })
    st.success(f"Closed {side} {qty}x {symbol} @ ${current_price:,.4f} | P&L: ${pnl:+,.2f}")


def _execute_option_trade(portfolio, symbol, opt_type, action, strike, premium, contracts, dte, tp_pct, sl_pct):
    total_cost = premium * 100 * contracts
    if action == "BUY":
        if total_cost > portfolio["cash"]:
            st.error(f"Insufficient cash. Need ${total_cost:,.2f}.")
            return False
        portfolio["cash"] -= total_cost
    else:
        margin = strike * 100 * contracts * 0.20
        if margin > portfolio["cash"]:
            st.error(f"Insufficient margin. Need ${margin:,.2f}.")
            return False
        portfolio["cash"] += total_cost
    portfolio["options"].append({
        "symbol": symbol, "option_type": opt_type, "action": action,
        "strike": strike, "premium": premium, "contracts": contracts,
        "total_cost": total_cost, "dte_remaining": dte,
        "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tp_pct": tp_pct, "sl_pct": sl_pct,
    })
    portfolio["history"].append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol, "Side": f"{action} {opt_type}",
        "Qty": f"{contracts}c", "Price": f"${premium:,.2f}",
        "Cost": f"${total_cost:,.2f}",
        "TP": f"{tp_pct}%" if tp_pct else "--", "SL": f"{sl_pct}%" if sl_pct else "--",
        "Type": f"Option ({opt_type} ${strike:,.0f})",
    })
    st.success(f"{action} {contracts}x {symbol} {opt_type} ${strike:,.2f} @ ${premium:,.2f}")
    return True


def _close_option(portfolio, idx, underlying_price):
    if idx < 0 or idx >= len(portfolio["options"]):
        st.error("Invalid option index.")
        return
    opt = portfolio["options"][idx]
    cur_prem = _estimate_option_price(underlying_price, opt["strike"], opt["option_type"],
                                       max(opt.get("dte_remaining", 1) - 1, 1))
    close_val = cur_prem * 100 * opt["contracts"]
    if opt["action"] == "BUY":
        portfolio["cash"] += close_val; pnl = close_val - opt["total_cost"]
    else:
        portfolio["cash"] -= close_val; pnl = opt["total_cost"] - close_val
    portfolio["history"].append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": opt["symbol"], "Side": f"CLOSE {opt['action']} {opt['option_type']}",
        "Qty": f"{opt['contracts']}c", "Price": f"${cur_prem:,.2f}",
        "Cost": f"${close_val:,.2f}", "TP": "--", "SL": "--",
        "Type": f"Option ({opt['option_type']} ${opt['strike']:,.0f})",
        "P&L": f"${pnl:+,.2f}",
    })
    portfolio["options"].pop(idx)
    st.success(f"Closed {opt['option_type']} ${opt['strike']:,.2f} | P&L: ${pnl:+,.2f}")


# 
# MAIN UI
# 

def show_paper_trading():
    st.title("Paper Trading")

    portfolio = _get_portfolio()

 # Import paper trading engine for advanced features
    try:
        from paper_trading_engine import get_paper_engine
        paper_engine = get_paper_engine()
        has_engine = True
    except Exception as e:
        paper_engine = None
        has_engine = False

    symbol_raw = st.text_input(
        "Symbol", value=st.session_state.get("pt_symbol_value", "AAPL"),
        key="pt_symbol_input",
        help="Stocks: AAPL | Forex: USD/JPY or USDJPY=X | Futures: ES=F | Crypto: BTC-USD",
    )
    symbol = symbol_raw.strip().upper()
    if not symbol:
        st.info("Enter a symbol to begin paper trading.")
        return
    # Use different key to avoid conflict with widget key
    st.session_state["pt_symbol_value"] = symbol

    asset_type = _detect_asset_type(symbol)
    st.caption(f"Asset type: {asset_type.upper()} | Symbol: {symbol}")

    # --- Always fetch live price directly for every render (no session_state caching) ---
    def get_live_price_fresh():
        return _get_live_price(symbol)

    live_price = get_live_price_fresh()
    if live_price <= 0:
        st.error(f"Could not fetch price for {symbol}. Check format: Forex use USD/JPY | Futures use ES=F | Crypto use BTC-USD")
        return

    # Chart customization panel
    chart_settings = _render_chart_controls()

    # Live chart (pass live_price for overlays if needed)
    _render_live_chart(symbol, portfolio, chart_settings)

    # Portfolio summary (use live price)
    @st.fragment(run_every=10)
    def _portfolio_summary():
        # Always fetch live price fresh for summary
        fresh_live_price = _get_live_price(symbol)
        total_equity = portfolio["cash"]
        for sym, pos in portfolio["positions"].items():
            p = _get_live_price(sym) if sym != symbol else fresh_live_price
            if p > 0:
                total_equity += pos["qty"] * p
        pnl = total_equity - portfolio["initial_capital"]
        pnl_pct = pnl / portfolio["initial_capital"] * 100
        price_fmt = f"${fresh_live_price:,.4f}" if asset_type == "forex" else f"${fresh_live_price:,.2f}"
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Cash", f"${portfolio['cash']:,.2f}")
        with c2: st.metric("Total Equity", f"{total_equity:,.2f}")
        with c3: st.metric("P&L", f"{pnl:+,.2f}", f"{pnl_pct:+.2f}%")
        with c4: st.metric("Market Price", price_fmt)
        # Show live price as a separate metric for clarity (but always fresh)
        st.caption(f"Live Price: {price_fmt} (auto-updating)")

    st.markdown("---")
    _portfolio_summary()

    # Trading tabs
    trade_tabs = st.tabs(["Equity Trade", "Options Trade", "Open Positions", "Trade History"])

    # Note: Performance Dashboard moved to Simulation Hub
    # Note: Breaking Trades moved to Dashboard

    with trade_tabs[0]:
        st.subheader(f"Trade {symbol}")
        price_fmt = f"${live_price:,.4f}" if asset_type == "forex" else f"${live_price:,.2f}"
        st.markdown(
            f'<div style="background:#1a1f2e;padding:10px 16px;border-radius:6px;margin-bottom:12px;">'
            f'<span style="color:#aaa;font-size:0.8rem;">Execution Price (Market/Live)</span><br>'
            f'<span style="color:white;font-size:1.4rem;font-weight:600;">{price_fmt}</span></div>',
            unsafe_allow_html=True)
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            trade_side = st.radio("Side", ["BUY", "SELL", "SHORT"], horizontal=True, key="pt_side")
        with c_t2:
            if "pt_qty" not in st.session_state:
                st.session_state["pt_qty"] = max(1, int(portfolio["cash"] * 0.10 / live_price)) if live_price > 0 else 1
            qty = st.number_input("Quantity", min_value=1, step=1, key="pt_qty")
        trade_cost = qty * live_price
        st.caption(f"Estimated cost: ${trade_cost:,.2f} | Available: ${portfolio['cash']:,.2f}")
        c_tp, c_sl = st.columns(2)
        with c_tp:
            enable_tp = st.checkbox("Set Take Profit", key="pt_en_tp")
            tp_price = None
            if enable_tp:
                tp_price = st.number_input("Take Profit", value=round(live_price * (1.05 if trade_side != "SHORT" else 0.95), 4),
                                           min_value=0.0001, step=0.5, format="%.4f", key="pt_tp")
        with c_sl:
            enable_sl = st.checkbox("Set Stop Loss", key="pt_en_sl")
            sl_price = None
            if enable_sl:
                sl_price = st.number_input("Stop Loss", value=round(live_price * (0.97 if trade_side != "SHORT" else 1.03), 4),
                                           min_value=0.0001, step=0.5, format="%.4f", key="pt_sl")
        if st.button(f"Execute {trade_side}", type="primary", key="pt_exec_eq", use_container_width=True):
            exec_price = get_live_price_fresh()
            if _execute_equity_trade(portfolio, symbol, trade_side, qty, exec_price, tp_price, sl_price):
                st.rerun()

    with trade_tabs[1]:
        st.subheader(f"Options -- {symbol}")
        if asset_type != "stock":
            st.warning("Options are primarily available for stock symbols.")
        price_fmt = f"${live_price:,.4f}" if asset_type == "forex" else f"${live_price:,.2f}"
        st.markdown(
            f'<div style="background:#1a1f2e;padding:10px 16px;border-radius:6px;margin-bottom:12px;">'
            f'<span style="color:#aaa;font-size:0.8rem;">Underlying Price</span><br>'
            f'<span style="color:white;font-size:1.4rem;font-weight:600;">{price_fmt}</span></div>',
            unsafe_allow_html=True)
        c_o1, c_o2, c_o3 = st.columns(3)
        with c_o1: opt_side = st.radio("Type", ["CALL", "PUT"], horizontal=True, key="pt_opt_side")
        with c_o2: opt_action = st.radio("Action", ["BUY", "SELL (Write)"], horizontal=True, key="pt_opt_act")
        with c_o3: dte = st.selectbox("Expiration (DTE)", [7, 14, 30, 45, 60, 90, 120, 180, 365], index=2, key="pt_opt_dte")
        strikes = _generate_strikes(live_price)
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - live_price))
        strike = st.select_slider("Strike Price", options=strikes, value=strikes[atm_idx],
                                  format_func=lambda x: f"${x:,.2f}", key="pt_opt_strike")
        ann_vol = 0.30
        df_vol = _fetch_data(symbol, period="3mo")
        if df_vol is not None:
            cs = _safe_close(df_vol)
            if cs is not None and len(cs) > 20:
                rets = cs.pct_change().dropna()
                if len(rets) > 10: ann_vol = float(rets.std() * np.sqrt(252))
        premium = _estimate_option_price(live_price, strike, opt_side, dte, ann_vol)
        moneyness = "ITM" if (opt_side == "CALL" and live_price > strike) or (opt_side == "PUT" and live_price < strike) else "OTM" if (opt_side == "CALL" and live_price < strike) or (opt_side == "PUT" and live_price > strike) else "ATM"
        st.markdown(
            f'<div style="background:#1a1f2e;padding:10px 16px;border-radius:6px;margin:8px 0;">'
            f'<span style="color:#aaa;">Est. Premium:</span> '
            f'<span style="color:white;font-weight:600;">${premium:,.2f}</span> per share '
            f'<span style="color:#aaa;">| {moneyness} | IV: {ann_vol:.0%} | DTE: {dte}</span></div>',
            unsafe_allow_html=True)
        c_oq1, c_oq2 = st.columns(2)
        with c_oq1:
            if "pt_opt_contracts" not in st.session_state: st.session_state["pt_opt_contracts"] = 1
            opt_contracts = st.number_input("Contracts (x100)", min_value=1, step=1, key="pt_opt_contracts")
        with c_oq2:
            total_prem = premium * 100 * opt_contracts
            st.markdown(f'<div style="padding:28px 0 0 0;"><span style="color:#aaa;">Total Cost:</span> '
                        f'<span style="color:white;font-weight:600;">${total_prem:,.2f}</span></div>', unsafe_allow_html=True)
        c_otp, c_osl = st.columns(2)
        with c_otp:
            opt_tp = st.checkbox("Take Profit (%)", key="pt_opt_en_tp")
            opt_tp_pct = st.number_input("TP %", value=50, min_value=10, max_value=500, step=10, key="pt_opt_tp_pct") if opt_tp else None
        with c_osl:
            opt_sl = st.checkbox("Stop Loss (%)", key="pt_opt_en_sl")
            opt_sl_pct = st.number_input("SL %", value=50, min_value=10, max_value=100, step=10, key="pt_opt_sl_pct") if opt_sl else None
        act_label = "Buy" if "BUY" in opt_action else "Sell"
        if st.button(f"{act_label} {opt_contracts} {opt_side} @ ${strike:,.2f}", type="primary", key="pt_exec_opt", use_container_width=True):
            action_str = "BUY" if "BUY" in opt_action else "SELL"
            exec_price = get_live_price_fresh()
            if _execute_option_trade(portfolio, symbol, opt_side, action_str, strike, premium, opt_contracts, dte, opt_tp_pct, opt_sl_pct):
                st.rerun()

    with trade_tabs[2]:
        st.subheader("Open Positions")
        if portfolio["positions"]:
            st.markdown("**Equity Positions**")
            rows = []
            for sym, pos in portfolio["positions"].items():
                p = _get_live_price(sym) if sym != symbol else live_price
                unrealized = (p - pos["avg_price"]) * pos["qty"]
                if pos.get("side") == "SHORT": unrealized = -unrealized
                rows.append({"Symbol": sym, "Side": pos.get("side", "LONG"), "Qty": pos["qty"],
                             "Avg Price": f"${pos['avg_price']:,.4f}", "Current": f"${p:,.4f}",
                             "P&L": f"${unrealized:+,.2f}",
                             "TP": f"${pos['take_profit']:,.4f}" if pos.get("take_profit") else "--",
                             "SL": f"${pos['stop_loss']:,.4f}" if pos.get("stop_loss") else "--"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            close_sym = st.selectbox("Close position", list(portfolio["positions"].keys()), key="pt_close_sym")
            if st.button(f"Close {close_sym}", key="pt_close_pos"):
                _close_position(portfolio, close_sym); st.rerun()
        else:
            st.caption("No open equity positions.")
        if portfolio.get("options"):
            st.markdown("**Options Positions**")
            opt_rows = []
            for i, opt in enumerate(portfolio["options"]):
                opt_rows.append({"#": i + 1, "Symbol": opt["symbol"], "Type": opt["option_type"],
                                 "Action": opt["action"], "Strike": f"${opt['strike']:,.2f}",
                                 "Premium": f"${opt['premium']:,.2f}", "Contracts": opt["contracts"],
                                 "DTE": opt.get("dte_remaining", "?"), "Cost": f"${opt['total_cost']:,.2f}"})
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
            close_idx = st.number_input("Close option #", min_value=1, max_value=len(portfolio["options"]), value=1, key="pt_close_oidx")
            if st.button("Close Option", key="pt_close_opt"):
                _close_option(portfolio, close_idx - 1, live_price); st.rerun()
        else:
            st.caption("No open options positions.")

    with trade_tabs[3]:
        st.subheader("Trade History")
        if portfolio["history"]:
            st.dataframe(pd.DataFrame(portfolio["history"]), use_container_width=True, hide_index=True, height=400)
            if st.button("Reset Portfolio", type="secondary", key="pt_reset"):
                st.session_state["pt_portfolio"] = {"cash": _DEFAULT_CAPITAL, "positions": {}, "options": [], "history": [], "initial_capital": _DEFAULT_CAPITAL}
                for k in list(st.session_state.keys()):
                    if k.startswith("pt_") and k != "pt_portfolio": del st.session_state[k]
                st.rerun()
        else:
            st.caption("No trades yet.")
