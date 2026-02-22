import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ml_analysis
from ml_analysis import get_analyzer
from indicators import add_indicators
from data_sources import get_stock, get_fx, get_futures_proxy

try:
    from data_sources import get_fresh_quote
    _HAS_FRESH = True
except ImportError:
    _HAS_FRESH = False

try:
    from data_sources import get_realtime_price as _ds_realtime
    _HAS_REALTIME = True
except ImportError:
    _HAS_REALTIME = False
    _ds_realtime = None

from regime import volatility_regime, risk_on_off
from trader_profile import show_trader_selection, get_trader_profile, get_recommendation_style
from database_manager import get_database_manager
import traceback
from trade_signal_overlay import generate_model_trades, add_trade_markers_to_fig, get_trade_summary

# ML libraries loaded lazily on first use
_ml_initialized = False
ML_AVAILABLE = False

def _ensure_ml():
    global _ml_initialized, ML_AVAILABLE
    if not _ml_initialized:
        ml_analysis.ensure_ml_libraries()
        ML_AVAILABLE = ml_analysis.SKLEARN_AVAILABLE and ml_analysis.XGBOOST_AVAILABLE
        _ml_initialized = True

# Optional imports
try:
    from options_analyzer import get_options_analyzer
    HAS_OPTIONS = True
except ImportError:
    HAS_OPTIONS = False

try:
    from graph_analysis import analyze_price_chart, analyze_rsi_chart, analyze_volume_chart
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI without external ta library."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_macd(close: pd.Series) -> tuple:
    """Compute MACD line, signal, histogram without external ta library."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _safe_col(df, col):
    """Safely extract a column that might be multi-level."""
    c = df[col]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c


@st.cache_data(ttl=300)
def fetch_market_data(symbol, asset_type, period, interval):
    """Cached data fetching to improve performance"""
    try:
        if asset_type == "Stock":
            df = get_stock(symbol, period=period, interval=interval)
        elif asset_type == "FX":
            # Handle different FX symbol formats
            fx_symbol = symbol.replace("=X", "").replace("-", "_").replace("/", "_")
            if "_" not in fx_symbol and len(fx_symbol) == 6:
                fx_symbol = f"{fx_symbol[:3]}_{fx_symbol[3:]}"
            df = get_fx(fx_symbol)
        elif asset_type == "Futures":
            df = get_futures_proxy(symbol, period=period, interval=interval)
        elif asset_type == "Crypto":
            df = get_stock(symbol, period=period, interval=interval)
        else:
            df = get_stock(symbol, period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def _robust_fetch(symbol: str, period: str = "6mo", interval: str = "1d"):
    """Robust multi-fallback data fetcher for any asset type.
    Returns (df, asset_type) or (None, asset_type).
    """
    sym = symbol.strip().upper()

    is_intraday = interval not in ("1d", "5d", "1wk", "1mo", "3mo")

    #  Forex: USD/JPY, EUR/USD, USDJPY=X 
    if '/' in sym and '=' not in sym:
        yf_sym = sym.replace('/', '') + '=X'

        # For daily intervals, try cached sources first
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(yf_sym, period=period)
            if df is not None and not df.empty:
                return df, 'fx'

        if not is_intraday:
            fx_key = sym.replace('/', '_')
            for key in [fx_key, fx_key.lower(), fx_key.upper()]:
                try:
                    df = get_fx(key)
                    if df is not None and not df.empty:
                        return df, 'fx'
                except Exception:
                    pass

        # Direct yfinance with interval
        try:
            import yfinance as yf
            df = yf.Ticker(yf_sym).history(period=period, interval=interval)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    if df.columns.duplicated().any():
                        df = df.loc[:, ~df.columns.duplicated(keep='first')]
                if "Close" in df.columns:
                    return df, 'fx'
        except Exception:
            pass

        return None, 'fx'

    #  Forex =X symbols 
    if sym.endswith('=X'):
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'fx'

        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'fx'

        return None, 'fx'

    #  Futures 
    if '=F' in sym:
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'futures'

        df = get_futures_proxy(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'futures'

        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'futures'

        return None, 'futures'

    #  Indices 
    if sym.startswith('^'):
        if not is_intraday and _HAS_FRESH:
            df = get_fresh_quote(sym, period=period)
            if df is not None and not df.empty:
                return df, 'index'

        df = get_stock(sym, period=period, interval=interval)
        if df is not None and not df.empty:
            return df, 'index'

        return None, 'index'

    #  Crypto / Stocks 
    if sym.endswith("-USD"):
        asset_type = 'crypto'
    else:
        asset_type = 'stock'

    if not is_intraday and _HAS_FRESH:
        df = get_fresh_quote(sym, period=period)
        if df is not None and not df.empty:
            return df, asset_type

    df = get_stock(sym, period=period, interval=interval)
    if df is not None and not df.empty:
        return df, asset_type

    # Last resort direct yfinance
    try:
        import yfinance as yf
        df = yf.Ticker(sym).history(period=period, interval=interval)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
            if "Close" in df.columns:
                return df, asset_type
    except Exception:
        pass

    return None, asset_type

def _has_valid_volume(df) -> bool:
    """Check if a dataframe has meaningful volume data (not all zero/NaN)."""
    if "Volume" not in df.columns:
        return False
    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    vol = vol.dropna()
    if len(vol) == 0:
        return False
    # Check if all zeros or near-zero
    if vol.sum() == 0 or vol.max() < 1:
        return False
    return True

def show_custom_dashboard():
    _ensure_ml()
    st.markdown("## Octavian Institutional Terminal")
    
    # Sidebar Configuration to declutter main view
    with st.sidebar:
        st.header("Configuration")
        
        search_query = st.text_input(
            "Ticker / Symbol",
            placeholder="AAPL, USD/JPY, ES=F, BTC-USD",
            help="Enter stock ticker, FX pair (USD/JPY), futures (ES=F), or crypto (BTC-USD)"
        )
        
        asset_type = st.selectbox(
            "Asset Class",
            ["Auto-detect", "Stock", "FX", "Futures", "Crypto"]
        )
        
        _CD_TIMEFRAMES = {
            "1 Day (1m)": {"period": "1d", "interval": "1m"},
            "5 Days (5m)": {"period": "5d", "interval": "5m"},
            "1 Month (30m)": {"period": "1mo", "interval": "30m"},
            "3 Months (Daily)": {"period": "3mo", "interval": "1d"},
            "6 Months (Daily)": {"period": "6mo", "interval": "1d"},
            "1 Year (Daily)": {"period": "1y", "interval": "1d"},
            "2 Years (Daily)": {"period": "2y", "interval": "1d"},
            "5 Years (Weekly)": {"period": "5y", "interval": "1wk"},
            "All Time (Weekly)": {"period": "max", "interval": "1wk"},
        }
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            tf_label = st.selectbox("Timeframe", list(_CD_TIMEFRAMES.keys()), index=7)
            tf_cfg = _CD_TIMEFRAMES[tf_label]
            period = tf_cfg["period"]
        with col_cfg2:
            interval = st.selectbox("Interval",
                ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                index=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"].index(tf_cfg["interval"]),
                help="Auto-set from timeframe. Override if needed.")

        st.markdown("---")
        show_model_trades = st.toggle("Show Potential Model Trades", value=False, key="cd_show_trades")
        if show_model_trades:
            trade_strategy = st.selectbox("Strategy", ["combined", "rsi", "sma_cross", "macd"],
                                           index=0, key="cd_trade_strat",
                                           format_func=lambda x: {"combined": "Combined (RSI+SMA+MACD)",
                                                                    "rsi": "RSI Crossover",
                                                                    "sma_cross": "SMA 20/50 Cross",
                                                                    "macd": "MACD Cross"}[x])
        else:
            trade_strategy = "combined"

        analyze_btn = st.button("Initialize Analysis", type="primary", use_container_width=True)

    # Info banner if ML missing
    if not ML_AVAILABLE:
        st.warning("ML libraries missing. Running in technical-analysis-only mode.")

    # Handle Analysis Trigger
    if analyze_btn and search_query:
        st.session_state['analyze_symbol'] = search_query.strip().upper()
        st.session_state['analyze_asset_type'] = asset_type
        st.session_state['analyze_period'] = period
        st.session_state['analyze_interval'] = interval
        st.rerun()

    # Main Analysis View
    if 'analyze_symbol' in st.session_state and st.session_state['analyze_symbol']:
        symbol = st.session_state['analyze_symbol']
        asset_type = st.session_state.get('analyze_asset_type', 'Auto-detect')
        period = st.session_state.get('analyze_period', '1y')
        interval = st.session_state.get('analyze_interval', '1d')
        
        # Auto-detect logic
        if asset_type == "Auto-detect":
            if "_" in symbol or symbol.endswith("=X") or '/' in symbol:
                asset_type = "FX"
            elif "=F" in symbol:
                asset_type = "Futures"
            elif symbol.endswith("-USD"):
                asset_type = "Crypto"
            else:
                asset_type = "Stock"  # Default to stock if no other conditions match
        
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            df, fetched_asset_type = _robust_fetch(symbol, period=period, interval=interval)
        
        if df is None or df.empty:
            st.error(f"No data found for {symbol}")
            st.info("Tips:\n"
                    "- Forex: `USD/JPY`, `EUR/USD`, `GBPUSD=X`\n"
                    "- Futures: `ES=F`, `GC=F`, `CL=F`\n"
                    "- Crypto: `BTC-USD`, `ETH-USD`\n"
                    "- Stocks: `AAPL`, `MSFT`, `NVDA`\n"
                    "- Indices: `^GSPC`, `^IXIC`")
            return

        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Extract close safely
        close = _safe_col(df, 'Close').dropna().astype(float)
        if len(close) < 5:
            st.error("Insufficient data (less than 5 bars).")
            return

        current_price = float(close.iloc[-1])
        # Use true previous trading day close from fast_info, not close.iloc[-2]
        prev_price = None
        if _HAS_REALTIME and _ds_realtime:
            try:
                _, _prev = _ds_realtime(symbol)
                if _prev and _prev > 0:
                    prev_price = float(_prev)
            except Exception:
                pass
        if prev_price is None:
            prev_price = float(close.iloc[-2]) if len(close) >= 2 else current_price
        pct_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

        has_vol = _has_valid_volume(df)

        #  Live Price Ticker (auto-refresh without full reload) 
        @st.fragment(run_every=30)
        def _live_price_ticker():
            try:
                live_p = None
                if _HAS_FRESH:
                    ldf = get_fresh_quote(symbol, period="1d")
                    if ldf is not None and not ldf.empty and "Close" in ldf.columns:
                        lc = ldf["Close"]
                        if isinstance(lc, pd.DataFrame):
                            lc = lc.iloc[:, 0]
                        lc = lc.dropna().astype(float)
                        if len(lc) > 0:
                            live_p = float(lc.iloc[-1])
                if live_p is None:
                    live_p = current_price
                live_chg = ((live_p - prev_price) / prev_price * 100) if prev_price > 0 else 0
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Price", f"${live_p:,.4f}" if live_p < 10 else f"${live_p:,.2f}", f"{live_chg:+.2f}%")
                with mc2:
                    hi = float(close.max())
                    st.metric("Period High", f"${hi:,.4f}" if hi < 10 else f"${hi:,.2f}")
                with mc3:
                    lo = float(close.min())
                    st.metric("Period Low", f"${lo:,.4f}" if lo < 10 else f"${lo:,.2f}")
                with mc4:
                    total_ret = ((live_p / float(close.iloc[0])) - 1) * 100
                    st.metric(f"Period Return ({period})", f"{total_ret:+.2f}%")
            except Exception:
                pass

        st.markdown(f"### {symbol} — {asset_type}")
        _live_price_ticker()

        #  Integrated AI Insights (from Dashboard) 
        try:
            from market_movers import _generate_ai_insights, _calculate_technicals
            # Calculate technicals for AI
            ai_tech = _calculate_technicals(df)
            ai_data = _generate_ai_insights(symbol, ai_tech, df)
            
            # Display colored signal banner
            sig = ai_data.get("signal", "NEUTRAL")
            sig_color = "#00ff88" if "BULLISH" in sig else "#ff5252" if "BEARISH" in sig else "#8b949e"
            
            st.markdown(
                f'<div style="border-left: 4px solid {sig_color}; padding: 10px 15px; background: rgba(255,255,255,0.05); margin-bottom: 20px; border-radius: 4px;">'
                f'<strong style="color: {sig_color}; font-size: 1.1em;">AI SIGNAL: {sig}</strong>'
                f'<span style="margin-left: 15px; color: #ccc;">Probability: Bull {ai_data.get("bullish_prob",0):.0%} | Bear {ai_data.get("bearish_prob",0):.0%}</span>'
                f'<div style="margin-top: 5px; font-size: 0.9em; color: #ddd;">{ai_data.get("outlook", "")}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            with st.expander("AI Analysis & Trade Ideas", expanded=True):
                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    st.markdown("**Analysis Insights**")
                    for insight in ai_data.get("insights", [])[:5]:
                        st.markdown(f"- {insight}")
                with col_ai2:
                    st.markdown("**Trade Setup Ideas**")
                    ideas = ai_data.get("trade_ideas", [])
                    if ideas:
                        for idea in ideas:
                            st.markdown(f"- {idea}")
                    else:
                        st.caption("No clear trade setup identified.")

        except ImportError:
            pass
        except Exception as e:
            st.error(f"AI Analysis Error: {e}")

        # Generate model trades if enabled
        trade_markers = []
        if show_model_trades and len(close) >= 30:
            trade_markers = generate_model_trades(close, df, trade_strategy)

        #  Tabs 
        tabs = st.tabs(["Price & Technicals", "Volume & Momentum", "Risk", "ML Predictions"])

        #  Tab 1: Price & Technicals 
        with tabs[0]:
            # Moving Averages
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(min(50, len(close))).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()

            # Only add volume subplot if volume data exists
            fig_price = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
                                    row_heights=[0.75, 0.25] if has_vol else [1.0],
                                    vertical_spacing=0.08)
            fig_price.add_trace(go.Scatter(x=close.index, y=close.values,
                mode='lines', name='Close', line=dict(color='white', width=2)), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=sma20.index, y=sma20.values,
                mode='lines', name='SMA 20', line=dict(color='#42a5f5', width=1, dash='dot')), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=sma50.index, y=sma50.values,
                mode='lines', name='SMA 50', line=dict(color='#ef5350', width=1, dash='dot')), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=ema20.index, y=ema20.values,
                mode='lines', name='EMA 20', line=dict(color='#66bb6a', width=1, dash='dash')), row=1, col=1)

            if has_vol:
                vol = _safe_col(df, 'Volume').dropna().astype(float)
                fig_price.add_trace(go.Bar(x=vol.index, y=vol.values,
                    name='Volume', marker_color='rgba(100,100,255,0.3)'), row=2, col=1)

            #  Add model trade markers 
            if show_model_trades and trade_markers:
                add_trade_markers_to_fig(fig_price, trade_markers, row=1, col=1)

            fig_price.update_layout(
                title=dict(text=f"{symbol} Price & Moving Averages", x=0.01, font=dict(size=14)),
                height=550 if show_model_trades else 480, template="plotly_dark", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_price, use_container_width=True)

            #  Trade summary if enabled 
            if show_model_trades and trade_markers:
                summary = get_trade_summary(trade_markers)
                if summary["total_trades"] > 0:
                    st.markdown("#### Model Trade Summary")
                    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                    with tc1: st.metric("Trades", summary["total_trades"])
                    with tc2: st.metric("Win Rate", f"{summary['win_rate']:.0%}")
                    with tc3: st.metric("Avg Win", f"{summary['avg_win']:+.2%}")
                    with tc4: st.metric("Avg Loss", f"{summary['avg_loss']:+.2%}")
                    with tc5: st.metric("Total Return", f"{summary['total_return']:+.2%}")
                    tc6, tc7, tc8 = st.columns(3)
                    with tc6: st.metric("Best Trade", f"{summary['best_trade']:+.2%}")
                    with tc7: st.metric("Worst Trade", f"{summary['worst_trade']:+.2%}")
                    with tc8:
                        pf = summary['profit_factor']
                        st.metric("Profit Factor", f"{pf:.2f}" if pf < 100 else "Inf")
                    with st.expander("Trade Log"):
                        log_rows = []
                        for m in trade_markers:
                            row_data = {"Time": str(m.timestamp)[:10], "Action": m.side,
                                        "Price": f"${m.price:,.2f}", "Reason": m.reason}
                            row_data["P&L"] = f"{m.pnl_pct:+.2%} (${m.pnl_dollar:+,.2f})" if m.pnl_pct is not None else "—"
                            log_rows.append(row_data)
                        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

            # RSI
            rsi = _compute_rsi(close, 14)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi.values,
                mode='lines', name='RSI', line=dict(color='#ab47bc', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation=dict(text="OB", showarrow=False, font=dict(size=10)))
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                              annotation=dict(text="OS", showarrow=False, font=dict(size=10)))
            fig_rsi.update_layout(
                title=dict(text=f"{symbol} RSI (14)", x=0.01, font=dict(size=13)),
                height=220, template="plotly_dark", yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=35, b=10), showlegend=False,
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD
            macd_line, signal_line, macd_hist = _compute_macd(close)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line.values,
                mode='lines', name='MACD', line=dict(color='cyan', width=2)))
            fig_macd.add_trace(go.Scatter(x=signal_line.index, y=signal_line.values,
                mode='lines', name='Signal', line=dict(color='magenta', width=1.5)))
            fig_macd.add_trace(go.Bar(x=macd_hist.index, y=macd_hist.values,
                name='Histogram', marker_color=['green' if v >= 0 else 'red' for v in macd_hist.values]))
            fig_macd.update_layout(
                title=dict(text=f"{symbol} MACD", x=0.01, font=dict(size=13)),
                height=220, template="plotly_dark",
                margin=dict(l=10, r=10, t=35, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            )
            st.plotly_chart(fig_macd, use_container_width=True)

            # Key levels
            current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50
            sma20_val = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else 0
            sma50_val = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else 0
            macd_val = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1: st.metric("RSI (14)", f"{current_rsi:.1f}", "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
            with lc2: st.metric("vs SMA20", f"{'Above' if current_price > sma20_val else 'Below'}", f"${sma20_val:,.2f}")
            with lc3: st.metric("vs SMA50", f"{'Above' if current_price > sma50_val else 'Below'}", f"${sma50_val:,.2f}")
            with lc4: st.metric("MACD Hist", f"{macd_val:.4f}", "Rising" if macd_val > 0 else "Falling")

        #  Tab 2: Volume & Momentum 
        with tabs[1]:
            if has_vol:
                vol = _safe_col(df, 'Volume').dropna().astype(float)
                vol_sma20 = vol.rolling(20).mean()
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=vol.index, y=vol.values, name='Volume', marker_color='rgba(100,149,237,0.5)'))
                fig_vol.add_trace(go.Scatter(x=vol_sma20.index, y=vol_sma20.values, mode='lines', name='Vol SMA 20', line=dict(color='orange', width=2)))
                fig_vol.update_layout(
                    title=dict(text=f"{symbol} Volume", x=0.01, font=dict(size=13)),
                    height=280, template="plotly_dark",
                    margin=dict(l=10, r=10, t=35, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.info(f"Volume data is not available for {symbol}. "
                        f"This is normal for forex pairs (OTC), certain indices, and some futures contracts. "
                        f"Price-based indicators (RSI, MACD, SMAs) remain fully functional.")

            st.markdown("### Return Distribution")
            daily_rets = close.pct_change().dropna()
            if len(daily_rets) > 10:
                fig_dist = px.histogram(x=daily_rets.values * 100, nbins=50, title="Daily Return Distribution (%)", color_discrete_sequence=['#636EFA'])
                fig_dist.update_layout(template="plotly_dark", height=280, xaxis_title="Daily Return %", yaxis_title="Frequency",
                                       margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)

        #  Tab 3: Risk 
        with tabs[2]:
            daily_rets = close.pct_change().dropna()
            if len(daily_rets) >= 20:
                ann_vol = float(daily_rets.std() * np.sqrt(252) * 100)
                var_95 = float(daily_rets.quantile(0.05) * 100)
                var_99 = float(daily_rets.quantile(0.01) * 100)
                cum_rets = (1 + daily_rets).cumprod()
                running_max = cum_rets.expanding().max()
                drawdown = (cum_rets - running_max) / running_max
                max_dd = float(drawdown.min() * 100)
                sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252)) if daily_rets.std() > 0 else 0
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1: st.metric("Ann. Volatility", f"{ann_vol:.2f}%")
                with rc2: st.metric("VaR (95%)", f"{var_95:+.2f}%")
                with rc3: st.metric("Max Drawdown", f"{max_dd:.2f}%")
                with rc4: st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100, mode='lines', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=1), name='Drawdown'))
                fig_dd.update_layout(title=dict(text="Drawdown (%)", x=0.01, font=dict(size=13)), height=230, template="plotly_dark", yaxis_title="Drawdown %", margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_dd, use_container_width=True)
                rolling_vol = daily_rets.rolling(21).std() * np.sqrt(252) * 100
                fig_rv = go.Figure()
                fig_rv.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines', name='21d Rolling Vol', line=dict(color='#42a5f5', width=2)))
                fig_rv.update_layout(title=dict(text="Rolling 21-Day Annualized Volatility (%)", x=0.01, font=dict(size=13)), height=230, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_rv, use_container_width=True)
            else:
                st.info("Insufficient data for risk analysis (need 20+ bars).")

        #  Tab 4: ML Predictions 
        with tabs[3]:
            if not ML_AVAILABLE:
                st.warning("ML libraries (scikit-learn, xgboost) not available.")
            elif len(close) < 60:
                st.warning("Need at least 60 bars for ML prediction.")
            else:
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    df_ml = pd.DataFrame({'Close': close})
                    df_ml['Return'] = df_ml['Close'].pct_change()
                    df_ml['Direction'] = (df_ml['Return'] > 0).astype(int)
                    df_ml['Lag1'] = df_ml['Close'].shift(1)
                    df_ml['Lag2'] = df_ml['Close'].shift(2)
                    df_ml['Lag3'] = df_ml['Close'].shift(3)
                    df_ml['SMA5'] = df_ml['Close'].rolling(5).mean()
                    df_ml['SMA20'] = df_ml['Close'].rolling(20).mean()
                    df_ml['Vol5'] = df_ml['Return'].rolling(5).std()
                    df_ml.dropna(inplace=True)
                    features = ['Lag1', 'Lag2', 'Lag3', 'SMA5', 'SMA20', 'Vol5']
                    split = int(len(df_ml) * 0.8)
                    train = df_ml[:split]
                    test = df_ml[split:]
                    if len(train) > 20 and len(test) > 5:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(train[features], train['Direction'])
                        test = test.copy()
                        test['Predicted'] = model.predict(test[features])
                        test['Strategy_Return'] = test['Predicted'].shift(1) * test['Return']
                        test['Cum_Strategy'] = (1 + test['Strategy_Return'].fillna(0)).cumprod()
                        test['Cum_Market'] = (1 + test['Return']).cumprod()
                        fig_ml = go.Figure()
                        fig_ml.add_trace(go.Scatter(x=test.index, y=test['Cum_Strategy'], mode='lines', name='ML Strategy', line=dict(color='#00e676', width=2)))
                        fig_ml.add_trace(go.Scatter(x=test.index, y=test['Cum_Market'], mode='lines', name='Buy & Hold', line=dict(color='#ff5252', width=2)))
                        fig_ml.update_layout(title=dict(text=f"{symbol} ML Strategy vs Buy & Hold", x=0.01, font=dict(size=13)), height=320, template="plotly_dark", yaxis_title="Cumulative Return", margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_ml, use_container_width=True)
                        accuracy = (test['Predicted'] == test['Direction']).mean()
                        strat_ret = float(test['Cum_Strategy'].iloc[-1] - 1) * 100
                        market_ret = float(test['Cum_Market'].iloc[-1] - 1) * 100
                        mlc1, mlc2, mlc3 = st.columns(3)
                        with mlc1: st.metric("Model Accuracy", f"{accuracy:.1%}")
                        with mlc2: st.metric("Strategy Return", f"{strat_ret:+.2f}%")
                        with mlc3: st.metric("Market Return", f"{market_ret:+.2f}%")
                    else:
                        st.warning("Insufficient data for train/test split.")
                except ImportError:
                    st.warning("scikit-learn not installed.")
                except Exception as e:
                    st.error(f"ML prediction error: {e}")

    #  Data Download 
    with st.expander("Download Data"):
        if 'analyze_symbol' in st.session_state and st.session_state['analyze_symbol']:
            try:
                csv = df.to_csv().encode('utf-8')
                st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv", key="dl_csv")
            except Exception:
                st.info("No data to download.")

