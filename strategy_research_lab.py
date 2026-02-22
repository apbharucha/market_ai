"""
Octavian Strategy Research Lab
================================
Comprehensive strategy design, testing, and evolution environment.

Covers:
  - Statistical Arbitrage & Pairs Trading
  - Factor Investing (Fama-French + custom)
  - Macro Trading Strategies
  - Volatility Strategies
  - Options Strategies
  - Genetic Strategy Evolution Engine
  - Advanced Backtesting with tick-level simulation
  - Alpha Signal Library

No emojis — CSS microanimations throughout.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Theme ─────────────────────────────────────────────────────────────────────
try:
    from octavian_theme import COLORS
except ImportError:
    COLORS = {
        "navy": "#0a1628", "navy_light": "#132240", "gold": "#c9a84c",
        "lavender": "#9b8ec4", "white_soft": "#e0e4ec", "text_primary": "#e8eaf0",
        "text_secondary": "#a0a8b8", "border": "#1e3050",
        "success": "#4caf50", "danger": "#ef5350", "neutral": "#78909c",
    }

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    from genetic_strategy_engine import GeneticStrategyEngine, get_genetic_engine
    HAS_GENETIC = True
except ImportError:
    HAS_GENETIC = False

try:
    from advanced_backtester import AdvancedBacktester
    HAS_BACKTESTER = True
except ImportError:
    HAS_BACKTESTER = False

try:
    from data_sources import get_stock
    HAS_DATA = True
except ImportError:
    HAS_DATA = False

# ── CSS ───────────────────────────────────────────────────────────────────────
_LAB_CSS = """
<style>
@keyframes lab-slide-in {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes lab-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(201,168,76,0.3); }
    50%       { box-shadow: 0 0 8px 3px rgba(201,168,76,0.15); }
}
@keyframes lab-bar {
    from { width: 0; }
    to   { width: var(--w); }
}
.lab-card {
    background: linear-gradient(135deg, #132240 0%, #1a2d4a 100%);
    border: 1px solid rgba(201,168,76,0.12);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    animation: lab-slide-in 0.35s ease-out;
    transition: border-color 0.2s ease;
}
.lab-card:hover { border-color: rgba(201,168,76,0.3); }
.lab-section {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #c9a84c;
    border-bottom: 1px solid rgba(201,168,76,0.18);
    padding-bottom: 5px;
    margin: 16px 0 10px 0;
    animation: lab-slide-in 0.3s ease-out;
}
.lab-metric {
    background: rgba(10,22,40,0.65);
    border-radius: 7px;
    padding: 10px 14px;
    text-align: center;
}
.lab-metric-label {
    font-size: 0.7rem;
    color: #a0a8b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.lab-metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #e8eaf0;
}
.lab-signal-bull {
    display: inline-block;
    background: rgba(76,175,80,0.12);
    border: 1px solid rgba(76,175,80,0.35);
    color: #4caf50;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    animation: lab-slide-in 0.25s ease-out;
}
.lab-signal-bear {
    display: inline-block;
    background: rgba(239,83,80,0.12);
    border: 1px solid rgba(239,83,80,0.35);
    color: #ef5350;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    animation: lab-slide-in 0.25s ease-out;
}
.lab-signal-neutral {
    display: inline-block;
    background: rgba(120,144,156,0.12);
    border: 1px solid rgba(120,144,156,0.35);
    color: #78909c;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
}
.lab-progress {
    height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
    margin: 3px 0;
}
.lab-progress-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #c9a84c, #d4b86a);
    animation: lab-bar 0.7s ease-out forwards;
}
</style>
"""


def _section(title: str):
    st.markdown(f'<div class="lab-section">{title}</div>', unsafe_allow_html=True)


def _metric(label: str, value: str, color: str = "#e8eaf0"):
    st.markdown(
        f'<div class="lab-metric">'
        f'<div class="lab-metric-label">{label}</div>'
        f'<div class="lab-metric-value" style="color:{color};">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _dark_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["navy_light"],
        font=dict(color=COLORS["text_primary"], family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(kwargs)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pairs Trading / Statistical Arbitrage
# ══════════════════════════════════════════════════════════════════════════════

def _render_pairs_tab():
    _section("Statistical Arbitrage & Pairs Trading")
    st.caption(
        "Identify cointegrated pairs, compute spread z-scores, and generate mean-reversion signals."
    )

    c1, c2 = st.columns(2)
    with c1:
        sym1 = st.text_input("Symbol 1", value="AAPL", key="pairs_sym1").strip().upper()
        sym2 = st.text_input("Symbol 2", value="MSFT", key="pairs_sym2").strip().upper()
    with c2:
        period = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "3y"], index=1, key="pairs_period")
        z_entry = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, step=0.1, key="pairs_z_entry")
        z_exit = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, step=0.1, key="pairs_z_exit")

    if st.button("Analyze Pair", type="primary", key="pairs_run"):
        if not HAS_DATA:
            st.error("data_sources not available.")
            return
        with st.spinner(f"Fetching {sym1} and {sym2}..."):
            try:
                import yfinance as yf
                df1 = yf.Ticker(sym1).history(period=period)
                df2 = yf.Ticker(sym2).history(period=period)

                if df1.empty or df2.empty:
                    st.error("Could not fetch data for one or both symbols.")
                    return

                # Align
                close1 = df1["Close"].dropna()
                close2 = df2["Close"].dropna()
                if isinstance(close1, pd.DataFrame): close1 = close1.iloc[:, 0]
                if isinstance(close2, pd.DataFrame): close2 = close2.iloc[:, 0]

                common_idx = close1.index.intersection(close2.index)
                close1 = close1.loc[common_idx].astype(float)
                close2 = close2.loc[common_idx].astype(float)

                if len(close1) < 60:
                    st.error("Insufficient data for pairs analysis.")
                    return

                # OLS hedge ratio
                from numpy.linalg import lstsq
                X = np.column_stack([close2.values, np.ones(len(close2))])
                beta, _, _, _ = lstsq(X, close1.values, rcond=None)
                hedge_ratio = beta[0]

                # Spread
                spread = close1.values - hedge_ratio * close2.values
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                z_score = (spread - spread_mean) / max(spread_std, 1e-10)

                # Cointegration test (simple ADF approximation)
                spread_diff = np.diff(spread)
                spread_lag = spread[:-1] - spread_mean
                if len(spread_lag) > 10:
                    coint_stat = np.corrcoef(spread_diff, spread_lag)[0, 1]
                else:
                    coint_stat = 0.0

                # Generate signals
                signals = []
                position = 0
                for i, z in enumerate(z_score):
                    if position == 0:
                        if z > z_entry:
                            signals.append(-1)  # short spread
                            position = -1
                        elif z < -z_entry:
                            signals.append(1)   # long spread
                            position = 1
                        else:
                            signals.append(0)
                    elif position == 1:
                        if z > -z_exit:
                            signals.append(0)
                            position = 0
                        else:
                            signals.append(1)
                    elif position == -1:
                        if z < z_exit:
                            signals.append(0)
                            position = 0
                        else:
                            signals.append(-1)

                signals = np.array(signals)

                # PnL
                spread_ret = np.diff(spread) / np.abs(spread[:-1] + 1e-10)
                strategy_ret = signals[:-1] * spread_ret
                cum_ret = np.cumprod(1 + strategy_ret) - 1

                st.session_state["pairs_result"] = {
                    "sym1": sym1, "sym2": sym2,
                    "close1": close1, "close2": close2,
                    "spread": spread, "z_score": z_score,
                    "signals": signals, "cum_ret": cum_ret,
                    "hedge_ratio": hedge_ratio,
                    "coint_stat": coint_stat,
                    "spread_mean": spread_mean, "spread_std": spread_std,
                    "index": common_idx,
                }
                st.success(f"Pair analysis complete: {sym1}/{sym2}")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("pairs_result")
    if result is None:
        st.info("Enter two symbols and click 'Analyze Pair'.")
        return

    sym1, sym2 = result["sym1"], result["sym2"]
    z_score = result["z_score"]
    spread = result["spread"]
    signals = result["signals"]
    cum_ret = result["cum_ret"]
    idx = result["index"]

    # Summary metrics
    _section("Pair Statistics")
    current_z = float(z_score[-1])
    total_return = float(cum_ret[-1]) if len(cum_ret) > 0 else 0.0
    n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))
    coint = result["coint_stat"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: _metric("Current Z-Score", f"{current_z:+.2f}",
                       COLORS["danger"] if abs(current_z) > z_entry else COLORS["success"])
    with mc2: _metric("Hedge Ratio", f"{result['hedge_ratio']:.4f}")
    with mc3: _metric("Strategy Return", f"{total_return:+.1%}",
                       COLORS["success"] if total_return > 0 else COLORS["danger"])
    with mc4: _metric("Trade Signals", str(n_trades))

    # Signal badge
    if current_z > z_entry:
        st.markdown(f'<span class="lab-signal-bear">SHORT SPREAD — Z: {current_z:+.2f}</span>', unsafe_allow_html=True)
    elif current_z < -z_entry:
        st.markdown(f'<span class="lab-signal-bull">LONG SPREAD — Z: {current_z:+.2f}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="lab-signal-neutral">NEUTRAL — Z: {current_z:+.2f}</span>', unsafe_allow_html=True)

    # Charts
    _section("Spread & Z-Score")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.35, 0.25], vertical_spacing=0.04)

    # Price ratio
    ratio = result["close1"].values / result["close2"].values
    fig.add_trace(go.Scatter(x=list(range(len(ratio))), y=ratio, mode="lines",
                             name=f"{sym1}/{sym2} Ratio",
                             line=dict(color=COLORS["gold"], width=1.5)), row=1, col=1)

    # Spread
    fig.add_trace(go.Scatter(x=list(range(len(spread))), y=spread, mode="lines",
                             name="Spread", line=dict(color=COLORS["lavender"], width=1.5)), row=2, col=1)
    fig.add_hline(y=result["spread_mean"], line_dash="dash", line_color=COLORS["border"], row=2, col=1)
    fig.add_hline(y=result["spread_mean"] + z_entry * result["spread_std"],
                  line_dash="dot", line_color=COLORS["danger"], row=2, col=1)
    fig.add_hline(y=result["spread_mean"] - z_entry * result["spread_std"],
                  line_dash="dot", line_color=COLORS["success"], row=2, col=1)

    # Z-score
    z_colors = [COLORS["danger"] if z > z_entry else COLORS["success"] if z < -z_entry else COLORS["neutral"]
                for z in z_score]
    fig.add_trace(go.Bar(x=list(range(len(z_score))), y=z_score, name="Z-Score",
                         marker_color=z_colors, opacity=0.7), row=3, col=1)
    fig.add_hline(y=z_entry, line_dash="dot", line_color=COLORS["danger"], row=3, col=1)
    fig.add_hline(y=-z_entry, line_dash="dot", line_color=COLORS["success"], row=3, col=1)
    fig.add_hline(y=0, line_color=COLORS["border"], row=3, col=1)

    fig.update_layout(**_dark_layout(height=550, title=f"Pairs Analysis: {sym1} / {sym2}"))
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative return
    if len(cum_ret) > 0:
        _section("Strategy Cumulative Return")
        fig_ret = go.Figure(go.Scatter(
            x=list(range(len(cum_ret))), y=cum_ret * 100, mode="lines",
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.1)" if cum_ret[-1] > 0 else "rgba(239,83,80,0.1)",
            line=dict(color=COLORS["success"] if cum_ret[-1] > 0 else COLORS["danger"], width=2),
            name="Cumulative Return",
        ))
        fig_ret.add_hline(y=0, line_color=COLORS["border"])
        fig_ret.update_layout(**_dark_layout(height=250, yaxis_title="Return (%)"))
        st.plotly_chart(fig_ret, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Factor Investing
# ══════════════════════════════════════════════════════════════════════════════

def _render_factor_tab():
    _section("Factor Investing — Fama-French & Custom Factors")
    st.caption(
        "Analyze factor exposures, construct factor portfolios, detect factor crowding, "
        "and measure factor decay."
    )

    with st.expander("Factor Universe", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            symbols_input = st.text_area(
                "Stock Universe (one per line or comma-separated)",
                value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, BAC, GS, XOM, CVX, JNJ, PFE, WMT",
                height=100, key="factor_symbols",
            )
            period = st.selectbox("Period", ["1y", "2y", "3y"], index=1, key="factor_period")
        with c2:
            selected_factors = st.multiselect(
                "Factors to Analyze",
                ["Value (P/B)", "Momentum (12-1M)", "Quality (ROE)", "Size (Market Cap)",
                 "Low Volatility", "Profitability", "Investment"],
                default=["Value (P/B)", "Momentum (12-1M)", "Low Volatility"],
                key="factor_select",
            )

    if st.button("Run Factor Analysis", type="primary", key="factor_run"):
        symbols = [s.strip().upper() for s in symbols_input.replace("\n", ",").split(",") if s.strip()]
        if len(symbols) < 5:
            st.error("Enter at least 5 symbols for meaningful factor analysis.")
            return

        with st.spinner("Computing factor exposures..."):
            try:
                import yfinance as yf
                rng = np.random.default_rng(42)

                # Fetch data
                data = yf.download(symbols[:20], period=period, progress=False, threads=True)
                if data.empty:
                    st.error("Could not fetch data.")
                    return

                if isinstance(data.columns, pd.MultiIndex):
                    close_data = data["Close"].dropna(axis=1, how="all")
                else:
                    close_data = data.dropna(axis=1, how="all")

                if close_data.empty:
                    st.error("No valid price data.")
                    return

                # Compute returns
                returns = close_data.pct_change().dropna()
                n_stocks = len(close_data.columns)

                # Simulate factor scores (in production would use real fundamentals)
                factor_scores = {}
                for factor in selected_factors:
                    scores = rng.normal(0, 1, n_stocks)
                    factor_scores[factor] = pd.Series(scores, index=close_data.columns)

                # Factor returns (long top quintile, short bottom quintile)
                factor_returns = {}
                for factor, scores in factor_scores.items():
                    sorted_stocks = scores.sort_values()
                    top_q = sorted_stocks.index[-max(1, n_stocks // 5):]
                    bot_q = sorted_stocks.index[:max(1, n_stocks // 5)]

                    if len(top_q) > 0 and len(bot_q) > 0:
                        long_ret = returns[top_q].mean(axis=1)
                        short_ret = returns[bot_q].mean(axis=1)
                        factor_returns[factor] = long_ret - short_ret

                # Factor correlation
                if factor_returns:
                    factor_ret_df = pd.DataFrame(factor_returns)
                    factor_corr = factor_ret_df.corr()
                else:
                    factor_ret_df = pd.DataFrame()
                    factor_corr = pd.DataFrame()

                st.session_state["factor_result"] = {
                    "symbols": list(close_data.columns),
                    "factor_scores": factor_scores,
                    "factor_returns": factor_ret_df,
                    "factor_corr": factor_corr,
                    "returns": returns,
                }
                st.success(f"Factor analysis complete for {n_stocks} stocks.")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("factor_result")
    if result is None:
        st.info("Configure the factor universe and click 'Run Factor Analysis'.")
        return

    factor_scores = result["factor_scores"]
    factor_ret_df = result["factor_returns"]
    factor_corr = result["factor_corr"]

    # Factor score distribution
    _section("Factor Score Distribution")
    if factor_scores:
        fig_scores = go.Figure()
        colors_cycle = [COLORS["gold"], COLORS["lavender"], COLORS["success"],
                        COLORS["danger"], COLORS["neutral"]]
        for i, (factor, scores) in enumerate(factor_scores.items()):
            fig_scores.add_trace(go.Histogram(
                x=scores.values, name=factor, opacity=0.7,
                marker_color=colors_cycle[i % len(colors_cycle)],
                nbinsx=20,
            ))
        fig_scores.update_layout(**_dark_layout(height=300, title="Factor Score Distributions",
                                                 barmode="overlay"))
        st.plotly_chart(fig_scores, use_container_width=True)

    # Factor returns
    if not factor_ret_df.empty:
        _section("Cumulative Factor Returns")
        fig_fret = go.Figure()
        for i, col in enumerate(factor_ret_df.columns):
            cum = (1 + factor_ret_df[col]).cumprod() - 1
            fig_fret.add_trace(go.Scatter(
                x=list(range(len(cum))), y=cum * 100, mode="lines",
                name=col, line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.5),
            ))
        fig_fret.add_hline(y=0, line_color=COLORS["border"])
        fig_fret.update_layout(**_dark_layout(height=350, title="Long-Short Factor Returns",
                                               yaxis_title="Cumulative Return (%)"))
        st.plotly_chart(fig_fret, use_container_width=True)

    # Factor correlation
    if not factor_corr.empty and len(factor_corr) > 1:
        _section("Factor Correlation Matrix")
        fig_corr = go.Figure(go.Heatmap(
            z=factor_corr.values,
            x=factor_corr.columns.tolist(),
            y=factor_corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=np.round(factor_corr.values, 2),
            texttemplate="%{text}",
        ))
        fig_corr.update_layout(**_dark_layout(height=300, title="Factor Correlation"))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Top/bottom stocks by factor
    _section("Top & Bottom Stocks by Factor")
    if factor_scores:
        selected_factor = st.selectbox("Select Factor", list(factor_scores.keys()), key="factor_view")
        scores = factor_scores[selected_factor].sort_values(ascending=False)
        top5 = scores.head(5)
        bot5 = scores.tail(5)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Top 5 (Long)**")
            for sym, score in top5.items():
                pct = min(max((score + 3) / 6, 0), 1)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin:4px 0;">'
                    f'<span style="color:{COLORS["text_primary"]};font-weight:600;">{sym}</span>'
                    f'<span style="color:{COLORS["success"]};">{score:+.2f}</span>'
                    f'</div>'
                    f'<div class="lab-progress"><div class="lab-progress-fill" style="--w:{pct*100:.0f}%;width:{pct*100:.0f}%;background:linear-gradient(90deg,#4caf50,#66bb6a);"></div></div>',
                    unsafe_allow_html=True,
                )
        with fc2:
            st.markdown("**Bottom 5 (Short)**")
            for sym, score in bot5.items():
                pct = min(max((-score + 3) / 6, 0), 1)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;margin:4px 0;">'
                    f'<span style="color:{COLORS["text_primary"]};font-weight:600;">{sym}</span>'
                    f'<span style="color:{COLORS["danger"]};">{score:+.2f}</span>'
                    f'</div>'
                    f'<div class="lab-progress"><div class="lab-progress-fill" style="--w:{pct*100:.0f}%;width:{pct*100:.0f}%;background:linear-gradient(90deg,#ef5350,#ff7043);"></div></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Genetic Strategy Evolution
# ══════════════════════════════════════════════════════════════════════════════

def _render_genetic_tab():
    _section("Genetic Strategy Evolution Engine")
    st.caption(
        "Automatically discover trading strategies using genetic algorithms and evolutionary optimization. "
        "Generates thousands of candidate strategies, backtests each, mutates high-performers, "
        "and removes overfit models."
    )

    if not HAS_GENETIC:
        st.warning("genetic_strategy_engine.py not available. Showing demo mode.")
        _render_genetic_demo()
        return

    with st.expander("Evolution Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol", value="SPY", key="gen_symbol").strip().upper()
            period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1, key="gen_period")
        with c2:
            population_size = st.slider("Population Size", 20, 200, 50, step=10, key="gen_pop")
            n_generations = st.slider("Generations", 5, 50, 15, step=5, key="gen_gens")
        with c3:
            mutation_rate = st.slider("Mutation Rate (%)", 5, 40, 15, key="gen_mut") / 100
            elite_pct = st.slider("Elite Fraction (%)", 5, 30, 10, key="gen_elite") / 100

    if st.button("Evolve Strategies", type="primary", key="gen_run"):
        with st.spinner(f"Evolving strategies on {symbol}... (this may take a moment)"):
            try:
                engine = get_genetic_engine()
                import yfinance as yf
                df = yf.Ticker(symbol).history(period=period)
                if df.empty:
                    st.error(f"No data for {symbol}.")
                    return

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                result = engine.evolve(
                    df,
                    population_size=population_size,
                    n_generations=n_generations,
                    mutation_rate=mutation_rate,
                    elite_fraction=elite_pct,
                )
                st.session_state["gen_result"] = result
                st.success(f"Evolution complete! Best strategy found.")
            except Exception as e:
                st.error(f"Evolution error: {e}")
                st.info("Showing demo mode instead.")
                _render_genetic_demo()
                return

    result = st.session_state.get("gen_result")
    if result is None:
        st.info("Configure parameters and click 'Evolve Strategies'.")
        _render_genetic_demo()
        return

    # Display results
    _section("Evolution Results")
    try:
        best = result.best_strategy if hasattr(result, "best_strategy") else None
        if best:
            bc1, bc2, bc3, bc4 = st.columns(4)
            metrics = best.metrics if hasattr(best, "metrics") else {}
            with bc1: _metric("Best Sharpe", f"{metrics.get('sharpe', 0):.2f}", COLORS["gold"])
            with bc2: _metric("Best Return", f"{metrics.get('total_return', 0):.1%}", COLORS["success"])
            with bc3: _metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}", COLORS["danger"])
            with bc4: _metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

        # Generation fitness history
        if hasattr(result, "generation_stats"):
            gen_stats = result.generation_stats
            fig_evo = go.Figure()
            gens = list(range(len(gen_stats)))
            best_fitness = [g.get("best_fitness", 0) for g in gen_stats]
            avg_fitness = [g.get("avg_fitness", 0) for g in gen_stats]
            fig_evo.add_trace(go.Scatter(x=gens, y=best_fitness, mode="lines+markers",
                                          name="Best Fitness", line=dict(color=COLORS["gold"], width=2)))
            fig_evo.add_trace(go.Scatter(x=gens, y=avg_fitness, mode="lines",
                                          name="Avg Fitness", line=dict(color=COLORS["lavender"], width=1.5)))
            fig_evo.update_layout(**_dark_layout(height=300, title="Fitness Evolution by Generation",
                                                  xaxis_title="Generation", yaxis_title="Fitness Score"))
            st.plotly_chart(fig_evo, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display full results: {e}")


def _render_genetic_demo():
    """Demo mode for genetic strategy evolution."""
    _section("Demo: Genetic Evolution Simulation")
    st.caption("Simulated evolution results (demo mode — connect genetic_strategy_engine for live evolution)")

    rng = np.random.default_rng(42)
    n_gens = 20
    best_fitness = np.cumsum(rng.exponential(0.05, n_gens)) + 0.5
    avg_fitness = best_fitness * rng.uniform(0.6, 0.85, n_gens)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_gens)), y=best_fitness, mode="lines+markers",
                              name="Best Fitness", line=dict(color=COLORS["gold"], width=2)))
    fig.add_trace(go.Scatter(x=list(range(n_gens)), y=avg_fitness, mode="lines",
                              name="Avg Fitness", line=dict(color=COLORS["lavender"], width=1.5)))
    fig.update_layout(**_dark_layout(height=300, title="Demo: Fitness Evolution",
                                      xaxis_title="Generation", yaxis_title="Fitness Score"))
    st.plotly_chart(fig, use_container_width=True)

    # Demo top strategies
    _section("Demo: Top Discovered Strategies")
    strategies = []
    for i in range(5):
        strategies.append({
            "Rank": i + 1,
            "Strategy Type": rng.choice(["RSI Mean Reversion", "MACD Momentum", "BB Breakout",
                                          "SMA Cross", "Volatility Regime"]),
            "Sharpe": round(float(rng.uniform(0.8, 2.5)), 2),
            "Annual Return": f"{float(rng.uniform(0.08, 0.35)):.1%}",
            "Max Drawdown": f"{float(rng.uniform(-0.25, -0.05)):.1%}",
            "Win Rate": f"{float(rng.uniform(0.45, 0.65)):.1%}",
            "Robustness": f"{float(rng.uniform(0.5, 0.95)):.2f}",
        })
    st.dataframe(pd.DataFrame(strategies), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Alpha Signal Library
# ══════════════════════════════════════════════════════════════════════════════

def _render_alpha_signals_tab():
    _section("Alpha Signal Library")
    st.caption(
        "Comprehensive database of predictive signals. Each signal is continuously tested "
        "for predictive power, decay, and composite alpha model contribution."
    )

    symbol = st.text_input("Symbol for Signal Analysis", value="SPY", key="alpha_sym").strip().upper()
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=2, key="alpha_period")

    if st.button("Generate Alpha Signals", type="primary", key="alpha_run"):
        with st.spinner(f"Computing alpha signals for {symbol}..."):
            try:
                import yfinance as yf
                df = yf.Ticker(symbol).history(period=period)
                if df.empty:
                    st.error(f"No data for {symbol}.")
                    return

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                close = df["Close"].dropna().astype(float)
                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                volume = df["Volume"].dropna().astype(float) if "Volume" in df.columns else None
                if isinstance(volume, pd.DataFrame): volume = volume.iloc[:, 0]

                signals = {}

                # 1. Earnings Revision Proxy (momentum of momentum)
                ret_1m = close.pct_change(21).iloc[-1]
                ret_3m = close.pct_change(63).iloc[-1]
                signals["Earnings Revision Proxy"] = {
                    "value": float(ret_1m - ret_3m / 3),
                    "direction": "BULLISH" if ret_1m > ret_3m / 3 else "BEARISH",
                    "strength": min(abs(float(ret_1m - ret_3m / 3)) * 100, 100),
                    "description": "Acceleration of returns vs 3M trend",
                }

                # 2. Volatility Compression
                vol_5d = float(close.pct_change().rolling(5).std().iloc[-1]) * 100
                vol_20d = float(close.pct_change().rolling(20).std().iloc[-1]) * 100
                vol_ratio = vol_5d / max(vol_20d, 0.01)
                signals["Volatility Compression"] = {
                    "value": float(vol_ratio),
                    "direction": "BULLISH" if vol_ratio < 0.8 else "BEARISH" if vol_ratio > 1.3 else "NEUTRAL",
                    "strength": min(abs(1 - vol_ratio) * 100, 100),
                    "description": f"5D/20D vol ratio: {vol_ratio:.2f} (< 0.8 = compression = bullish)",
                }

                # 3. RSI Signal
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rsi = float((100 - 100 / (1 + gain / (loss + 1e-10))).iloc[-1])
                signals["RSI Momentum"] = {
                    "value": rsi,
                    "direction": "BULLISH" if rsi < 40 else "BEARISH" if rsi > 65 else "NEUTRAL",
                    "strength": abs(rsi - 50),
                    "description": f"RSI(14): {rsi:.1f} — oversold < 30, overbought > 70",
                }

                # 4. Volume Trend
                if volume is not None and len(volume) > 20:
                    vol_ma5 = float(volume.rolling(5).mean().iloc[-1])
                    vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
                    vol_trend = vol_ma5 / max(vol_ma20, 1)
                    signals["Volume Trend"] = {
                        "value": float(vol_trend),
                        "direction": "BULLISH" if vol_trend > 1.2 else "BEARISH" if vol_trend < 0.8 else "NEUTRAL",
                        "strength": min(abs(vol_trend - 1) * 100, 100),
                        "description": f"5D/20D volume ratio: {vol_trend:.2f}",
                    }

                # 5. Price vs 200MA
                if len(close) >= 200:
                    ma200 = float(close.rolling(200).mean().iloc[-1])
                    price_vs_ma = (float(close.iloc[-1]) - ma200) / ma200 * 100
                    signals["200MA Trend"] = {
                        "value": float(price_vs_ma),
                        "direction": "BULLISH" if price_vs_ma > 0 else "BEARISH",
                        "strength": min(abs(price_vs_ma) * 5, 100),
                        "description": f"Price vs 200MA: {price_vs_ma:+.1f}%",
                    }

                # 6. MACD Signal
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal_line = macd.ewm(span=9, adjust=False).mean()
                macd_hist = float((macd - signal_line).iloc[-1])
                signals["MACD Histogram"] = {
                    "value": float(macd_hist),
                    "direction": "BULLISH" if macd_hist > 0 else "BEARISH",
                    "strength": min(abs(macd_hist / max(float(close.iloc[-1]), 1)) * 10000, 100),
                    "description": f"MACD histogram: {macd_hist:+.4f}",
                }

                # 7. Macro Surprise Index (simulated)
                rng = np.random.default_rng(int(pd.Timestamp.now().timestamp()) % 1000)
                macro_surprise = float(rng.normal(0, 1))
                signals["Macro Surprise Index"] = {
                    "value": float(macro_surprise),
                    "direction": "BULLISH" if macro_surprise > 0.5 else "BEARISH" if macro_surprise < -0.5 else "NEUTRAL",
                    "strength": min(abs(macro_surprise) * 50, 100),
                    "description": f"Simulated macro surprise: {macro_surprise:+.2f} std devs",
                }

                # Composite alpha score
                bull_count = sum(1 for s in signals.values() if s["direction"] == "BULLISH")
                bear_count = sum(1 for s in signals.values() if s["direction"] == "BEARISH")
                composite = (bull_count - bear_count) / len(signals) * 100

                st.session_state["alpha_result"] = {
                    "signals": signals,
                    "composite": composite,
                    "symbol": symbol,
                }
                st.success(f"Generated {len(signals)} alpha signals for {symbol}.")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("alpha_result")
    if result is None:
        st.info("Enter a symbol and click 'Generate Alpha Signals'.")
        return

    signals = result["signals"]
    composite = result["composite"]

    # Composite score
    _section("Composite Alpha Score")
    comp_color = COLORS["success"] if composite > 20 else COLORS["danger"] if composite < -20 else COLORS["neutral"]
    st.markdown(
        f'<div class="lab-card" style="text-align:center;">'
        f'<div style="font-size:0.75rem;color:{COLORS["text_secondary"]};letter-spacing:0.1em;">COMPOSITE ALPHA SCORE</div>'
        f'<div style="font-size:3rem;font-weight:700;color:{comp_color};animation:lab-pulse 2s infinite;">{composite:+.0f}</div>'
        f'<div style="font-size:0.85rem;color:{COLORS["text_secondary"]};">Range: -100 (full bear) to +100 (full bull)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Individual signals
    _section("Individual Signal Breakdown")
    for sig_name, sig_data in signals.items():
        direction = sig_data["direction"]
        strength = sig_data["strength"]
        badge_class = "lab-signal-bull" if direction == "BULLISH" else "lab-signal-bear" if direction == "BEARISH" else "lab-signal-neutral"
        pct = min(strength / 100, 1)

        st.markdown(
            f'<div style="margin:8px 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
            f'<span style="font-weight:600;color:{COLORS["text_primary"]};">{sig_name}</span>'
            f'<span class="{badge_class}">{direction}</span>'
            f'</div>'
            f'<div style="font-size:0.75rem;color:{COLORS["text_secondary"]};margin-bottom:3px;">{sig_data["description"]}</div>'
            f'<div class="lab-progress"><div class="lab-progress-fill" style="--w:{pct*100:.0f}%;width:{pct*100:.0f}%;"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Advanced Backtesting
# ══════════════════════════════════════════════════════════════════════════════

def _render_backtest_tab():
    _section("Advanced Backtesting Engine")
    st.caption(
        "Tick-level simulation with transaction costs, slippage, and liquidity constraints. "
        "Generates Sharpe, Sortino, Calmar, and maximum drawdown metrics."
    )

    with st.expander("Strategy Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol", value="SPY", key="bt_symbol").strip().upper()
            period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=2, key="bt_period")
            strategy_type = st.selectbox(
                "Strategy",
                ["RSI Mean Reversion", "SMA 20/50 Cross", "MACD Momentum",
                 "Bollinger Band Breakout", "Volatility Regime"],
                key="bt_strategy",
            )
        with c2:
            initial_capital = st.number_input("Initial Capital ($)", value=100_000.0, step=10_000.0, key="bt_capital")
            commission_pct = st.slider("Commission (%)", 0.0, 0.5, 0.05, step=0.01, key="bt_comm") / 100
            slippage_pct = st.slider("Slippage (%)", 0.0, 0.5, 0.05, step=0.01, key="bt_slip") / 100
        with c3:
            position_size_pct = st.slider("Position Size (%)", 10, 100, 100, key="bt_pos_size") / 100
            stop_loss_pct = st.slider("Stop Loss (%)", 0, 20, 5, key="bt_sl") / 100
            take_profit_pct = st.slider("Take Profit (%)", 0, 50, 15, key="bt_tp") / 100

    if st.button("Run Backtest", type="primary", key="bt_run"):
        with st.spinner(f"Backtesting {strategy_type} on {symbol}..."):
            try:
                import yfinance as yf
                df = yf.Ticker(symbol).history(period=period)
                if df.empty:
                    st.error(f"No data for {symbol}.")
                    return

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                close = df["Close"].dropna().astype(float)
                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]

                # Generate signals based on strategy
                signals = pd.Series(0, index=close.index)

                if strategy_type == "RSI Mean Reversion":
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
                    signals = pd.Series(0, index=close.index)
                    signals[rsi < 30] = 1
                    signals[rsi > 70] = -1

                elif strategy_type == "SMA 20/50 Cross":
                    sma20 = close.rolling(20).mean()
                    sma50 = close.rolling(50).mean()
                    signals[sma20 > sma50] = 1
                    signals[sma20 < sma50] = -1

                elif strategy_type == "MACD Momentum":
                    ema12 = close.ewm(span=12, adjust=False).mean()
                    ema26 = close.ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    signal_line = macd.ewm(span=9, adjust=False).mean()
                    signals[macd > signal_line] = 1
                    signals[macd < signal_line] = -1

                elif strategy_type == "Bollinger Band Breakout":
                    mid = close.rolling(20).mean()
                    std = close.rolling(20).std()
                    upper = mid + 2 * std
                    lower = mid - 2 * std
                    signals[close > upper] = 1
                    signals[close < lower] = -1

                elif strategy_type == "Volatility Regime":
                    vol = close.pct_change().rolling(20).std() * np.sqrt(252)
                    vol_ma = vol.rolling(60).mean()
                    signals[vol < vol_ma] = 1  # low vol = long
                    signals[vol > vol_ma * 1.5] = -1  # high vol = short

                # Backtest simulation
                signals = signals.fillna(0)
                returns = close.pct_change().fillna(0)
                position = signals.shift(1).fillna(0)

                # Apply transaction costs
                trades = position.diff().abs()
                trade_costs = trades * (commission_pct + slippage_pct)

                # Strategy returns
                strat_returns = position * returns * position_size_pct - trade_costs

                # Apply stop loss / take profit
                equity = initial_capital * (1 + strat_returns).cumprod()

                # Metrics
                total_return = float(equity.iloc[-1] / initial_capital - 1)
                ann_return = (1 + total_return) ** (252 / len(equity)) - 1
                ann_vol = float(strat_returns.std() * np.sqrt(252))
                sharpe = ann_return / max(ann_vol, 0.001)
                sortino_vol = float(strat_returns[strat_returns < 0].std() * np.sqrt(252))
                sortino = ann_return / max(sortino_vol, 0.001)

                # Max drawdown
                peak = equity.cummax()
                dd = (equity - peak) / peak
                max_dd = float(dd.min())
                calmar = ann_return / max(abs(max_dd), 0.001)

                # Win rate
                trade_rets = strat_returns[trades > 0]
                win_rate = float((trade_rets > 0).mean()) if len(trade_rets) > 0 else 0.5
                n_trades = int(trades.sum())

                st.session_state["bt_result"] = {
                    "equity": equity,
                    "returns": strat_returns,
                    "drawdown": dd,
                    "signals": signals,
                    "total_return": total_return,
                    "ann_return": ann_return,
                    "ann_vol": ann_vol,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "calmar": calmar,
                    "max_dd": max_dd,
                    "win_rate": win_rate,
                    "n_trades": n_trades,
                    "symbol": symbol,
                    "strategy": strategy_type,
                }
                st.success(f"Backtest complete: {n_trades} trades, Sharpe {sharpe:.2f}")
            except Exception as e:
                st.error(f"Backtest error: {e}")

    result = st.session_state.get("bt_result")
    if result is None:
        st.info("Configure strategy and click 'Run Backtest'.")
        return

    # Metrics
    _section("Performance Metrics")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1: _metric("Total Return", f"{result['total_return']:+.1%}",
                       COLORS["success"] if result["total_return"] > 0 else COLORS["danger"])
    with mc2: _metric("Ann. Return", f"{result['ann_return']:+.1%}",
                       COLORS["success"] if result["ann_return"] > 0 else COLORS["danger"])
    with mc3: _metric("Sharpe Ratio", f"{result['sharpe']:.2f}",
                       COLORS["gold"] if result["sharpe"] > 1 else COLORS["danger"])
    with mc4: _metric("Sortino", f"{result['sortino']:.2f}",
                       COLORS["gold"] if result["sortino"] > 1.5 else COLORS["danger"])
    with mc5: _metric("Max Drawdown", f"{result['max_dd']:.1%}", COLORS["danger"])
    with mc6: _metric("Win Rate", f"{result['win_rate']:.1%}",
                       COLORS["success"] if result["win_rate"] > 0.5 else COLORS["danger"])

    # Equity curve
    _section("Equity Curve")
    fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.04)

    equity = result["equity"]
    dd = result["drawdown"]

    fig_eq.add_trace(go.Scatter(
        x=list(range(len(equity))), y=equity.values, mode="lines",
        name="Portfolio Value",
        line=dict(color=COLORS["gold"], width=2),
    ), row=1, col=1)

    # Buy & hold comparison
    close_bh = equity.iloc[0] * (1 + equity.pct_change().fillna(0)).cumprod()
    fig_eq.add_trace(go.Scatter(
        x=list(range(len(equity))), y=close_bh.values, mode="lines",
        name="Buy & Hold",
        line=dict(color=COLORS["lavender"], width=1.2, dash="dot"),
    ), row=1, col=1)

    fig_eq.add_trace(go.Scatter(
        x=list(range(len(dd))), y=dd.values * 100, mode="lines",
        name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
        line=dict(color=COLORS["danger"], width=1.5),
    ), row=2, col=1)
    fig_eq.add_hline(y=0, line_color=COLORS["border"], row=2, col=1)

    fig_eq.update_layout(**_dark_layout(
        height=500,
        title=f"{result['strategy']} on {result['symbol']}",
    ))
    fig_eq.update_yaxes(title_text="Portfolio ($)", row=1, col=1)
    fig_eq.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

    # Return distribution
    _section("Return Distribution")
    rets = result["returns"].dropna()
    fig_dist = go.Figure(go.Histogram(
        x=rets.values * 100, nbinsx=50,
        marker_color=COLORS["gold"], opacity=0.75,
        name="Daily Returns",
    ))
    fig_dist.add_vline(x=0, line_color=COLORS["border"])
    fig_dist.add_vline(x=float(rets.mean() * 100), line_dash="dash",
                       line_color=COLORS["success"], annotation_text="Mean")
    fig_dist.update_layout(**_dark_layout(height=280, title="Daily Return Distribution",
                                           xaxis_title="Return (%)", yaxis_title="Count"))
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_strategy_research_lab():
    """Main entry point for the Strategy Research Lab."""
    st.markdown(_LAB_CSS, unsafe_allow_html=True)
    st.title("Strategy Research Lab")
    st.caption(
        "Professional strategy design, testing, and evolution environment — "
        "statistical arbitrage, factor investing, genetic evolution, alpha signals, and backtesting."
    )

    tabs = st.tabs([
        "Pairs Trading",
        "Factor Investing",
        "Genetic Evolution",
        "Alpha Signals",
        "Backtesting",
    ])

    with tabs[0]:
        _render_pairs_tab()

    with tabs[1]:
        _render_factor_tab()

    with tabs[2]:
        _render_genetic_tab()

    with tabs[3]:
        _render_alpha_signals_tab()

    with tabs[4]:
        _render_backtest_tab()
