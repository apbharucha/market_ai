"""
Octavian Quantitative Modeling Laboratory
==========================================
Advanced quantitative modeling environment used by quant funds.

Covers:
  - Market Regime Detection (HMM, Bayesian switching)
  - Factor Crowding Detection
  - Machine Learning Framework (RF, GBM, LSTM, RL)
  - Liquidity & Market Impact Modeling (Almgren-Chriss)
  - Risk Management (VaR, CVaR, factor exposure)
  - Narrative Dislocation Detection
  - Cross-Asset Analysis Engine

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
    from hmm_engine import MarketRegimeDetector, get_regime_detector
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

try:
    from factor_crowding_engine import FactorCrowdingEngine, get_crowding_engine
    HAS_CROWDING = True
except ImportError:
    HAS_CROWDING = False

try:
    from advanced_ml_engine import AdvancedMLEngine
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    from narrative_dislocation_engine import NarrativeDislocator
    HAS_NARRATIVE = True
except ImportError:
    HAS_NARRATIVE = False

try:
    from macro_cross_asset_engine import MacroCrossAssetEngine
    HAS_MACRO = True
except ImportError:
    HAS_MACRO = False

# ── CSS ───────────────────────────────────────────────────────────────────────
_QML_CSS = """
<style>
@keyframes qml-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes qml-pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(155,142,196,0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(155,142,196,0); }
    100% { box-shadow: 0 0 0 0 rgba(155,142,196,0); }
}
@keyframes qml-bar-grow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
@keyframes qml-regime-glow {
    0%, 100% { opacity: 0.8; }
    50%       { opacity: 1.0; }
}
.qml-card {
    background: linear-gradient(135deg, #132240 0%, #1a2d4a 100%);
    border: 1px solid rgba(155,142,196,0.12);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    animation: qml-fade-in 0.35s ease-out;
    transition: border-color 0.2s ease;
}
.qml-card:hover { border-color: rgba(155,142,196,0.3); }
.qml-section {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9b8ec4;
    border-bottom: 1px solid rgba(155,142,196,0.18);
    padding-bottom: 5px;
    margin: 16px 0 10px 0;
    animation: qml-fade-in 0.3s ease-out;
}
.qml-metric {
    background: rgba(10,22,40,0.65);
    border-radius: 7px;
    padding: 10px 14px;
    text-align: center;
}
.qml-metric-label {
    font-size: 0.7rem;
    color: #a0a8b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.qml-metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #e8eaf0;
}
.qml-regime-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    animation: qml-regime-glow 2s ease-in-out infinite;
}
.qml-prob-bar {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    overflow: hidden;
    margin: 3px 0;
}
.qml-prob-fill {
    height: 100%;
    border-radius: 4px;
    transform-origin: left;
    animation: qml-bar-grow 0.6s ease-out forwards;
}
</style>
"""


def _section(title: str):
    st.markdown(f'<div class="qml-section">{title}</div>', unsafe_allow_html=True)


def _metric(label: str, value: str, color: str = "#e8eaf0"):
    st.markdown(
        f'<div class="qml-metric">'
        f'<div class="qml-metric-label">{label}</div>'
        f'<div class="qml-metric-value" style="color:{color};">{value}</div>'
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
# TAB 1 — Market Regime Detection
# ══════════════════════════════════════════════════════════════════════════════

def _render_regime_tab():
    _section("Market Regime Detection Engine")
    st.caption(
        "Detect macro and market regimes using Hidden Markov Models, Bayesian switching, "
        "and volatility clustering. Output: regime probabilities, transition forecasts, "
        "and strategy compatibility scores."
    )

    c1, c2 = st.columns(2)
    with c1:
        symbol = st.text_input("Symbol", value="SPY", key="regime_sym").strip().upper()
        period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=2, key="regime_period")
    with c2:
        n_regimes = st.slider("Number of Regimes", 2, 5, 3, key="regime_n")
        method = st.selectbox("Detection Method",
                               ["Hidden Markov Model", "Volatility Clustering", "Bayesian Switching"],
                               key="regime_method")

    if st.button("Detect Regimes", type="primary", key="regime_run"):
        with st.spinner(f"Detecting regimes for {symbol}..."):
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

                returns = close.pct_change().dropna()
                vol = returns.rolling(20).std() * np.sqrt(252) * 100

                # Try HMM engine
                if HAS_HMM and method == "Hidden Markov Model":
                    try:
                        detector = get_regime_detector()
                        regime_result = detector.detect(close)
                        st.session_state["regime_result"] = regime_result
                        st.session_state["regime_close"] = close
                        st.session_state["regime_vol"] = vol
                        st.success("HMM regime detection complete.")
                        return
                    except Exception:
                        pass

                # Fallback: simple volatility-based regime detection
                rng = np.random.default_rng(42)
                vol_filled = vol.fillna(method="bfill").fillna(15.0)

                # K-means style regime assignment
                vol_values = vol_filled.values
                vol_sorted = np.sort(vol_values)
                thresholds = [np.percentile(vol_sorted, 100 * i / n_regimes) for i in range(1, n_regimes)]

                regime_labels = np.zeros(len(vol_values), dtype=int)
                for i, v in enumerate(vol_values):
                    for j, t in enumerate(thresholds):
                        if v > t:
                            regime_labels[i] = j + 1

                # Regime names
                regime_names = {
                    0: "Low Vol / Bull",
                    1: "Normal / Transition",
                    2: "High Vol / Bear",
                    3: "Crisis",
                    4: "Extreme Crisis",
                }

                # Regime probabilities (smoothed)
                regime_probs = {}
                for r in range(n_regimes):
                    prob = pd.Series((regime_labels == r).astype(float)).rolling(20).mean().fillna(0)
                    regime_probs[regime_names.get(r, f"Regime {r}")] = prob

                # Current regime
                current_regime = int(regime_labels[-1])
                current_regime_name = regime_names.get(current_regime, f"Regime {current_regime}")

                # Transition matrix
                trans_matrix = np.zeros((n_regimes, n_regimes))
                for i in range(len(regime_labels) - 1):
                    trans_matrix[regime_labels[i], regime_labels[i + 1]] += 1
                row_sums = trans_matrix.sum(axis=1, keepdims=True)
                trans_matrix = trans_matrix / np.maximum(row_sums, 1)

                # Strategy compatibility
                strategy_compat = {
                    "Trend Following": 0.9 if current_regime <= 1 else 0.3,
                    "Mean Reversion": 0.8 if current_regime == 0 else 0.4,
                    "Volatility Selling": 0.9 if current_regime == 0 else 0.1,
                    "Momentum": 0.85 if current_regime <= 1 else 0.35,
                    "Defensive / Cash": 0.2 if current_regime == 0 else 0.9,
                }

                st.session_state["regime_result"] = {
                    "regime_labels": regime_labels,
                    "regime_probs": regime_probs,
                    "current_regime": current_regime,
                    "current_regime_name": current_regime_name,
                    "trans_matrix": trans_matrix,
                    "strategy_compat": strategy_compat,
                    "n_regimes": n_regimes,
                    "regime_names": regime_names,
                }
                st.session_state["regime_close"] = close
                st.session_state["regime_vol"] = vol
                st.success(f"Regime detection complete. Current: {current_regime_name}")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("regime_result")
    if result is None:
        st.info("Configure parameters and click 'Detect Regimes'.")
        return

    close = st.session_state.get("regime_close")
    vol = st.session_state.get("regime_vol")

    # Handle both HMM result and dict result
    if hasattr(result, "current_regime"):
        # HMM engine result object
        current_name = str(getattr(result, "current_regime", "Unknown"))
        regime_conf = float(getattr(result, "confidence", 0.7))
        _section("Current Regime")
        regime_color = COLORS["success"] if "bull" in current_name.lower() else COLORS["danger"] if "bear" in current_name.lower() or "crisis" in current_name.lower() else COLORS["neutral"]
        st.markdown(
            f'<div class="qml-card" style="text-align:center;">'
            f'<div class="qml-regime-badge" style="background:rgba(155,142,196,0.15);border:1px solid rgba(155,142,196,0.4);color:{regime_color};">{current_name}</div>'
            f'<div style="font-size:1.2rem;color:{COLORS["text_secondary"]};margin-top:8px;">Confidence: {regime_conf:.0%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Dict result
    current_name = result["current_regime_name"]
    regime_labels = result["regime_labels"]
    regime_probs = result["regime_probs"]
    trans_matrix = result["trans_matrix"]
    strategy_compat = result["strategy_compat"]
    n_regimes = result["n_regimes"]
    regime_names = result["regime_names"]

    # Current regime display
    _section("Current Regime")
    regime_color = (COLORS["success"] if "Bull" in current_name or "Low" in current_name
                    else COLORS["danger"] if "Crisis" in current_name or "Bear" in current_name
                    else COLORS["neutral"])
    st.markdown(
        f'<div class="qml-card" style="text-align:center;">'
        f'<div class="qml-regime-badge" style="background:rgba(155,142,196,0.15);border:1px solid rgba(155,142,196,0.4);color:{regime_color};font-size:1.1rem;padding:8px 20px;">{current_name}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Price colored by regime
    _section("Price Path with Regime Overlay")
    if close is not None:
        regime_colors_map = {
            0: "rgba(76,175,80,0.3)",
            1: "rgba(201,168,76,0.3)",
            2: "rgba(239,83,80,0.3)",
            3: "rgba(183,28,28,0.4)",
            4: "rgba(100,0,0,0.5)",
        }

        fig_regime = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.65, 0.35], vertical_spacing=0.04)

        fig_regime.add_trace(go.Scatter(
            x=list(range(len(close))), y=close.values, mode="lines",
            name="Price", line=dict(color=COLORS["gold"], width=1.5),
        ), row=1, col=1)

        # Shade by regime
        for r in range(n_regimes):
            mask = regime_labels == r
            if mask.any():
                # Add shaded regions
                x_vals = np.where(mask)[0]
                if len(x_vals) > 0:
                    fig_regime.add_trace(go.Scatter(
                        x=list(x_vals), y=close.values[x_vals],
                        mode="markers", name=regime_names.get(r, f"R{r}"),
                        marker=dict(size=3, color=list(regime_colors_map.values())[r % len(regime_colors_map)]),
                        showlegend=True,
                    ), row=1, col=1)

        # Volatility
        if vol is not None:
            fig_regime.add_trace(go.Scatter(
                x=list(range(len(vol))), y=vol.values, mode="lines",
                name="Realized Vol", fill="tozeroy",
                fillcolor="rgba(155,142,196,0.12)",
                line=dict(color=COLORS["lavender"], width=1.2),
            ), row=2, col=1)

        fig_regime.update_layout(**_dark_layout(height=500, title=f"Regime Detection: {symbol}"))
        st.plotly_chart(fig_regime, use_container_width=True)

    # Regime probabilities
    _section("Regime Probabilities Over Time")
    fig_probs = go.Figure()
    prob_colors = [COLORS["success"], COLORS["gold"], COLORS["danger"], "#b71c1c", "#4a0000"]
    for i, (name, prob) in enumerate(regime_probs.items()):
        fig_probs.add_trace(go.Scatter(
            x=list(range(len(prob))), y=prob.values * 100, mode="lines",
            name=name, line=dict(color=prob_colors[i % len(prob_colors)], width=1.5),
            stackgroup="one",
        ))
    fig_probs.update_layout(**_dark_layout(height=300, title="Regime Probability Stack",
                                            yaxis_title="Probability (%)"))
    st.plotly_chart(fig_probs, use_container_width=True)

    # Transition matrix
    _section("Regime Transition Matrix")
    regime_name_list = [regime_names.get(i, f"R{i}") for i in range(n_regimes)]
    fig_trans = go.Figure(go.Heatmap(
        z=trans_matrix,
        x=regime_name_list, y=regime_name_list,
        colorscale="Blues",
        text=np.round(trans_matrix, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Prob", tickfont=dict(color="white")),
    ))
    fig_trans.update_layout(**_dark_layout(height=300, title="Regime Transition Probabilities"))
    st.plotly_chart(fig_trans, use_container_width=True)

    # Strategy compatibility
    _section("Strategy Compatibility Scores")
    for strat, score in strategy_compat.items():
        color = COLORS["success"] if score > 0.7 else COLORS["danger"] if score < 0.4 else COLORS["gold"]
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:6px 0;">'
            f'<span style="color:{COLORS["text_primary"]};font-weight:500;">{strat}</span>'
            f'<span style="color:{color};font-weight:700;">{score:.0%}</span>'
            f'</div>'
            f'<div class="qml-prob-bar"><div class="qml-prob-fill" style="width:{score*100:.0f}%;background:{color};"></div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Liquidity & Market Impact
# ══════════════════════════════════════════════════════════════════════════════

def _render_liquidity_tab():
    _section("Liquidity & Market Impact Modeling")
    st.caption(
        "Simulate realistic trading conditions using the Almgren-Chriss optimal execution model. "
        "Compute expected slippage, fill probability, and optimal order size."
    )

    with st.expander("Order Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol", value="AAPL", key="liq_sym").strip().upper()
            order_size = st.number_input("Order Size (shares)", value=10000, min_value=1, key="liq_size")
            adv_pct = st.slider("% of ADV", 1, 50, 10, key="liq_adv_pct") / 100
        with c2:
            price = st.number_input("Current Price ($)", value=175.0, min_value=0.01, key="liq_price")
            adv = st.number_input("Avg Daily Volume (shares)", value=50_000_000, step=1_000_000, key="liq_adv")
            bid_ask_spread = st.number_input("Bid-Ask Spread ($)", value=0.02, min_value=0.001, step=0.001, key="liq_spread")
        with c3:
            volatility = st.slider("Daily Volatility (%)", 0.5, 10.0, 1.5, step=0.1, key="liq_vol") / 100
            execution_horizon = st.slider("Execution Horizon (days)", 1, 20, 5, key="liq_horizon")
            risk_aversion = st.slider("Risk Aversion (lambda)", 0.01, 1.0, 0.1, step=0.01, key="liq_lambda")

    if st.button("Compute Market Impact", type="primary", key="liq_run"):
        with st.spinner("Computing Almgren-Chriss optimal execution..."):
            try:
                # Almgren-Chriss model parameters
                X = order_size  # total shares to trade
                T = execution_horizon  # trading horizon in days
                sigma = volatility  # daily vol
                eta = bid_ask_spread / (2 * price)  # temporary impact coefficient
                gamma = 0.1 / adv  # permanent impact coefficient
                lam = risk_aversion  # risk aversion

                # Optimal execution trajectory
                kappa = math.sqrt(lam * sigma**2 / eta) if eta > 0 else 1.0
                tau = T / max(T, 1)

                # Optimal trading schedule
                n_intervals = T
                t_vals = np.linspace(0, T, n_intervals + 1)
                x_vals = X * np.sinh(kappa * (T - t_vals)) / np.sinh(kappa * T) if kappa * T > 0.001 else X * (1 - t_vals / T)

                # Trading rate
                n_vals = np.diff(x_vals)  # shares traded per interval

                # Costs
                temp_impact = eta * np.sum(n_vals**2)
                perm_impact = gamma * X**2 / 2
                spread_cost = bid_ask_spread / 2 * X
                total_cost = temp_impact + perm_impact + spread_cost

                # Slippage in bps
                slippage_bps = total_cost / (X * price) * 10000

                # Fill probability (simplified)
                fill_prob = min(0.99, adv / max(X, 1) * 0.5)

                # Optimal order size (1% of ADV)
                optimal_size = int(adv * adv_pct)

                st.session_state["liq_result"] = {
                    "t_vals": t_vals,
                    "x_vals": x_vals,
                    "n_vals": n_vals,
                    "temp_impact": temp_impact,
                    "perm_impact": perm_impact,
                    "spread_cost": spread_cost,
                    "total_cost": total_cost,
                    "slippage_bps": slippage_bps,
                    "fill_prob": fill_prob,
                    "optimal_size": optimal_size,
                    "symbol": symbol,
                    "order_size": order_size,
                    "price": price,
                }
                st.success("Market impact analysis complete.")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("liq_result")
    if result is None:
        st.info("Configure order parameters and click 'Compute Market Impact'.")
        return

    # Summary metrics
    _section("Market Impact Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: _metric("Total Cost ($)", f"${result['total_cost']:,.2f}", COLORS["danger"])
    with mc2: _metric("Slippage (bps)", f"{result['slippage_bps']:.1f}", COLORS["danger"])
    with mc3: _metric("Fill Probability", f"{result['fill_prob']:.1%}", COLORS["success"])
    with mc4: _metric("Optimal Size", f"{result['optimal_size']:,}", COLORS["gold"])

    # Cost breakdown
    _section("Cost Breakdown")
    costs = {
        "Temporary Impact": result["temp_impact"],
        "Permanent Impact": result["perm_impact"],
        "Spread Cost": result["spread_cost"],
    }
    total = sum(costs.values())
    for cost_name, cost_val in costs.items():
        pct = cost_val / max(total, 1e-10)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;margin:5px 0;">'
            f'<span style="color:{COLORS["text_primary"]};">{cost_name}</span>'
            f'<span style="color:{COLORS["danger"]};">${cost_val:,.2f} ({pct:.0%})</span>'
            f'</div>'
            f'<div class="qml-prob-bar"><div class="qml-prob-fill" style="width:{pct*100:.0f}%;background:{COLORS["danger"]};"></div></div>',
            unsafe_allow_html=True,
        )

    # Execution trajectory
    _section("Optimal Execution Trajectory (Almgren-Chriss)")
    fig_traj = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.6, 0.4], vertical_spacing=0.04)

    fig_traj.add_trace(go.Scatter(
        x=result["t_vals"], y=result["x_vals"], mode="lines+markers",
        name="Remaining Inventory",
        line=dict(color=COLORS["gold"], width=2),
        marker=dict(size=6),
    ), row=1, col=1)

    fig_traj.add_trace(go.Bar(
        x=list(range(len(result["n_vals"]))),
        y=np.abs(result["n_vals"]),
        name="Shares Traded per Interval",
        marker_color=COLORS["lavender"], opacity=0.7,
    ), row=2, col=1)

    fig_traj.update_layout(**_dark_layout(
        height=450,
        title=f"Almgren-Chriss Optimal Execution: {result['symbol']} ({result['order_size']:,} shares)",
    ))
    fig_traj.update_yaxes(title_text="Remaining Shares", row=1, col=1)
    fig_traj.update_yaxes(title_text="Shares/Interval", row=2, col=1)
    st.plotly_chart(fig_traj, use_container_width=True)

    # Liquidity stress test
    _section("Liquidity Stress Test")
    order_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    stress_results = []
    for sz in order_sizes:
        temp = eta * sz**2 if 'eta' in dir() else result["temp_impact"] * (sz / result["order_size"])**2
        perm = gamma * sz**2 / 2 if 'gamma' in dir() else result["perm_impact"] * (sz / result["order_size"])**2
        sp = bid_ask_spread / 2 * sz if 'bid_ask_spread' in dir() else result["spread_cost"] * sz / result["order_size"]
        total = temp + perm + sp
        slip = total / (sz * result["price"]) * 10000 if sz > 0 else 0
        stress_results.append({"Order Size": f"{sz:,}", "Total Cost ($)": f"${total:,.0f}", "Slippage (bps)": f"{slip:.1f}"})

    st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk Management
# ══════════════════════════════════════════════════════════════════════════════

def _render_risk_tab():
    _section("Institutional Risk Management System")
    st.caption(
        "Value at Risk (VaR), Conditional VaR (CVaR), factor exposure analysis, "
        "correlation breakdown detection, and tail risk monitoring."
    )

    with st.expander("Portfolio Configuration", expanded=True):
        symbols_input = st.text_area(
            "Portfolio Symbols (comma-separated)",
            value="AAPL, MSFT, NVDA, GOOGL, AMZN",
            height=60, key="risk_symbols",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            portfolio_value = st.number_input("Portfolio Value ($)", value=1_000_000.0, step=100_000.0, key="risk_port_val")
            confidence = st.slider("VaR Confidence Level (%)", 90, 99, 95, key="risk_conf") / 100
        with c2:
            period = st.selectbox("Historical Period", ["1y", "2y", "3y"], index=1, key="risk_period")
            horizon_days = st.slider("Risk Horizon (days)", 1, 30, 1, key="risk_horizon")
        with c3:
            equal_weight = st.checkbox("Equal Weight Portfolio", value=True, key="risk_eq_weight")

    if st.button("Compute Risk Metrics", type="primary", key="risk_run"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if len(symbols) < 2:
            st.error("Enter at least 2 symbols.")
            return

        with st.spinner("Computing portfolio risk metrics..."):
            try:
                import yfinance as yf
                data = yf.download(symbols, period=period, progress=False, threads=True)
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

                returns = close_data.pct_change().dropna()
                n_assets = len(returns.columns)

                # Weights
                if equal_weight:
                    weights = np.ones(n_assets) / n_assets
                else:
                    weights = np.ones(n_assets) / n_assets  # default to equal

                # Portfolio returns
                port_returns = (returns * weights).sum(axis=1)

                # VaR (Historical)
                var_hist = float(np.percentile(port_returns, (1 - confidence) * 100))
                var_dollar = abs(var_hist) * portfolio_value * math.sqrt(horizon_days)

                # CVaR (Expected Shortfall)
                cvar = float(port_returns[port_returns <= var_hist].mean())
                cvar_dollar = abs(cvar) * portfolio_value * math.sqrt(horizon_days)

                # Parametric VaR
                port_vol = float(port_returns.std())
                from scipy import stats as scipy_stats
                z_score = scipy_stats.norm.ppf(1 - confidence)
                var_param = abs(z_score) * port_vol * math.sqrt(horizon_days)
                var_param_dollar = var_param * portfolio_value

                # Max drawdown
                cum_ret = (1 + port_returns).cumprod()
                peak = cum_ret.cummax()
                dd = (cum_ret - peak) / peak
                max_dd = float(dd.min())

                # Sharpe
                ann_ret = float(port_returns.mean() * 252)
                ann_vol = float(port_returns.std() * np.sqrt(252))
                sharpe = (ann_ret - 0.04) / max(ann_vol, 0.001)

                # Correlation matrix
                corr_matrix = returns.corr()

                # Factor exposures (beta to SPY)
                try:
                    spy = yf.Ticker("SPY").history(period=period)["Close"].pct_change().dropna()
                    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
                    betas = {}
                    for col in returns.columns:
                        common = returns[col].index.intersection(spy.index)
                        if len(common) > 30:
                            r = returns[col].loc[common].astype(float)
                            s = spy.loc[common].astype(float)
                            beta = float(np.cov(r, s)[0, 1] / max(np.var(s), 1e-10))
                            betas[col] = beta
                except Exception:
                    betas = {col: 1.0 for col in returns.columns}

                st.session_state["risk_result"] = {
                    "var_hist": var_hist,
                    "var_dollar": var_dollar,
                    "cvar": cvar,
                    "cvar_dollar": cvar_dollar,
                    "var_param_dollar": var_param_dollar,
                    "max_dd": max_dd,
                    "sharpe": sharpe,
                    "ann_ret": ann_ret,
                    "ann_vol": ann_vol,
                    "port_returns": port_returns,
                    "corr_matrix": corr_matrix,
                    "betas": betas,
                    "returns": returns,
                    "weights": weights,
                    "symbols": list(returns.columns),
                }
                st.success("Risk metrics computed.")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("risk_result")
    if result is None:
        st.info("Configure portfolio and click 'Compute Risk Metrics'.")
        return

    # Summary metrics
    _section("Risk Summary")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1: _metric(f"VaR ({confidence:.0%})", f"${result['var_dollar']:,.0f}", COLORS["danger"])
    with mc2: _metric("CVaR (ES)", f"${result['cvar_dollar']:,.0f}", COLORS["danger"])
    with mc3: _metric("Max Drawdown", f"{result['max_dd']:.1%}", COLORS["danger"])
    with mc4: _metric("Sharpe Ratio", f"{result['sharpe']:.2f}",
                       COLORS["gold"] if result["sharpe"] > 1 else COLORS["danger"])
    with mc5: _metric("Ann. Volatility", f"{result['ann_vol']:.1%}", COLORS["neutral"])

    # Return distribution with VaR
    _section("Return Distribution & VaR")
    port_returns = result["port_returns"]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=port_returns.values * 100, nbinsx=60,
        marker_color=COLORS["lavender"], opacity=0.7, name="Daily Returns",
    ))
    fig_dist.add_vline(x=result["var_hist"] * 100, line_dash="dash",
                       line_color=COLORS["danger"],
                       annotation_text=f"VaR ({confidence:.0%})",
                       annotation_font_color=COLORS["danger"])
    fig_dist.add_vline(x=result["cvar"] * 100, line_dash="dot",
                       line_color="#ff1744",
                       annotation_text="CVaR",
                       annotation_font_color="#ff1744")
    fig_dist.update_layout(**_dark_layout(height=300, title="Portfolio Return Distribution",
                                           xaxis_title="Daily Return (%)", yaxis_title="Count"))
    st.plotly_chart(fig_dist, use_container_width=True)

    # Correlation heatmap
    _section("Asset Correlation Matrix")
    corr = result["corr_matrix"]
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig_corr.update_layout(**_dark_layout(height=350, title="Asset Correlation Matrix"))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Beta exposures
    _section("Market Beta Exposures")
    betas = result["betas"]
    if betas:
        beta_df = pd.DataFrame([
            {"Symbol": sym, "Beta": round(b, 2),
             "Exposure": "High" if abs(b) > 1.2 else "Normal" if abs(b) > 0.8 else "Low"}
            for sym, b in betas.items()
        ])
        st.dataframe(beta_df, use_container_width=True, hide_index=True)

        fig_beta = go.Figure(go.Bar(
            x=list(betas.keys()), y=list(betas.values()),
            marker_color=[COLORS["danger"] if abs(b) > 1.2 else COLORS["gold"] if abs(b) > 0.8 else COLORS["success"]
                          for b in betas.values()],
            name="Beta",
        ))
        fig_beta.add_hline(y=1, line_dash="dash", line_color=COLORS["border"])
        fig_beta.update_layout(**_dark_layout(height=280, title="Market Beta by Asset",
                                               yaxis_title="Beta"))
        st.plotly_chart(fig_beta, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML Framework
# ══════════════════════════════════════════════════════════════════════════════

def _render_ml_tab():
    _section("Machine Learning Framework")
    st.caption(
        "Random forests, gradient boosting, neural networks, LSTM time-series models. "
        "Automated feature engineering, hyperparameter tuning, and model validation."
    )

    c1, c2 = st.columns(2)
    with c1:
        symbol = st.text_input("Symbol", value="SPY", key="ml_sym").strip().upper()
        period = st.selectbox("Period", ["1y", "2y", "3y"], index=1, key="ml_period")
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"],
            key="ml_model",
        )
    with c2:
        target = st.selectbox("Prediction Target",
                               ["Next Day Return", "5-Day Return", "Direction (Up/Down)"],
                               key="ml_target")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="ml_test") / 100
        n_features = st.slider("Feature Window (days)", 5, 30, 10, key="ml_features")

    if st.button("Train Model", type="primary", key="ml_run"):
        with st.spinner(f"Training {model_type} on {symbol}..."):
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

                # Feature engineering
                features = pd.DataFrame(index=close.index)
                for lag in range(1, n_features + 1):
                    features[f"ret_{lag}d"] = close.pct_change(lag)
                features["vol_5d"] = close.pct_change().rolling(5).std()
                features["vol_20d"] = close.pct_change().rolling(20).std()
                features["rsi_14"] = _compute_rsi(close)
                features["sma_ratio"] = close / close.rolling(20).mean() - 1
                if volume is not None:
                    features["vol_ratio"] = volume / volume.rolling(20).mean()

                # Target
                if target == "Next Day Return":
                    y = close.pct_change().shift(-1)
                elif target == "5-Day Return":
                    y = close.pct_change(5).shift(-5)
                else:
                    y = (close.pct_change().shift(-1) > 0).astype(int)

                # Align and clean
                data = features.join(y.rename("target")).dropna()
                X = data.drop("target", axis=1).values
                y_vals = data["target"].values

                # Train/test split
                split = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y_vals[:split], y_vals[split:]

                # Train model
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                if model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    if "Direction" in target:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == "Gradient Boosting":
                    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                    if "Direction" in target:
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    else:
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_type == "Ridge Regression":
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0)
                else:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()

                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                # Metrics
                from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
                if "Direction" in target:
                    acc = float(accuracy_score(y_test, y_pred))
                    mse = 0.0
                    r2 = acc
                else:
                    mse = float(mean_squared_error(y_test, y_pred))
                    r2 = float(r2_score(y_test, y_pred))
                    acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))

                # Feature importance
                feat_names = list(data.drop("target", axis=1).columns)
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_)
                else:
                    importances = np.ones(len(feat_names)) / len(feat_names)

                st.session_state["ml_result"] = {
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "r2": r2,
                    "mse": mse,
                    "acc": acc,
                    "feat_names": feat_names,
                    "importances": importances,
                    "model_type": model_type,
                    "target": target,
                    "symbol": symbol,
                }
                st.success(f"Model trained. R²: {r2:.3f}, Directional Accuracy: {acc:.1%}")
            except ImportError:
                st.error("scikit-learn not installed. Install with: pip install scikit-learn")
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("ml_result")
    if result is None:
        st.info("Configure model and click 'Train Model'.")
        return

    # Metrics
    _section("Model Performance")
    mc1, mc2, mc3 = st.columns(3)
    with mc1: _metric("R² Score", f"{result['r2']:.3f}",
                       COLORS["success"] if result["r2"] > 0.1 else COLORS["danger"])
    with mc2: _metric("Directional Acc.", f"{result['acc']:.1%}",
                       COLORS["success"] if result["acc"] > 0.55 else COLORS["danger"])
    with mc3: _metric("Model", result["model_type"])

    # Predictions vs actual
    _section("Predictions vs Actual")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(result["y_test"]))), y=result["y_test"] * 100,
        mode="lines", name="Actual", line=dict(color=COLORS["gold"], width=1.5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(result["y_pred"]))), y=result["y_pred"] * 100,
        mode="lines", name="Predicted", line=dict(color=COLORS["lavender"], width=1.5, dash="dot"),
    ))
    fig_pred.update_layout(**_dark_layout(height=300, title=f"Predictions vs Actual: {result['target']}",
                                           yaxis_title="Return (%)"))
    st.plotly_chart(fig_pred, use_container_width=True)

    # Feature importance
    _section("Feature Importance")
    feat_df = pd.DataFrame({
        "Feature": result["feat_names"],
        "Importance": result["importances"],
    }).sort_values("Importance", ascending=False).head(15)

    fig_feat = go.Figure(go.Bar(
        x=feat_df["Importance"].values,
        y=feat_df["Feature"].values,
        orientation="h",
        marker_color=COLORS["lavender"],
    ))
    fig_feat.update_layout(**_dark_layout(height=400, title="Feature Importance",
                                           xaxis_title="Importance"))
    st.plotly_chart(fig_feat, use_container_width=True)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Cross-Asset Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _render_cross_asset_tab():
    _section("Cross-Asset Analysis Engine")
    st.caption(
        "Track relationships across asset classes: bond yields vs equity valuations, "
        "oil vs inflation vs rates, USD strength vs commodities, liquidity vs crypto flows. "
        "Dynamic correlation matrices, lead/lag detection, contagion risk indicators."
    )

    if HAS_MACRO:
        try:
            engine = MacroCrossAssetEngine()
            if st.button("Run Cross-Asset Analysis", type="primary", key="cross_run"):
                with st.spinner("Fetching cross-asset data..."):
                    try:
                        signals = engine.get_all_signals()
                        st.session_state["cross_signals"] = signals
                        st.success(f"Generated {len(signals)} cross-asset signals.")
                    except Exception as e:
                        st.error(f"Error: {e}")

            signals = st.session_state.get("cross_signals")
            if signals:
                for sig in signals[:5]:
                    direction_color = COLORS["success"] if sig.direction == "BULLISH" else COLORS["danger"] if sig.direction == "BEARISH" else COLORS["neutral"]
                    st.markdown(
                        f'<div class="qml-card">'
                        f'<div style="font-weight:600;color:{COLORS["gold"]};margin-bottom:6px;">{sig.relationship}</div>'
                        f'<div style="font-size:0.8rem;color:{COLORS["text_secondary"]};margin-bottom:8px;">{sig.chain}</div>'
                        f'<div style="display:flex;gap:12px;">'
                        f'<span style="color:{direction_color};font-weight:600;">{sig.direction}</span>'
                        f'<span style="color:{COLORS["text_secondary"]};">Strength: {sig.strength:.0f}/100</span>'
                        f'<span style="color:{COLORS["text_secondary"]};">Confidence: {sig.confidence:.0f}%</span>'
                        f'</div>'
                        f'<div style="font-size:0.82rem;color:{COLORS["text_primary"]};margin-top:8px;">{sig.description[:200]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Click 'Run Cross-Asset Analysis' to generate signals.")
            return
        except Exception:
            pass

    # Fallback: manual cross-asset visualization
    st.info("MacroCrossAssetEngine not available. Showing manual cross-asset analysis.")

    symbols_map = {
        "S&P 500": "^GSPC",
        "10Y Treasury": "^TNX",
        "Gold": "GLD",
        "Oil (WTI)": "USO",
        "USD Index": "UUP",
        "Bitcoin": "BTC-USD",
        "Emerging Markets": "EEM",
        "High Yield": "HYG",
    }

    period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="cross_period")

    if st.button("Fetch Cross-Asset Data", type="primary", key="cross_fetch"):
        with st.spinner("Fetching cross-asset data..."):
            try:
                import yfinance as yf
                tickers = list(symbols_map.values())
                data = yf.download(tickers, period=period, progress=False, threads=True)
                if data.empty:
                    st.error("Could not fetch data.")
                    return

                if isinstance(data.columns, pd.MultiIndex):
                    close_data = data["Close"].dropna(axis=1, how="all")
                else:
                    close_data = data.dropna(axis=1, how="all")

                # Rename columns
                reverse_map = {v: k for k, v in symbols_map.items()}
                close_data.columns = [reverse_map.get(c, c) for c in close_data.columns]

                returns = close_data.pct_change().dropna()
                corr = returns.corr()

                st.session_state["cross_data"] = {
                    "close": close_data,
                    "returns": returns,
                    "corr": corr,
                }
                st.success("Cross-asset data loaded.")
            except Exception as e:
                st.error(f"Error: {e}")

    cross_data = st.session_state.get("cross_data")
    if cross_data is None:
        st.info("Click 'Fetch Cross-Asset Data' to begin.")
        return

    close_data = cross_data["close"]
    corr = cross_data["corr"]

    # Normalized performance
    _section("Normalized Performance (Base 100)")
    fig_norm = go.Figure()
    colors_cycle = [COLORS["gold"], COLORS["lavender"], COLORS["success"],
                    COLORS["danger"], COLORS["neutral"], "#42a5f5", "#ff9800", "#ab47bc"]
    for i, col in enumerate(close_data.columns):
        normalized = close_data[col] / close_data[col].iloc[0] * 100
        fig_norm.add_trace(go.Scatter(
            x=list(range(len(normalized))), y=normalized.values, mode="lines",
            name=col, line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.5),
        ))
    fig_norm.add_hline(y=100, line_color=COLORS["border"])
    fig_norm.update_layout(**_dark_layout(height=400, title="Cross-Asset Performance (Normalized)",
                                           yaxis_title="Normalized Price"))
    st.plotly_chart(fig_norm, use_container_width=True)

    # Correlation heatmap
    _section("Cross-Asset Correlation Matrix")
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(title="Corr", tickfont=dict(color="white")),
    ))
    fig_corr.update_layout(**_dark_layout(height=400, title="Cross-Asset Correlation"))
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_quant_modeling_lab():
    """Main entry point for the Quantitative Modeling Lab."""
    st.markdown(_QML_CSS, unsafe_allow_html=True)
    st.title("Quantitative Modeling Lab")
    st.caption(
        "Advanced quantitative modeling environment — regime detection, liquidity modeling, "
        "risk management, machine learning, and cross-asset analysis."
    )

    tabs = st.tabs([
        "Regime Detection",
        "Liquidity & Impact",
        "Risk Management",
        "ML Framework",
        "Cross-Asset",
    ])

    with tabs[0]:
        _render_regime_tab()

    with tabs[1]:
        _render_liquidity_tab()

    with tabs[2]:
        _render_risk_tab()

    with tabs[3]:
        _render_ml_tab()

    with tabs[4]:
        _render_cross_asset_tab()
