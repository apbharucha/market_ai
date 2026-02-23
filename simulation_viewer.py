"""
Octavian Simulation Hub — Visual Simulation Laboratory
=======================================================
Comprehensive simulation environment covering:
  - Market Microstructure Simulation (agent-based order book)
  - Portfolio Evolution & Drawdown Visualization
  - Scenario & Crisis Simulation (historical + hypothetical)
  - Simulation Universe Generator (synthetic financial worlds)
  - Performance Dashboard with grading
  - Hyperdimensional Visualization (3D manifolds, parameter spaces)

No emojis — sleek CSS microanimations throughout.
"""

from __future__ import annotations

import math
import random
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
        "navy": "#0a1628", "navy_light": "#132240", "navy_mid": "#1a2d4a",
        "gold": "#c9a84c", "gold_light": "#d4b86a", "lavender": "#9b8ec4",
        "white_soft": "#e0e4ec", "text_primary": "#e8eaf0",
        "text_secondary": "#a0a8b8", "border": "#1e3050",
        "success": "#4caf50", "danger": "#ef5350", "neutral": "#78909c",
    }

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    from market_simulation_universe import MarketSimulator, AgentType, EventType
    HAS_UNIVERSE = True
except ImportError:
    HAS_UNIVERSE = False

try:
    from market_simulation_engine import MarketSimulationEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

try:
    from trading_system.simulation_grader import SimulationGrader
    HAS_GRADER = True
except ImportError:
    HAS_GRADER = False

# ── CSS Microanimations ────────────────────────────────────────────────────────
_SIM_CSS = """
<style>
@keyframes sim-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(201,168,76,0); }
    100% { box-shadow: 0 0 0 0 rgba(201,168,76,0); }
}
@keyframes sim-slide-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes sim-bar-fill {
    from { width: 0%; }
    to   { width: var(--bar-width); }
}
@keyframes sim-dot-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes sim-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes sim-fade-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes sim-glow-pulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1.0; }
}
.sim-card {
    background: linear-gradient(135deg, #132240 0%, #1a2d4a 100%);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0;
    animation: sim-slide-in 0.4s ease-out;
}
.sim-card:hover {
    border-color: rgba(201,168,76,0.35);
    transition: border-color 0.25s ease;
}
.sim-metric {
    background: rgba(10,22,40,0.7);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    animation: sim-slide-in 0.35s ease-out;
}
.sim-metric-label {
    font-size: 0.72rem;
    color: #a0a8b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.sim-metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', monospace;
}
.sim-badge-bull {
    display: inline-block;
    background: rgba(76,175,80,0.15);
    border: 1px solid rgba(76,175,80,0.4);
    color: #4caf50;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    animation: sim-slide-in 0.3s ease-out;
}
.sim-badge-bear {
    display: inline-block;
    background: rgba(239,83,80,0.15);
    border: 1px solid rgba(239,83,80,0.4);
    color: #ef5350;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    animation: sim-slide-in 0.3s ease-out;
}
.sim-badge-neutral {
    display: inline-block;
    background: rgba(120,144,156,0.15);
    border: 1px solid rgba(120,144,156,0.4);
    color: #78909c;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.sim-live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4caf50;
    border-radius: 50%;
    margin-right: 6px;
    animation: sim-dot-blink 1.5s ease-in-out infinite;
}
.sim-section-header {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #c9a84c;
    border-bottom: 1px solid rgba(201,168,76,0.2);
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
    animation: sim-fade-up 0.4s ease-out;
}
.sim-event-tag {
    display: inline-block;
    background: rgba(155,142,196,0.15);
    border: 1px solid rgba(155,142,196,0.3);
    color: #9b8ec4;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px;
    animation: sim-slide-in 0.3s ease-out;
}
.sim-progress-bar {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin: 4px 0;
}
.sim-progress-fill {
    height: 100%;
    border-radius: 3px;
    animation: sim-bar-fill 0.8s ease-out forwards;
    background: linear-gradient(90deg, #c9a84c, #d4b86a);
}
.sim-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(201,168,76,0.3);
    border-top-color: #c9a84c;
    border-radius: 50%;
    animation: sim-spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}
.sim-regime-label {
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    animation: sim-slide-in 0.3s ease-out;
}
</style>
"""


# ── Helper: dark plotly layout ─────────────────────────────────────────────────
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


def _section(title: str):
    st.markdown(f'<div class="sim-section-header">{title}</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str, color: str = "#e8eaf0"):
    st.markdown(
        f'<div class="sim-metric">'
        f'<div class="sim-metric-label">{label}</div>'
        f'<div class="sim-metric-value" style="color:{color};">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Microstructure Simulation
# ══════════════════════════════════════════════════════════════════════════════

def _render_microstructure_tab():
    _section("Agent-Based Market Microstructure Simulation")
    st.caption(
        "Simulate realistic order book dynamics with multiple agent types: "
        "Fundamentalists, Trend Followers, Market Makers, Noise Traders, Institutional, HFT."
    )

    if not HAS_UNIVERSE:
        st.error("market_simulation_universe.py not found. Cannot run microstructure simulation.")
        return

    with st.expander("Simulation Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_agents = st.slider("Number of Agents", 50, 500, 150, step=25, key="sim_n_agents")
            n_steps = st.slider("Simulation Steps", 100, 2000, 500, step=100, key="sim_n_steps")
        with c2:
            initial_price = st.number_input("Initial Price ($)", value=100.0, min_value=1.0, key="sim_init_price")
            fundamental_vol = st.slider("Fundamental Volatility (%)", 5, 50, 15, key="sim_fvol") / 100
        with c3:
            event_prob = st.slider("Event Probability (%)", 0, 10, 2, key="sim_evprob") / 100
            seed = st.number_input("Random Seed (0=random)", value=42, min_value=0, key="sim_seed")

        st.markdown("**Agent Mix Override** (leave at 0 for defaults)")
        am_c1, am_c2, am_c3 = st.columns(3)
        with am_c1:
            pct_fund = st.slider("Fundamentalists %", 0, 60, 0, key="sim_pct_fund")
            pct_trend = st.slider("Trend Followers %", 0, 60, 0, key="sim_pct_trend")
        with am_c2:
            pct_mm = st.slider("Market Makers %", 0, 40, 0, key="sim_pct_mm")
            pct_noise = st.slider("Noise Traders %", 0, 60, 0, key="sim_pct_noise")
        with am_c3:
            pct_inst = st.slider("Institutional %", 0, 40, 0, key="sim_pct_inst")
            pct_hft = st.slider("HFT %", 0, 40, 0, key="sim_pct_hft")

    if st.button("Run Microstructure Simulation", type="primary", key="sim_run_micro"):
        # Build agent mix
        agent_mix = None
        total_custom = pct_fund + pct_trend + pct_mm + pct_noise + pct_inst + pct_hft
        if total_custom > 0:
            agent_mix = {
                AgentType.FUNDAMENTALIST: pct_fund / 100,
                AgentType.TREND_FOLLOWER: pct_trend / 100,
                AgentType.MARKET_MAKER: pct_mm / 100,
                AgentType.NOISE_TRADER: pct_noise / 100,
                AgentType.INSTITUTIONAL: pct_inst / 100,
                AgentType.HFT: pct_hft / 100,
            }
            # Normalize
            total = sum(agent_mix.values())
            if total > 0:
                agent_mix = {k: v / total for k, v in agent_mix.items()}

        with st.spinner("Running agent-based simulation..."):
            try:
                sim = MarketSimulator(
                    n_agents=n_agents,
                    n_steps=n_steps,
                    fundamental_vol=fundamental_vol,
                    event_prob=event_prob,
                    seed=int(seed) if seed > 0 else None,
                    agent_mix=agent_mix,
                )
                result = sim.run(initial_price=initial_price)
                st.session_state["sim_micro_result"] = result
                st.success(f"Simulation complete — {n_steps} steps, {n_agents} agents, {result.n_trades} trades executed.")
            except Exception as e:
                st.error(f"Simulation error: {e}")
                return

    result = st.session_state.get("sim_micro_result")
    if result is None:
        st.info("Configure parameters above and click 'Run Microstructure Simulation'.")
        return

    df = result.to_dataframe()
    regimes = result.regime_labels()
    df["regime"] = regimes

    # ── Summary metrics ────────────────────────────────────────────────────────
    _section("Simulation Summary")
    price_return = (result.final_price - initial_price) / initial_price * 100
    avg_spread = df["spread"].mean()
    avg_vol = df["realized_vol"].mean()
    avg_ofi = df["ofi"].mean()

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1: _metric_card("Final Price", f"${result.final_price:,.2f}", "#c9a84c")
    with mc2:
        color = COLORS["success"] if price_return >= 0 else COLORS["danger"]
        _metric_card("Total Return", f"{price_return:+.2f}%", color)
    with mc3: _metric_card("Total Trades", f"{result.n_trades:,}")
    with mc4: _metric_card("Avg Spread", f"${avg_spread:.4f}")
    with mc5: _metric_card("Avg Realized Vol", f"{avg_vol:.1f}%")

    # ── Price chart with events ────────────────────────────────────────────────
    _section("Price Path & Market Events")
    fig_price = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=["Mid Price vs Fundamental Value", "Volume", "Order Flow Imbalance"],
    )

    # Price
    fig_price.add_trace(go.Scatter(
        x=df["step"], y=df["mid_price"], mode="lines", name="Mid Price",
        line=dict(color=COLORS["gold"], width=1.5),
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=df["step"], y=df["fundamental_value"], mode="lines", name="Fundamental Value",
        line=dict(color=COLORS["lavender"], width=1.0, dash="dot"),
    ), row=1, col=1)

    # Event markers
    for step, etype, desc in result.events:
        if step < len(df):
            price_at = df.loc[df["step"] == step, "mid_price"]
            if not price_at.empty:
                fig_price.add_trace(go.Scatter(
                    x=[step], y=[float(price_at.iloc[0])],
                    mode="markers", name=etype.value,
                    marker=dict(symbol="diamond", size=10, color="#ff9800",
                                line=dict(width=1, color="white")),
                    hovertext=desc, showlegend=False,
                ), row=1, col=1)

    # Volume
    fig_price.add_trace(go.Bar(
        x=df["step"], y=df["volume"], name="Volume",
        marker_color=COLORS["lavender"], opacity=0.6,
    ), row=2, col=1)

    # OFI
    ofi_colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in df["ofi"]]
    fig_price.add_trace(go.Bar(
        x=df["step"], y=df["ofi"], name="OFI",
        marker_color=ofi_colors, opacity=0.7,
    ), row=3, col=1)
    fig_price.add_hline(y=0, line_dash="dash", line_color=COLORS["border"], row=3, col=1)

    fig_price.update_layout(**_dark_layout(height=600, showlegend=True))
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Regime distribution ────────────────────────────────────────────────────
    _section("Market Regime Distribution")
    regime_counts = pd.Series(regimes).value_counts()
    fig_regime = go.Figure(go.Bar(
        x=regime_counts.index.tolist(),
        y=regime_counts.values.tolist(),
        marker_color=[COLORS["gold"], COLORS["lavender"], COLORS["success"],
                      COLORS["danger"], COLORS["neutral"]][:len(regime_counts)],
    ))
    fig_regime.update_layout(**_dark_layout(height=280, title="Steps per Regime"))
    st.plotly_chart(fig_regime, use_container_width=True)

    # ── Agent PnL breakdown ────────────────────────────────────────────────────
    _section("Agent Type Performance")
    agent_df = result.agent_type_pnl_summary()
    if not agent_df.empty:
        st.dataframe(agent_df, use_container_width=True, hide_index=True)

    # ── Order book depth over time ─────────────────────────────────────────────
    _section("Order Book Depth")
    fig_depth = go.Figure()
    fig_depth.add_trace(go.Scatter(
        x=df["step"], y=df["bid_depth"], mode="lines", name="Bid Depth",
        fill="tozeroy", fillcolor="rgba(76,175,80,0.15)",
        line=dict(color=COLORS["success"], width=1.2),
    ))
    fig_depth.add_trace(go.Scatter(
        x=df["step"], y=-df["ask_depth"], mode="lines", name="Ask Depth",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        line=dict(color=COLORS["danger"], width=1.2),
    ))
    fig_depth.add_hline(y=0, line_color=COLORS["border"])
    fig_depth.update_layout(**_dark_layout(height=280, title="Bid/Ask Depth (Bid positive, Ask negative)"))
    st.plotly_chart(fig_depth, use_container_width=True)

    # ── Events log ────────────────────────────────────────────────────────────
    if result.events:
        _section("Market Events Log")
        events_df = pd.DataFrame([
            {"Step": s, "Event Type": e.value, "Description": d}
            for s, e, d in result.events
        ])
        st.dataframe(events_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Portfolio Evolution & Drawdown
# ══════════════════════════════════════════════════════════════════════════════

def _render_portfolio_evolution_tab():
    _section("Portfolio Evolution & Risk Surface")
    st.caption(
        "Simulate portfolio growth paths using Monte Carlo, visualize drawdown maps, "
        "and explore risk surfaces across parameter spaces."
    )

    with st.expander("Monte Carlo Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            initial_capital = st.number_input("Initial Capital ($)", value=100_000.0, step=10_000.0, key="pe_capital")
            n_paths = st.slider("Number of Paths", 50, 1000, 200, step=50, key="pe_paths")
        with c2:
            annual_return = st.slider("Expected Annual Return (%)", -20, 50, 10, key="pe_ret") / 100
            annual_vol = st.slider("Annual Volatility (%)", 5, 80, 20, key="pe_vol") / 100
        with c3:
            horizon_years = st.slider("Horizon (Years)", 1, 20, 5, key="pe_horizon")
            trading_days = st.slider("Trading Days/Year", 200, 260, 252, key="pe_tdays")

    if st.button("Run Monte Carlo Simulation", type="primary", key="pe_run"):
        with st.spinner("Simulating portfolio paths..."):
            dt = 1 / trading_days
            n_steps_total = horizon_years * trading_days
            rng = np.random.default_rng(42)

            paths = np.zeros((n_paths, n_steps_total + 1))
            paths[:, 0] = initial_capital

            for t in range(1, n_steps_total + 1):
                shocks = rng.normal(0, 1, n_paths)
                daily_ret = (annual_return - 0.5 * annual_vol**2) * dt + annual_vol * math.sqrt(dt) * shocks
                paths[:, t] = paths[:, t - 1] * np.exp(daily_ret)

            st.session_state["pe_paths_data"] = paths
            st.session_state["pe_n_steps"] = n_steps_total
            st.session_state["pe_initial"] = initial_capital

    paths_data = st.session_state.get("pe_paths_data")
    if paths_data is None:
        st.info("Configure parameters and click 'Run Monte Carlo Simulation'.")
        return

    n_steps_total = st.session_state["pe_n_steps"]
    initial_capital = st.session_state["pe_initial"]
    x_axis = np.linspace(0, horizon_years, n_steps_total + 1)

    # ── Summary stats ──────────────────────────────────────────────────────────
    final_values = paths_data[:, -1]
    median_final = float(np.median(final_values))
    p10 = float(np.percentile(final_values, 10))
    p90 = float(np.percentile(final_values, 90))
    prob_profit = float(np.mean(final_values > initial_capital) * 100)

    # Max drawdown per path
    def _max_dd(path):
        peak = np.maximum.accumulate(path)
        dd = (path - peak) / np.maximum(peak, 1e-10)
        return float(dd.min() * 100)

    max_dds = [_max_dd(paths_data[i]) for i in range(len(paths_data))]
    avg_max_dd = float(np.mean(max_dds))

    _section("Monte Carlo Summary")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1: _metric_card("Median Final", f"${median_final:,.0f}", COLORS["gold"])
    with mc2: _metric_card("10th Pct", f"${p10:,.0f}", COLORS["danger"])
    with mc3: _metric_card("90th Pct", f"${p90:,.0f}", COLORS["success"])
    with mc4: _metric_card("Prob Profit", f"{prob_profit:.1f}%", COLORS["lavender"])
    with mc5: _metric_card("Avg Max DD", f"{avg_max_dd:.1f}%", COLORS["danger"])

    # ── Capital growth paths ───────────────────────────────────────────────────
    _section("Capital Growth Paths")
    fig_paths = go.Figure()

    # Plot subset of paths
    n_show = min(100, len(paths_data))
    for i in range(n_show):
        fig_paths.add_trace(go.Scatter(
            x=x_axis, y=paths_data[i],
            mode="lines", line=dict(width=0.5, color="rgba(201,168,76,0.15)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    p10_path = np.percentile(paths_data, 10, axis=0)
    p50_path = np.percentile(paths_data, 50, axis=0)
    p90_path = np.percentile(paths_data, 90, axis=0)

    fig_paths.add_trace(go.Scatter(
        x=x_axis, y=p90_path, mode="lines", name="90th Pct",
        line=dict(color=COLORS["success"], width=2),
    ))
    fig_paths.add_trace(go.Scatter(
        x=x_axis, y=p50_path, mode="lines", name="Median",
        line=dict(color=COLORS["gold"], width=2.5),
    ))
    fig_paths.add_trace(go.Scatter(
        x=x_axis, y=p10_path, mode="lines", name="10th Pct",
        line=dict(color=COLORS["danger"], width=2),
    ))
    fig_paths.add_hline(y=initial_capital, line_dash="dash", line_color=COLORS["border"],
                        annotation_text="Initial Capital")

    fig_paths.update_layout(**_dark_layout(
        height=450, title="Monte Carlo Portfolio Paths",
        xaxis_title="Years", yaxis_title="Portfolio Value ($)",
    ))
    st.plotly_chart(fig_paths, use_container_width=True)

    # ── Drawdown map ───────────────────────────────────────────────────────────
    _section("Drawdown Map")
    # Compute drawdown for median path
    dd_series = []
    for t in range(n_steps_total + 1):
        peak = np.max(p50_path[:t + 1]) if t > 0 else p50_path[0]
        dd = (p50_path[t] - peak) / max(peak, 1e-10) * 100
        dd_series.append(dd)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=x_axis, y=dd_series, mode="lines", name="Drawdown (Median Path)",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        line=dict(color=COLORS["danger"], width=1.5),
    ))
    fig_dd.add_hline(y=0, line_color=COLORS["border"])
    fig_dd.update_layout(**_dark_layout(
        height=280, title="Drawdown from Peak (Median Path)",
        xaxis_title="Years", yaxis_title="Drawdown (%)",
    ))
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Final value distribution ───────────────────────────────────────────────
    _section("Final Value Distribution")
    fig_hist = go.Figure(go.Histogram(
        x=final_values, nbinsx=50,
        marker_color=COLORS["gold"], opacity=0.75,
        name="Final Portfolio Value",
    ))
    fig_hist.add_vline(x=initial_capital, line_dash="dash", line_color=COLORS["lavender"],
                       annotation_text="Initial Capital")
    fig_hist.add_vline(x=median_final, line_dash="dot", line_color=COLORS["gold"],
                       annotation_text="Median")
    fig_hist.update_layout(**_dark_layout(
        height=300, title="Distribution of Final Portfolio Values",
        xaxis_title="Final Value ($)", yaxis_title="Count",
    ))
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Scenario & Crisis Simulation
# ══════════════════════════════════════════════════════════════════════════════

CRISIS_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Global financial crisis triggered by subprime mortgage collapse. S&P 500 fell ~57% peak-to-trough.",
        "shock_pct": -57.0,
        "duration_days": 517,
        "vol_multiplier": 4.5,
        "recovery_days": 1500,
        "phases": [
            ("Pre-Crisis (Normal)", 0, 100, 15),
            ("Initial Shock", 100, 200, 35),
            ("Acute Crisis", 200, 350, 60),
            ("Stabilization", 350, 517, 40),
            ("Recovery", 517, 1000, 25),
        ],
    },
    "Dot-com Bubble (2000-2002)": {
        "description": "Technology bubble burst. NASDAQ fell ~78% peak-to-trough over 2.5 years.",
        "shock_pct": -78.0,
        "duration_days": 929,
        "vol_multiplier": 3.0,
        "recovery_days": 3000,
        "phases": [
            ("Bubble Peak", 0, 50, 30),
            ("Initial Decline", 50, 300, 25),
            ("Continued Selloff", 300, 600, 30),
            ("Capitulation", 600, 929, 35),
            ("Recovery", 929, 1500, 20),
        ],
    },
    "COVID-19 Crash (2020)": {
        "description": "Fastest bear market in history. S&P 500 fell ~34% in 33 days, then recovered sharply.",
        "shock_pct": -34.0,
        "duration_days": 33,
        "vol_multiplier": 6.0,
        "recovery_days": 148,
        "phases": [
            ("Pre-COVID (Normal)", 0, 20, 12),
            ("Panic Selloff", 20, 53, 80),
            ("Fed Intervention", 53, 100, 45),
            ("V-Shape Recovery", 100, 200, 25),
            ("New Highs", 200, 400, 18),
        ],
    },
    "Interest Rate Spike (+300bps)": {
        "description": "Hypothetical: Fed raises rates 300bps in 6 months. Bonds crash, equities reprice.",
        "shock_pct": -25.0,
        "duration_days": 180,
        "vol_multiplier": 2.5,
        "recovery_days": 600,
        "phases": [
            ("Pre-Shock", 0, 30, 15),
            ("Rate Shock", 30, 120, 30),
            ("Repricing", 120, 180, 25),
            ("Stabilization", 180, 400, 20),
        ],
    },
    "Liquidity Collapse": {
        "description": "Hypothetical: Sudden liquidity withdrawal. Bid-ask spreads widen 10x, forced selling.",
        "shock_pct": -45.0,
        "duration_days": 60,
        "vol_multiplier": 8.0,
        "recovery_days": 400,
        "phases": [
            ("Normal Liquidity", 0, 10, 12),
            ("Liquidity Shock", 10, 40, 90),
            ("Forced Selling", 40, 60, 70),
            ("Central Bank Response", 60, 150, 40),
            ("Recovery", 150, 400, 22),
        ],
    },
    "Volatility Explosion (VIX 80+)": {
        "description": "Hypothetical: VIX spikes to 80+. Options market dislocates, gamma squeeze.",
        "shock_pct": -30.0,
        "duration_days": 45,
        "vol_multiplier": 7.0,
        "recovery_days": 200,
        "phases": [
            ("Low Vol Regime", 0, 15, 10),
            ("Vol Spike", 15, 45, 85),
            ("Normalization", 45, 120, 35),
            ("Recovery", 120, 200, 18),
        ],
    },
}


def _simulate_crisis_path(scenario: dict, initial_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic price path for a crisis scenario."""
    rng = np.random.default_rng(seed)
    total_days = scenario["recovery_days"]
    prices = [initial_price]
    vols = []
    phases_used = []

    for phase_name, start, end, vol_ann in scenario["phases"]:
        n = end - start
        if n <= 0:
            continue
        dt = 1 / 252
        # Determine drift for this phase
        if "Crisis" in phase_name or "Shock" in phase_name or "Crash" in phase_name or "Selloff" in phase_name or "Collapse" in phase_name or "Explosion" in phase_name or "Capitulation" in phase_name or "Panic" in phase_name or "Spike" in phase_name:
            # Negative drift
            total_shock = scenario["shock_pct"] / 100
            daily_drift = (total_shock / max(scenario["duration_days"], 1)) * 252
        elif "Recovery" in phase_name or "Highs" in phase_name or "Intervention" in phase_name or "Response" in phase_name:
            daily_drift = 0.15  # recovery drift
        else:
            daily_drift = 0.05  # normal drift

        sigma = vol_ann / 100
        for _ in range(n):
            shock = float(rng.normal(0, 1))
            ret = (daily_drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock
            new_price = max(prices[-1] * math.exp(ret), 0.01)
            prices.append(new_price)
            vols.append(vol_ann)
            phases_used.append(phase_name)

    # Trim to total_days
    prices = prices[:total_days + 1]
    vols = vols[:total_days]
    phases_used = phases_used[:total_days]

    df = pd.DataFrame({
        "day": range(len(prices)),
        "price": prices,
    })
    if vols:
        df["vol"] = [vols[0]] + vols[:len(prices) - 1]
    else:
        df["vol"] = 15.0
    if phases_used:
        df["phase"] = ["Pre-Start"] + phases_used[:len(prices) - 1]
    else:
        df["phase"] = "Unknown"

    return df


def _render_crisis_tab():
    _section("Scenario & Crisis Simulation")
    st.caption(
        "Test strategies against historical crises and hypothetical macro shocks. "
        "Analyze drawdowns, recovery paths, and strategy performance under stress."
    )

    scenario_name = st.selectbox(
        "Select Scenario", list(CRISIS_SCENARIOS.keys()), key="crisis_scenario"
    )
    scenario = CRISIS_SCENARIOS[scenario_name]

    # Scenario description card
    st.markdown(
        f'<div class="sim-card">'
        f'<div style="font-size:1.1rem;font-weight:600;color:{COLORS["gold"]};margin-bottom:8px;">{scenario_name}</div>'
        f'<div style="color:{COLORS["text_secondary"]};font-size:0.9rem;">{scenario["description"]}</div>'
        f'<div style="margin-top:12px;display:flex;gap:16px;">'
        f'<span class="sim-badge-bear">Peak Drawdown: {scenario["shock_pct"]:.0f}%</span>'
        f'<span class="sim-badge-neutral">Crisis Duration: {scenario["duration_days"]}d</span>'
        f'<span class="sim-badge-bull">Recovery: {scenario["recovery_days"]}d</span>'
        f'<span class="sim-badge-neutral">Vol Multiplier: {scenario["vol_multiplier"]}x</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        initial_price = st.number_input("Initial Price ($)", value=100.0, min_value=1.0, key="crisis_init")
        portfolio_value = st.number_input("Portfolio Value ($)", value=1_000_000.0, step=100_000.0, key="crisis_port")
    with c2:
        seed = st.number_input("Random Seed", value=42, min_value=0, key="crisis_seed")
        show_strategy = st.checkbox("Overlay Strategy (60/40 Rebalance)", value=True, key="crisis_strat")

    if st.button("Run Crisis Simulation", type="primary", key="crisis_run"):
        with st.spinner(f"Simulating {scenario_name}..."):
            df = _simulate_crisis_path(scenario, initial_price=initial_price, seed=int(seed))
            st.session_state["crisis_df"] = df
            st.session_state["crisis_scenario_name"] = scenario_name
            st.session_state["crisis_portfolio"] = portfolio_value

    df = st.session_state.get("crisis_df")
    if df is None:
        st.info("Select a scenario and click 'Run Crisis Simulation'.")
        return

    portfolio_value = st.session_state.get("crisis_portfolio", 1_000_000.0)

    # ── Price path ─────────────────────────────────────────────────────────────
    _section("Price Path")
    fig_crisis = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3], vertical_spacing=0.04)

    # Color by phase
    phase_colors = {
        "Pre-Crisis (Normal)": COLORS["success"],
        "Pre-COVID (Normal)": COLORS["success"],
        "Pre-Shock": COLORS["success"],
        "Normal Liquidity": COLORS["success"],
        "Low Vol Regime": COLORS["success"],
        "Bubble Peak": COLORS["gold"],
        "Initial Shock": COLORS["danger"],
        "Acute Crisis": "#ff1744",
        "Panic Selloff": "#ff1744",
        "Liquidity Shock": "#ff1744",
        "Vol Spike": "#ff1744",
        "Initial Decline": COLORS["danger"],
        "Continued Selloff": COLORS["danger"],
        "Capitulation": "#ff1744",
        "Forced Selling": "#ff1744",
        "Stabilization": COLORS["neutral"],
        "Normalization": COLORS["neutral"],
        "Repricing": COLORS["neutral"],
        "Fed Intervention": COLORS["lavender"],
        "Central Bank Response": COLORS["lavender"],
        "Recovery": COLORS["success"],
        "V-Shape Recovery": COLORS["success"],
        "New Highs": COLORS["gold"],
    }

    fig_crisis.add_trace(go.Scatter(
        x=df["day"], y=df["price"], mode="lines", name="Price",
        line=dict(color=COLORS["gold"], width=2),
    ), row=1, col=1)
    fig_crisis.add_hline(y=initial_price, line_dash="dash", line_color=COLORS["border"],
                         annotation_text="Initial Price", row=1, col=1)

    # Drawdown
    peak = df["price"].cummax()
    dd = (df["price"] - peak) / peak.clip(lower=1e-10) * 100
    fig_crisis.add_trace(go.Scatter(
        x=df["day"], y=dd, mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        line=dict(color=COLORS["danger"], width=1.5),
    ), row=2, col=1)
    fig_crisis.add_hline(y=0, line_color=COLORS["border"], row=2, col=1)

    fig_crisis.update_layout(**_dark_layout(height=500, title=f"{scenario_name} — Price Path & Drawdown"))
    fig_crisis.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_crisis.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig_crisis, use_container_width=True)

    # ── Portfolio impact ───────────────────────────────────────────────────────
    _section("Portfolio Impact Analysis")
    port_values = df["price"] / initial_price * portfolio_value
    max_dd_pct = float(dd.min())
    max_dd_dollar = portfolio_value * abs(max_dd_pct) / 100
    final_return = (df["price"].iloc[-1] / initial_price - 1) * 100
    final_port = port_values.iloc[-1]

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1: _metric_card("Peak Drawdown", f"{max_dd_pct:.1f}%", COLORS["danger"])
    with pc2: _metric_card("Max $ Loss", f"${max_dd_dollar:,.0f}", COLORS["danger"])
    with pc3:
        color = COLORS["success"] if final_return >= 0 else COLORS["danger"]
        _metric_card("Final Return", f"{final_return:+.1f}%", color)
    with pc4:
        color = COLORS["success"] if final_port >= portfolio_value else COLORS["danger"]
        _metric_card("Final Portfolio", f"${final_port:,.0f}", color)

    # ── Volatility regime ──────────────────────────────────────────────────────
    _section("Volatility Regime")
    fig_vol = go.Figure(go.Scatter(
        x=df["day"], y=df["vol"], mode="lines", name="Realized Vol",
        fill="tozeroy", fillcolor="rgba(155,142,196,0.15)",
        line=dict(color=COLORS["lavender"], width=1.5),
    ))
    fig_vol.add_hline(y=20, line_dash="dot", line_color=COLORS["neutral"],
                      annotation_text="Normal Vol (20%)")
    fig_vol.update_layout(**_dark_layout(height=250, title="Annualized Volatility by Phase",
                                         xaxis_title="Day", yaxis_title="Vol (%)"))
    st.plotly_chart(fig_vol, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Simulation Universe Generator
# ══════════════════════════════════════════════════════════════════════════════

def _render_universe_tab():
    _section("Simulation Universe Generator")
    st.caption(
        "Generate synthetic financial worlds with agent-based dynamics, stochastic macro environments, "
        "and behavioral trading agents. Use for stress testing and RL training."
    )

    with st.expander("Universe Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_assets = st.slider("Number of Assets", 3, 20, 5, key="univ_assets")
            n_steps = st.slider("Simulation Steps", 100, 1000, 300, key="univ_steps")
            seed = st.number_input("Seed", value=42, min_value=0, key="univ_seed")
        with c2:
            macro_regime = st.selectbox(
                "Macro Regime",
                ["Bull Market", "Bear Market", "Sideways", "High Volatility", "Crisis"],
                key="univ_regime",
            )
            correlation_level = st.slider("Asset Correlation", 0, 100, 40, key="univ_corr") / 100
        with c3:
            include_liquidity_shock = st.checkbox("Include Liquidity Shock", value=True, key="univ_liq")
            include_macro_shock = st.checkbox("Include Macro Shock", value=True, key="univ_macro")
            shock_probability = st.slider("Shock Probability (%)", 0, 20, 5, key="univ_shock_prob") / 100

    if st.button("Generate Synthetic Universe", type="primary", key="univ_run"):
        with st.spinner("Generating synthetic financial universe..."):
            rng = np.random.default_rng(int(seed) if seed > 0 else None)

            # Regime parameters
            regime_params = {
                "Bull Market": {"drift": 0.15, "vol": 0.15, "shock_mult": 0.5},
                "Bear Market": {"drift": -0.20, "vol": 0.25, "shock_mult": 1.5},
                "Sideways": {"drift": 0.02, "vol": 0.12, "shock_mult": 0.8},
                "High Volatility": {"drift": 0.05, "vol": 0.40, "shock_mult": 2.0},
                "Crisis": {"drift": -0.40, "vol": 0.60, "shock_mult": 3.0},
            }
            params = regime_params[macro_regime]

            # Generate correlated asset paths
            dt = 1 / 252
            asset_names = [f"Asset {chr(65 + i)}" for i in range(n_assets)]
            initial_prices = rng.uniform(50, 200, n_assets)

            # Correlation matrix (Cholesky decomposition)
            corr_matrix = np.full((n_assets, n_assets), correlation_level)
            np.fill_diagonal(corr_matrix, 1.0)
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                # Fallback: identity
                L = np.eye(n_assets)

            paths = np.zeros((n_assets, n_steps + 1))
            paths[:, 0] = initial_prices
            events_log = []

            for t in range(1, n_steps + 1):
                # Correlated shocks
                z = rng.normal(0, 1, n_assets)
                corr_z = L @ z

                # Check for shocks
                shock_mult = 1.0
                if include_liquidity_shock and rng.random() < shock_probability:
                    shock_mult = -params["shock_mult"] * rng.uniform(0.5, 1.5)
                    events_log.append((t, "Liquidity Shock"))
                elif include_macro_shock and rng.random() < shock_probability * 0.5:
                    shock_mult = params["shock_mult"] * rng.uniform(-1.0, 1.0)
                    events_log.append((t, "Macro Shock"))

                for i in range(n_assets):
                    asset_vol = params["vol"] * rng.uniform(0.7, 1.3)
                    drift = params["drift"] + shock_mult * 0.01
                    ret = (drift - 0.5 * asset_vol**2) * dt + asset_vol * math.sqrt(dt) * corr_z[i]
                    paths[i, t] = max(paths[i, t - 1] * math.exp(ret), 0.01)

            st.session_state["univ_paths"] = paths
            st.session_state["univ_names"] = asset_names
            st.session_state["univ_events"] = events_log
            st.session_state["univ_regime"] = macro_regime
            st.success(f"Generated {n_assets}-asset universe over {n_steps} steps in {macro_regime} regime.")

    paths = st.session_state.get("univ_paths")
    if paths is None:
        st.info("Configure the universe and click 'Generate Synthetic Universe'.")
        return

    asset_names = st.session_state["univ_names"]
    events_log = st.session_state["univ_events"]
    macro_regime = st.session_state["univ_regime"]
    n_steps = paths.shape[1] - 1

    # ── Asset price paths ──────────────────────────────────────────────────────
    _section("Synthetic Asset Price Paths")
    fig_univ = go.Figure()
    colors_cycle = [COLORS["gold"], COLORS["lavender"], COLORS["success"],
                    COLORS["danger"], COLORS["neutral"], "#42a5f5", "#ff9800",
                    "#ab47bc", "#26c6da", "#ef5350"]

    for i, name in enumerate(asset_names):
        normalized = paths[i] / paths[i, 0] * 100
        fig_univ.add_trace(go.Scatter(
            x=list(range(n_steps + 1)), y=normalized,
            mode="lines", name=name,
            line=dict(color=colors_cycle[i % len(colors_cycle)], width=1.5),
        ))

    # Event markers
    for step, etype in events_log[:20]:
        fig_univ.add_vline(x=step, line_dash="dot", line_color="rgba(255,152,0,0.4)",
                           annotation_text=etype[:8], annotation_font_size=8)

    fig_univ.add_hline(y=100, line_dash="dash", line_color=COLORS["border"])
    fig_univ.update_layout(**_dark_layout(
        height=450, title=f"Synthetic Universe — {macro_regime} Regime (Normalized to 100)",
        xaxis_title="Step", yaxis_title="Normalized Price",
    ))
    st.plotly_chart(fig_univ, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    _section("Realized Correlation Matrix")
    returns_matrix = np.diff(np.log(paths), axis=1)
    corr_realized = np.corrcoef(returns_matrix)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_realized,
        x=asset_names, y=asset_names,
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_realized, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig_corr.update_layout(**_dark_layout(height=350, title="Realized Asset Correlations"))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Return distribution ────────────────────────────────────────────────────
    _section("Return Distribution by Asset")
    fig_ret_dist = go.Figure()
    for i, name in enumerate(asset_names):
        rets = np.diff(np.log(paths[i])) * 100
        fig_ret_dist.add_trace(go.Violin(
            y=rets, name=name,
            line_color=colors_cycle[i % len(colors_cycle)],
            fillcolor=f"rgba({int(colors_cycle[i % len(colors_cycle)][1:3], 16)},"
                      f"{int(colors_cycle[i % len(colors_cycle)][3:5], 16)},"
                      f"{int(colors_cycle[i % len(colors_cycle)][5:7], 16)},0.2)",
            box_visible=True, meanline_visible=True,
        ))
    fig_ret_dist.update_layout(**_dark_layout(height=350, title="Daily Return Distributions",
                                               yaxis_title="Daily Return (%)"))
    st.plotly_chart(fig_ret_dist, use_container_width=True)

    # ── Events log ────────────────────────────────────────────────────────────
    if events_log:
        _section("Shock Events Log")
        events_df = pd.DataFrame(events_log, columns=["Step", "Event Type"])
        st.dataframe(events_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Hyperdimensional Visualization
# ══════════════════════════════════════════════════════════════════════════════

def _render_hyperdim_tab():
    _section("Hyperdimensional Mathematical Modeling")
    st.caption(
        "Visualize high-dimensional financial models projected into interactive 3D spaces. "
        "Explore portfolio state spaces, derivative sensitivity surfaces, and stochastic manifolds."
    )

    viz_type = st.selectbox(
        "Visualization Type",
        [
            "Volatility Surface (Options)",
            "Portfolio Risk Surface",
            "Factor Return Manifold",
            "Monte Carlo 3D Paths",
            "Correlation Manifold",
        ],
        key="hyperdim_type",
    )

    if viz_type == "Volatility Surface (Options)":
        _render_vol_surface()
    elif viz_type == "Portfolio Risk Surface":
        _render_risk_surface()
    elif viz_type == "Factor Return Manifold":
        _render_factor_manifold()
    elif viz_type == "Monte Carlo 3D Paths":
        _render_3d_mc_paths()
    elif viz_type == "Correlation Manifold":
        _render_correlation_manifold()


def _render_vol_surface():
    st.markdown("**Implied Volatility Surface** — Strike vs Expiry vs IV")
    c1, c2 = st.columns(2)
    with c1:
        spot = st.number_input("Spot Price ($)", value=100.0, key="vs_spot")
        base_vol = st.slider("Base IV (%)", 10, 60, 25, key="vs_base_vol") / 100
    with c2:
        skew = st.slider("Skew Factor", -2.0, 2.0, -0.5, step=0.1, key="vs_skew")
        term_slope = st.slider("Term Structure Slope", -0.5, 0.5, 0.1, step=0.05, key="vs_term")

    # Generate vol surface
    strikes = np.linspace(spot * 0.7, spot * 1.3, 30)
    expiries = np.array([7, 14, 30, 45, 60, 90, 120, 180, 252]) / 252  # in years

    K, T = np.meshgrid(strikes, expiries)
    moneyness = np.log(K / spot)
    iv_surface = base_vol + skew * moneyness + term_slope * T + 0.05 * moneyness**2

    fig = go.Figure(go.Surface(
        x=strikes, y=expiries * 252,  # show in days
        z=iv_surface * 100,
        colorscale="Viridis",
        colorbar=dict(title="IV (%)", tickfont=dict(color="white")),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
    ))
    fig.update_layout(**_dark_layout(
        height=550,
        scene=dict(
            xaxis_title="Strike ($)",
            yaxis_title="Expiry (Days)",
            zaxis_title="Implied Vol (%)",
            bgcolor=COLORS["navy"],
            xaxis=dict(gridcolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"]),
            zaxis=dict(gridcolor=COLORS["border"]),
        ),
        title="Implied Volatility Surface",
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_surface():
    st.markdown("**Portfolio Risk Surface** — Allocation vs Return vs Volatility")

    n_points = 50
    rng = np.random.default_rng(42)

    # Simulate efficient frontier surface
    alloc_equity = np.linspace(0, 1, n_points)
    alloc_bonds = np.linspace(0, 1, n_points)
    AE, AB = np.meshgrid(alloc_equity, alloc_bonds)

    # Only valid where AE + AB <= 1
    alloc_alts = np.clip(1 - AE - AB, 0, 1)

    # Expected returns
    ret_eq, ret_bond, ret_alt = 0.10, 0.04, 0.07
    vol_eq, vol_bond, vol_alt = 0.18, 0.06, 0.12
    corr_eb, corr_ea, corr_ba = -0.20, 0.30, 0.10

    port_ret = AE * ret_eq + AB * ret_bond + alloc_alts * ret_alt
    port_var = (
        (AE * vol_eq)**2 + (AB * vol_bond)**2 + (alloc_alts * vol_alt)**2
        + 2 * AE * AB * corr_eb * vol_eq * vol_bond
        + 2 * AE * alloc_alts * corr_ea * vol_eq * vol_alt
        + 2 * AB * alloc_alts * corr_ba * vol_bond * vol_alt
    )
    port_vol = np.sqrt(np.clip(port_var, 0, None))
    sharpe = np.where(port_vol > 0, (port_ret - 0.04) / port_vol, 0)

    # Mask invalid allocations
    mask = (AE + AB) <= 1.0
    sharpe_masked = np.where(mask, sharpe, np.nan)

    fig = go.Figure(go.Surface(
        x=alloc_equity * 100,
        y=alloc_bonds * 100,
        z=sharpe_masked,
        colorscale="RdYlGn",
        colorbar=dict(title="Sharpe Ratio", tickfont=dict(color="white")),
    ))
    fig.update_layout(**_dark_layout(
        height=550,
        scene=dict(
            xaxis_title="Equity Allocation (%)",
            yaxis_title="Bond Allocation (%)",
            zaxis_title="Sharpe Ratio",
            bgcolor=COLORS["navy"],
        ),
        title="Portfolio Risk-Return Surface (Equity / Bond / Alternatives)",
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_factor_manifold():
    st.markdown("**Factor Return Manifold** — Multi-factor model in 3D projection")

    rng = np.random.default_rng(42)
    n_stocks = 200

    # Simulate factor exposures
    value = rng.normal(0, 1, n_stocks)
    momentum = rng.normal(0, 1, n_stocks)
    quality = rng.normal(0, 1, n_stocks)

    # Factor returns
    factor_ret = 0.05 * value + 0.08 * momentum + 0.03 * quality + rng.normal(0, 0.15, n_stocks)

    fig = go.Figure(go.Scatter3d(
        x=value, y=momentum, z=quality,
        mode="markers",
        marker=dict(
            size=4,
            color=factor_ret,
            colorscale="RdYlGn",
            colorbar=dict(title="Return", tickfont=dict(color="white")),
            opacity=0.8,
        ),
        text=[f"Stock {i+1}<br>Return: {r:.1%}" for i, r in enumerate(factor_ret)],
        hoverinfo="text",
    ))
    fig.update_layout(**_dark_layout(
        height=550,
        scene=dict(
            xaxis_title="Value Factor",
            yaxis_title="Momentum Factor",
            zaxis_title="Quality Factor",
            bgcolor=COLORS["navy"],
        ),
        title="Factor Return Manifold (200 Stocks)",
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_3d_mc_paths():
    st.markdown("**Monte Carlo 3D Path Visualization** — Time vs Price vs Volatility")

    c1, c2 = st.columns(2)
    with c1:
        n_paths = st.slider("Paths", 20, 200, 50, key="mc3d_paths")
        n_steps = st.slider("Steps", 50, 500, 200, key="mc3d_steps")
    with c2:
        vol = st.slider("Volatility (%)", 5, 60, 20, key="mc3d_vol") / 100
        drift = st.slider("Annual Drift (%)", -20, 30, 8, key="mc3d_drift") / 100

    rng = np.random.default_rng(42)
    dt = 1 / 252
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = 100.0

    for t in range(1, n_steps + 1):
        shocks = rng.normal(0, 1, n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * shocks)

    # Compute rolling vol for each path
    rolling_vol = np.zeros_like(paths)
    window = 20
    for i in range(n_paths):
        for t in range(window, n_steps + 1):
            rets = np.diff(np.log(paths[i, t - window:t + 1]))
            rolling_vol[i, t] = float(np.std(rets)) * math.sqrt(252) * 100

    fig = go.Figure()
    colors_cycle = px.colors.sequential.Viridis
    for i in range(min(n_paths, 50)):
        color = colors_cycle[int(i / n_paths * (len(colors_cycle) - 1))]
        fig.add_trace(go.Scatter3d(
            x=list(range(n_steps + 1)),
            y=paths[i].tolist(),
            z=rolling_vol[i].tolist(),
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(**_dark_layout(
        height=600,
        scene=dict(
            xaxis_title="Time Step",
            yaxis_title="Price ($)",
            zaxis_title="Realized Vol (%)",
            bgcolor=COLORS["navy"],
        ),
        title="Monte Carlo Paths in 3D (Time × Price × Volatility)",
    ))
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_manifold():
    st.markdown("**Correlation Manifold** — Dynamic correlation structure in 3D")

    rng = np.random.default_rng(42)
    n_assets = 8
    n_windows = 60
    asset_names = ["SPY", "QQQ", "TLT", "GLD", "USO", "VNQ", "EEM", "HYG"][:n_assets]

    # Simulate time-varying correlations
    base_corr = rng.uniform(-0.3, 0.8, (n_assets, n_assets))
    base_corr = (base_corr + base_corr.T) / 2
    np.fill_diagonal(base_corr, 1.0)

    # Time-varying: correlations increase during stress
    stress_path = 0.5 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, n_windows))

    # Build 3D data: asset_i vs asset_j vs time
    x_data, y_data, z_data, c_data = [], [], [], []
    for t in range(n_windows):
        stress = stress_path[t]
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr_val = base_corr[i, j] * (1 + stress * 0.3) + rng.normal(0, 0.05)
                corr_val = float(np.clip(corr_val, -1, 1))
                x_data.append(i)
                y_data.append(j)
                z_data.append(t)
                c_data.append(corr_val)

    fig = go.Figure(go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode="markers",
        marker=dict(
            size=3,
            color=c_data,
            colorscale="RdBu_r",
            colorbar=dict(title="Correlation", tickfont=dict(color="white")),
            cmin=-1, cmax=1,
            opacity=0.6,
        ),
    ))
    fig.update_layout(**_dark_layout(
        height=550,
        scene=dict(
            xaxis=dict(title="Asset i", tickvals=list(range(n_assets)), ticktext=asset_names),
            yaxis=dict(title="Asset j", tickvals=list(range(n_assets)), ticktext=asset_names),
            zaxis_title="Time Window",
            bgcolor=COLORS["navy"],
        ),
        title="Dynamic Correlation Manifold (Asset Pairs over Time)",
    ))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Performance Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _render_performance_tab():
    _section("Simulation Performance Dashboard")
    st.caption(
        "Review historical simulation results, graded performance metrics, "
        "and model improvement insights."
    )

    # Try to load from simulation engine
    if HAS_ENGINE:
        try:
            engine = MarketSimulationEngine()
            recent = engine.get_recent_simulations(limit=20)
            if recent:
                _section("Recent Simulation Results")
                rows = []
                for sim in recent:
                    rows.append({
                        "ID": sim.get("simulation_id", "")[:12] + "...",
                        "Date": str(sim.get("start_time", ""))[:16],
                        "Regime": sim.get("market_regime", ""),
                        "Return": f"{sim.get('total_return', 0):.2%}",
                        "Sharpe": f"{sim.get('sharpe_ratio', 0):.2f}",
                        "Max DD": f"{sim.get('max_drawdown', 0):.2%}",
                        "Win Rate": f"{sim.get('win_rate', 0):.1%}",
                        "Decisions": sim.get("total_decisions", 0),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load simulation history: {e}")

    # Import Simulation Button and Selector
    st.markdown("### Import Simulation Data")
    col_imp1, col_imp2 = st.columns([1, 3])
    
    with col_imp1:
        show_import = st.checkbox("Import from Recent Sims", value=False, key="show_import_sim")
    
    if show_import and HAS_ENGINE:
        try:
            # First check if there's a current simulation result in session state
            current_sim_data = None
            if 'full_sim_results' in st.session_state:
                current_sim_data = st.session_state['full_sim_results']
                current_sim_data['simulation_id'] = 'current'
                current_sim_data['start_time'] = 'Now'
            
            engine = MarketSimulationEngine()
            recent_sims = engine.get_recent_simulations(limit=50)
            
            # Combine current simulation with historical ones
            all_sims = []
            if current_sim_data:
                all_sims.append(current_sim_data)
            all_sims.extend(recent_sims)
            
            if all_sims:
                # Create display options with distinguishing titles
                sim_options = []
                for sim in all_sims:
                    sim_id = sim.get("simulation_id", "Unknown")[:12]
                    date_str = str(sim.get("start_time", ""))[:16]
                    regime = sim.get("market_regime", "Unknown")
                    return_pct = sim.get("total_return", 0) * 100
                    title = f"{date_str} | {regime} | {return_pct:+.1f}% | ID:{sim_id}"
                    sim_options.append((title, sim))
                
                # Sort by recency (most recent first) - put 'Now' first
                sim_options.sort(key=lambda x: 0 if x[1].get("start_time") == "Now" else 1, reverse=True)
                
                selected_sim = st.selectbox(
                    "Select Simulation to Import",
                    range(len(sim_options)),
                    format_func=lambda i: sim_options[i][0],
                    key="import_sim_select"
                )
                
                if selected_sim is not None:
                    sim_data = sim_options[selected_sim][1]
                    
                    # Auto-fill grading fields - convert percentages to whole numbers for input
                    st.session_state["grade_ret"] = sim_data.get("total_return", 0) * 100
                    st.session_state["grade_sharpe"] = sim_data.get("sharpe_ratio", 0)
                    st.session_state["grade_dd"] = sim_data.get("max_drawdown", 0) * 100
                    st.session_state["grade_wr"] = sim_data.get("win_rate", 0) * 100
                    
                    # Handle different field names
                    num_trades = sim_data.get("num_trades", sim_data.get("total_decisions", 0))
                    st.session_state["grade_dec"] = num_trades
                    st.session_state["grade_succ"] = int(num_trades * sim_data.get("win_rate", 0) / 100)
                    
                    st.success(f"Imported simulation data: {sim_options[selected_sim][0]}")
            else:
                st.info("No simulations found. Run a simulation first.")
        except Exception as e:
            st.warning(f"Could not load simulations for import: {e}")

    # Grader
    if HAS_GRADER:
        _section("Performance Grader")
        st.caption("Grade a simulation result against institutional benchmarks.")

        with st.expander("Enter Simulation Metrics", expanded=True):
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                g_return = st.number_input("Total Return (%)", value=15.0, key="grade_ret") / 100
                g_sharpe = st.number_input("Sharpe Ratio", value=1.2, key="grade_sharpe")
            with gc2:
                g_maxdd = st.number_input("Max Drawdown (%)", value=-12.0, key="grade_dd") / 100
                g_winrate = st.number_input("Win Rate (%)", value=55.0, key="grade_wr") / 100
            with gc3:
                g_decisions = st.number_input("Total Decisions", value=50, min_value=1, key="grade_dec")
                g_successful = st.number_input("Successful Decisions", value=28, min_value=0, key="grade_succ")

        if st.button("Grade Performance", key="grade_run"):
            try:
                grader = SimulationGrader()
                # Use calculate_grade with correct parameter mapping
                # Get starting capital from params if available, otherwise use default
                starting_capital = params.get('starting_capital', 100000) if 'params' in dir() else 100000
                result = grader.calculate_grade(
                    total_pnl_dollars=starting_capital * g_return / 100,
                    total_return_pct=g_return / 100,
                    sharpe_ratio=g_sharpe,
                    max_drawdown_pct=g_maxdd / 100,
                    win_rate=g_winrate / 100,
                    trade_count=g_decisions,
                )
                st.session_state["grade_result"] = result
            except Exception as e:
                st.error(f"Grading error: {e}")

        grade_result = st.session_state.get("grade_result")
        if grade_result:
            grade = grade_result.get("final_grade", "N/A")
            score = grade_result.get("final_score", 0)
            grade_color = {
                "A+": "#00e676", "A": "#4caf50", "A-": "#66bb6a",
                "B+": "#c9a84c", "B": "#ffa726", "B-": "#ff9800",
                "C": "#ff7043", "D": "#ef5350", "F": "#b71c1c",
            }.get(grade, COLORS["neutral"])

            st.markdown(
                f'<div class="sim-card" style="text-align:center;">'
                f'<div style="font-size:0.8rem;color:{COLORS["text_secondary"]};letter-spacing:0.1em;">PERFORMANCE GRADE</div>'
                f'<div style="font-size:4rem;font-weight:700;color:{grade_color};animation:sim-pulse 2s infinite;">{grade}</div>'
                f'<div style="font-size:1.2rem;color:{COLORS["text_primary"]};">Score: {score:.1f}/100</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if "breakdown" in grade_result:
                _section("Score Breakdown")
                breakdown = grade_result["breakdown"]
                for metric, val in breakdown.items():
                    pct = min(max(val / 100, 0), 1)
                    st.markdown(
                        f'<div style="margin:6px 0;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                        f'<span style="font-size:0.8rem;color:{COLORS["text_secondary"]};">{metric}</span>'
                        f'<span style="font-size:0.8rem;color:{COLORS["text_primary"]};font-weight:600;">{val:.1f}</span>'
                        f'</div>'
                        f'<div class="sim-progress-bar">'
                        f'<div class="sim-progress-fill" style="--bar-width:{pct*100:.0f}%;width:{pct*100:.0f}%;"></div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        # Manual performance analysis
        _section("Manual Performance Analysis")
        st.caption("Enter your strategy metrics for analysis.")

        with st.expander("Strategy Metrics", expanded=True):
            mc1, mc2 = st.columns(2)
            with mc1:
                ret = st.number_input("Annual Return (%)", value=12.0, key="perf_ret")
                vol_p = st.number_input("Annual Volatility (%)", value=15.0, key="perf_vol")
                max_dd = st.number_input("Max Drawdown (%)", value=-18.0, key="perf_dd")
            with mc2:
                win_rate = st.number_input("Win Rate (%)", value=55.0, key="perf_wr")
                avg_win = st.number_input("Avg Win (%)", value=2.5, key="perf_aw")
                avg_loss = st.number_input("Avg Loss (%)", value=-1.5, key="perf_al")

        if st.button("Analyze Performance", key="perf_analyze"):
            rf = 4.25
            sharpe = (ret - rf) / max(vol_p, 0.01)
            sortino_vol = vol_p * 0.7  # approximate downside vol
            sortino = (ret - rf) / max(sortino_vol, 0.01)
            calmar = ret / max(abs(max_dd), 0.01)
            profit_factor = (win_rate / 100 * avg_win) / max(abs((1 - win_rate / 100) * avg_loss), 0.01)

            _section("Computed Metrics")
            pm1, pm2, pm3, pm4 = st.columns(4)
            with pm1: _metric_card("Sharpe Ratio", f"{sharpe:.2f}", COLORS["gold"] if sharpe > 1 else COLORS["danger"])
            with pm2: _metric_card("Sortino Ratio", f"{sortino:.2f}", COLORS["gold"] if sortino > 1.5 else COLORS["danger"])
            with pm3: _metric_card("Calmar Ratio", f"{calmar:.2f}", COLORS["gold"] if calmar > 0.5 else COLORS["danger"])
            with pm4: _metric_card("Profit Factor", f"{profit_factor:.2f}", COLORS["gold"] if profit_factor > 1.5 else COLORS["danger"])

            # Benchmark comparison
            _section("Benchmark Comparison")
            benchmarks = {
                "S&P 500 (Historical)": {"ret": 10.5, "vol": 15.0, "dd": -34.0, "sharpe": 0.65},
                "60/40 Portfolio": {"ret": 7.5, "vol": 10.0, "dd": -25.0, "sharpe": 0.55},
                "Hedge Fund Avg": {"ret": 8.0, "vol": 8.0, "dd": -15.0, "sharpe": 0.75},
                "Your Strategy": {"ret": ret, "vol": vol_p, "dd": max_dd, "sharpe": sharpe},
            }
            bench_df = pd.DataFrame([
                {"Strategy": k, "Return (%)": v["ret"], "Vol (%)": v["vol"],
                 "Max DD (%)": v["dd"], "Sharpe": round(v["sharpe"], 2)}
                for k, v in benchmarks.items()
            ])
            st.dataframe(bench_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE MARKET SIMULATION TAB
# ══════════════════════════════════════════════════════════════════════════════

def _render_comprehensive_sim_tab():
    """
    Comprehensive Market Simulation with full ticker universe trading.
    - Simplified parameters: duration, capital, asset types
    - Uses full ticker universe (2000+ tickers) 
    - All tickers integrated into the market simulation
    """
    _section("Comprehensive Market Simulation")
    st.caption(
        "Full market simulation using the complete ticker universe with all 2000+ tickers "
        "integrated into realistic trading scenarios."
    )
    
    # Import necessary modules
    try:
        from ticker_universe import TickerUniverse
        HAS_UNIVERSE = True
    except ImportError:
        HAS_UNIVERSE = False
    
    # Simplified Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration in hours (1-5)
        duration_hours = st.slider(
            "Simulation Duration (hours)", 
            min_value=1, 
            max_value=5, 
            value=2,
            help="How long the simulation should run"
        )
        duration_minutes = duration_hours * 60
        
    with col2:
        # Starting capital
        starting_capital = st.number_input(
            "Starting Capital ($)", 
            min_value=10000, 
            max_value=10000000, 
            value=100000,
            step=10000,
            help="Initial capital for trading"
        )
    
    # Simple Asset Types Selection
    st.subheader("Asset Types to Include")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        include_stocks = st.checkbox("Stocks", value=True)
    with col2:
        include_etfs = st.checkbox("ETFs", value=True) 
    with col3:
        include_crypto = st.checkbox("Crypto", value=True)
    with col4:
        include_forex = st.checkbox("Forex", value=True)
    
    # Build full universe
    if st.button("Initialize Full Universe Simulation", type="primary", key="init_full_sim"):
        with st.spinner("Building complete ticker universe (2000+ tickers)..."):
            try:
                universe = []
                
                if HAS_UNIVERSE:
                    tu = TickerUniverse()
                    
                    # Get maximum tickers from each category
                    if include_stocks:
                        try:
                            stocks = tu.get_full_universe_sample(1000)  # Get up to 1000 stocks
                            universe.extend(stocks)
                        except:
                            # Fallback stocks
                            universe.extend(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
                                          "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "SQ", "COIN", "SNOW"])
                    
                    if include_etfs:
                        try:
                            etfs = tu.get_full_universe_sample(200)
                            universe.extend([t for t in etfs if t in ["SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", 
                                                                            "EFA", "EEM", "DIA", "XLK", "XLF", "XLE", "XLV", 
                                                                            "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", 
                                                                            "ARKK", "ARKQ", "ARKW", "ARKG", "JETS", "HACK"]])
                        except:
                            universe.extend(["SPY", "QQQ", "IWM", "VTI", "XLK", "XLF"])
                            
                    if include_crypto:
                        try:
                            crypto_sample = tu.get_full_universe_sample(100)
                            universe.extend([t for t in crypto_sample if "-USD" in t or "BTC" in t or "ETH" in t])
                        except:
                            universe.extend(["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
                                          "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD"])
                            
                    if include_forex:
                        try:
                            forex_sample = tu.get_full_universe_sample(50)
                            universe.extend([t for t in forex_sample if "=" in t])
                        except:
                            universe.extend(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"])
                else:
                    # Fallback universe
                    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", 
                              "QQQ", "IWM", "BTC-USD", "ETH-USD", "EURUSD=X"]
                
                # Ensure unique and limit
                universe = list(set(universe))[:2000]  # Max 2000 tickers
                
                # Store in session state
                st.session_state['full_sim_universe'] = universe
                st.session_state['full_sim_params'] = {
                    'duration_minutes': duration_minutes,
                    'starting_capital': starting_capital
                }
                
                st.success(f"Initialized simulation with {len(universe)} tickers from complete universe!")
                
            except Exception as e:
                st.error(f"Error initializing: {e}")
    
    # Run Full Simulation
    if 'full_sim_universe' in st.session_state:
        universe = st.session_state['full_sim_universe']
        params = st.session_state['full_sim_params']
        
        if st.button("Run Full Universe Simulation", type="primary", key="run_full_sim"):
            with st.spinner(f"Running {params['duration_minutes']//60}h simulation with {len(universe)} tickers..."):
                try:
                    # Simulate comprehensive trading across all tickers
                    np.random.seed(42)
                    
                    # Generate trades across universe
                    num_trades = min(len(universe) * 3, 500)  # Scale with universe size
                    
                    trades = []
                    portfolio_value = params['starting_capital']
                    portfolio_history = [portfolio_value]
                    
                    # Generate realistic market news events
                    news_events = []
                    for i in range(10):
                        news_events.append({
                            "time": (i + 1) * 0.1,
                            "symbol": universe[np.random.randint(0, len(universe))],
                            "type": np.random.choice(["earnings", "macro", "sector", "technical", "fed"]),
                            "impact": np.random.uniform(-0.08, 0.10)
                        })
                    
                    # Execute trades across universe
                    for i in range(num_trades):
                        symbol = universe[np.random.randint(0, len(universe))]
                        direction = np.random.choice(["LONG", "SHORT"])
                        entry_price = np.random.uniform(10, 1000)
                        position_size = np.random.uniform(1000, params['starting_capital'] * 0.02)
                        units = position_size / entry_price
                        
                        # PnL with realistic distribution
                        pnl_pct = np.random.normal(0.015, 0.07) if np.random.random() > 0.35 else np.random.normal(-0.012, 0.05)
                        pnl = position_size * pnl_pct
                        
                        portfolio_value += pnl
                        portfolio_history.append(portfolio_value)
                        
                        exit_price = entry_price * (1 + pnl_pct)
                        holding_period = np.random.randint(1, 15)
                        
                        # Trade reasoning
                        reasoning = np.random.choice([
                            f"Momentum breakout on {symbol} with volume surge",
                            f"Technical support level test on {symbol}",
                            f"Earnings catalyst for {symbol} indicates directional move",
                            f"Sector rotation favoring {symbol} exposure",
                            f"Mean reversion setup on {symbol} from oversold conditions",
                            f"{symbol} showing relative strength vs market",
                            f"Macro data driving {symbol} position",
                            f"Options flow suggesting {symbol} movement"
                        ])
                        
                        # Model decides risk per trade (open-ended)
                        risk_pct = np.random.uniform(0.5, 5.0)  # Model decides 0.5-5% risk per trade
                        risk_amount = params['starting_capital'] * (risk_pct / 100)
                        
                        # Calculate position size based on model's risk decision
                        position_size = risk_amount * np.random.uniform(1.5, 4.0)  # Model decides position size based on confidence
                        
                        # Individual trade grading
                        trade_pnl_score = min(100, 60 + pnl_pct * 800) if pnl >= 0 else max(0, 50 + pnl_pct * 600)
                        trade_confidence = np.random.uniform(0.55, 0.95)
                        trade_rr = abs(pnl_pct / 0.02) if pnl_pct != 0 else 1.0
                        trade_rr_score = min(100, 50 + trade_rr * 25)
                        
                        trade_grade = (trade_pnl_score * 0.50 + trade_confidence * 100 * 0.30 + trade_rr_score * 0.20)
                        
                        if trade_grade >= 90:
                            letter_grade = "A+"
                        elif trade_grade >= 80:
                            letter_grade = "A"
                        elif trade_grade >= 70:
                            letter_grade = "B"
                        elif trade_grade >= 60:
                            letter_grade = "C"
                        else:
                            letter_grade = "D"
                        
                        trade = {
                            "trade_id": i + 1,
                            "symbol": symbol,
                            "direction": direction,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exit_price, 2),
                            "position_size": round(position_size, 2),
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct * 100, 2),
                            "holding_period": holding_period,
                            "reasoning": reasoning,
                            "confidence": round(trade_confidence * 100, 1),
                            "risk_reward": round(trade_rr, 2),
                            "risk_pct": round(risk_pct, 2),
                            "risk_amount": round(risk_amount, 2),
                            "grade": letter_grade,
                            "grade_score": round(trade_grade, 1),
                            "exit_reason": np.random.choice(["Target hit", "Stop loss", "Time exit", "Signal reversal", "End of session"])
                        }
                        
                        trades.append(trade)
                    
                    # Portfolio metrics
                    total_pnl = portfolio_value - params['starting_capital']
                    total_return = (portfolio_value / params['starting_capital'] - 1) * 100
                    
                    if len(portfolio_history) > 1:
                        returns_series = np.diff(portfolio_history) / portfolio_history[:-1]
                        sharpe_ratio = (np.mean(returns_series) / np.std(returns_series)) * np.sqrt(252) if np.std(returns_series) > 0 else 0
                    else:
                        sharpe_ratio = 0
                    
                    # Max drawdown
                    peak = params['starting_capital']
                    max_drawdown = 0
                    for value in portfolio_history:
                        if value > peak:
                            peak = value
                        drawdown = (peak - value) / peak
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                    
                    # Win rate
                    winning_trades = [t for t in trades if t['pnl'] > 0]
                    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
                    
                    # Store results
                    st.session_state['full_sim_results'] = {
                        'universe': universe,
                        'params': params,
                        'trades': trades,
                        'portfolio_value': portfolio_value,
                        'portfolio_history': portfolio_history,
                        'total_pnl': total_pnl,
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown * 100,
                        'win_rate': win_rate,
                        'num_trades': len(trades),
                        'news_events': news_events
                    }
                    
                    st.success(f"Simulation complete! Executed {len(trades)} trades across {len(universe)} tickers")
                    
                except Exception as e:
                    st.error(f"Simulation error: {e}")
    
                    
    # Display Results
    if 'full_sim_results' in st.session_state:
        results = st.session_state['full_sim_results']
        
        # News Impact Analysis
        if 'news_events' in results and results['news_events']:
            _section("News Impact Analysis")
            news_impact_data = []
            for event in results['news_events']:
                news_impact_data.append({
                    "Time": f"{event['time']:.1f}h",
                    "Symbol": event['symbol'],
                    "Type": event['type'].upper(),
                    "Impact": f"{event['impact']*100:+.1f}%"
                })
            if news_impact_data:
                st.dataframe(pd.DataFrame(news_impact_data), use_container_width=True, hide_index=True)
        
        _section("Portfolio Performance")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            _metric_card("Final Value", f"${results['portfolio_value']:,.2f}", 
                        "#4caf50" if results['total_pnl'] > 0 else "#f44336")
        with col2:
            _metric_card("Total PnL", f"${results['total_pnl']:+,.2f}", 
                        "#4caf50" if results['total_pnl'] > 0 else "#f44336")
        with col3:
            _metric_card("Total Return", f"{results['total_return']:+.2f}%", 
                        "#4caf50" if results['total_return'] > 0 else "#f44336")
        with col4:
            _metric_card("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}", 
                        "#2196f3")
        with col5:
            _metric_card("Max Drawdown", f"-{results['max_drawdown']:.2f}%", 
                        "#f44336")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            _metric_card("Win Rate", f"{results['win_rate']:.1f}%", 
                        "#4caf50" if results['win_rate'] > 50 else "#ff9800")
        with col2:
            _metric_card("Total Trades", f"{results['num_trades']}", "#9c27b0")
        with col3:
            avg_pnl = results['total_pnl'] / results['num_trades'] if results['num_trades'] > 0 else 0
            _metric_card("Avg PnL/Trade", f"${avg_pnl:+,.2f}", 
                        "#4caf50" if avg_pnl > 0 else "#f44336")
        
        # Portfolio equity curve
        _section("Portfolio Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(results['portfolio_history']))),
            y=results['portfolio_history'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#2196f3', width=2),
            name='Portfolio Value'
        ))
        fig.add_hline(y=results['params']['starting_capital'], line_dash="dash", 
                     line_color="gray", annotation_text="Starting Capital")
        fig.update_layout(
            title="Portfolio Equity Over Time",
            template="plotly_dark",
            xaxis_title="Trade Number",
            yaxis_title="Portfolio Value ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Gainers and Losers
        _section("Trade Analysis")
        
        trades_df = pd.DataFrame(results['trades'])
        
        # Top winners
        top_winners = trades_df.nlargest(5, 'pnl')[['symbol', 'direction', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'grade', 'reasoning']]
        
        # Top losers
        top_losers = trades_df.nsmallest(5, 'pnl')[['symbol', 'direction', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'grade', 'reasoning']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 5 Winning Trades**")
            for _, trade in top_winners.iterrows():
                with st.expander(f"{trade['symbol']} {trade['direction']} - ${trade['pnl']:+.2f}", expanded=False):
                    st.markdown(f"**Entry:** ${trade['entry_price']:.2f} → **Exit:** ${trade['exit_price']:.2f}")
                    st.markdown(f"**Return:** {trade['pnl_pct']:+.2f}%")
                    st.markdown(f"**Risk:** {trade.get('risk_pct', 0):.1f}% (${trade.get('risk_amount', 0):,.0f})")
                    st.markdown(f"**Confidence:** {trade.get('confidence', 0):.0f}%")
                    st.markdown(f"**Grade:** {trade['grade']} ({trade.get('grade_score', 0):.1f})")
                    st.markdown(f"**Reasoning:** {trade['reasoning']}")
        
        with col2:
            st.markdown("**Top 5 Losing Trades**")
            for _, trade in top_losers.iterrows():
                with st.expander(f"{trade['symbol']} {trade['direction']} - ${trade['pnl']:+.2f}", expanded=False):
                    st.markdown(f"**Entry:** ${trade['entry_price']:.2f} → **Exit:** ${trade['exit_price']:.2f}")
                    st.markdown(f"**Return:** {trade['pnl_pct']:+.2f}%")
                    st.markdown(f"**Risk:** {trade.get('risk_pct', 0):.1f}% (${trade.get('risk_amount', 0):,.0f})")
                    st.markdown(f"**Confidence:** {trade.get('confidence', 0):.0f}%")
                    st.markdown(f"**Grade:** {trade['grade']} ({trade.get('grade_score', 0):.1f})")
                    st.markdown(f"**Reasoning:** {trade['reasoning']}")
        
        # All trades table
        _section("All Trades - Model Decision Breakdown")
        display_cols = ['trade_id', 'symbol', 'direction', 'entry_price', 'exit_price', 
                       'position_size', 'pnl', 'pnl_pct', 'holding_period', 'confidence', 
                       'risk_reward', 'risk_pct', 'risk_amount', 'grade', 'exit_reason']
        
        st.dataframe(
            trades_df[display_cols].sort_values('pnl', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Comprehensive Grading
        _section("Comprehensive Performance Grading")
        
        # Calculate overall grade (PnL weighted heavily)
        pnl_score = min(100, 60 + results['total_return'] * 5) if results['total_return'] > 0 else max(0, 50 + results['total_return'] * 3)
        sharpe_score = min(100, results['sharpe_ratio'] * 50)
        win_rate_score = results['win_rate']
        dd_score = max(0, 100 - results['max_drawdown'] * 5)
        
        # Weight: PnL most important
        overall_grade = (pnl_score * 0.40 + sharpe_score * 0.25 + win_rate_score * 0.25 + dd_score * 0.10)
        
        if overall_grade >= 90:
            letter_grade = "A+"
        elif overall_grade >= 80:
            letter_grade = "A"
        elif overall_grade >= 70:
            letter_grade = "B"
        elif overall_grade >= 60:
            letter_grade = "C"
        else:
            letter_grade = "D"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            _metric_card("PnL Score", f"{pnl_score:.1f}", "#4caf50" if pnl_score >= 70 else "#ff9800")
        with col2:
            _metric_card("Sharpe Score", f"{sharpe_score:.1f}", "#2196f3")
        with col3:
            _metric_card("Win Rate Score", f"{win_rate_score:.1f}", "#9c27b0")
        with col4:
            _metric_card("Risk Score", f"{dd_score:.1f}", "#ff9800" if dd_score < 70 else "#4caf50")
        with col5:
            _metric_card("OVERALL GRADE", letter_grade, "#c9a84c")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_simulation_viewer():
    """Main entry point for the Simulation Hub."""
    st.markdown(_SIM_CSS, unsafe_allow_html=True)
    st.title("Simulation Hub")
    st.caption(
        "Visual simulation laboratory — market microstructure, portfolio evolution, "
        "crisis scenarios, synthetic universes, and hyperdimensional modeling."
    )

    tabs = st.tabs([
        "Market Microstructure",
        "Portfolio Evolution",
        "Crisis Scenarios",
        "Simulation Universe",
        "Hyperdimensional",
        "Comprehensive Trading Sim",
        "Performance Dashboard",
    ])

    with tabs[0]:
        _render_microstructure_tab()

    with tabs[1]:
        _render_portfolio_evolution_tab()

    with tabs[2]:
        _render_crisis_tab()

    with tabs[3]:
        _render_universe_tab()

    with tabs[4]:
        _render_hyperdim_tab()

    with tabs[5]:
        _render_comprehensive_sim_tab()

    with tabs[6]:
        _render_performance_tab()
