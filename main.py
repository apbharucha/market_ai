import matplotlib
import streamlit as st

matplotlib.use("Agg")
import pandas as pd
import plotly.graph_objs as go

try:
    from data_sources import get_fresh_quote

    HAS_FRESH = True
except ImportError:
    HAS_FRESH = False

# Only import lightweight, always-needed modules at startup
from octavian_theme import COLORS, apply_theme, render_header, section_header
from trader_profile import (
    show_personalized_dashboard,
    show_profile_settings,
    show_trader_selection,
)

try:
    from data_sources import get_realtime_price, get_realtime_prices_batch

    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False
    get_realtime_price = None
    get_realtime_prices_batch = None

st.set_page_config(layout="wide", page_title="Octavian Terminal", page_icon="O")

# Apply professional theme
apply_theme()

# --- Sidebar Navigation ---
try:
    st.sidebar.image("/Users/aavibharucha/Documents/market_ai/logo.png", width=280)
except Exception:
    st.sidebar.markdown(
        f'<div style="text-align:center;padding:16px 12px 8px;">'
        f'<span style="color:{COLORS["gold"]};font-family:Inter,sans-serif;font-size:1.3rem;'
        f'font-weight:700;letter-spacing:4px;">OCTAVIAN</span>'
        f'<br><span style="color:{COLORS["lavender"]};font-size:0.65rem;'
        f'letter-spacing:2px;font-weight:300;">TERMINAL v4.0</span></div>',
        unsafe_allow_html=True,
    )

st.sidebar.caption("Multi-Asset Market Intelligence")

# Navigation — clean, professional labels (no emojis)
nav_options = [
    "Dashboard",
    "Watchlist",
    "Market Scanner",
    "Symbol Analysis",
    "Chart Analysis",
    "Intelligence Center",
    "Financial Model Generator",
    "Daily Briefing",
    "Quant Portal",
    "Strategy Research Lab",
    "Paper Trading",
    "Simulation Hub",
    "Trader Profile",
    "Settings & Analytics",
]
selection = st.sidebar.radio("Navigation", nav_options, label_visibility="collapsed")

st.sidebar.markdown("---")
show_trader_selection(key_suffix="_sidebar")

# --- Main Content Router ---

if selection == "Dashboard":
    st.title("Live Market Dashboard")

    dash_tabs = st.tabs(["My View", "Market Overview", "Breaking Trades"])

    with dash_tabs[0]:
        show_personalized_dashboard()

    with dash_tabs[1]:
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = pd.Timestamp.now()

        col_refresh1, col_refresh2 = st.columns([1, 4])
        with col_refresh1:
            if st.button("Refresh Now", key="refresh_live_dashboard"):
                st.session_state.last_refresh = pd.Timestamp.now()
                st.rerun()

        time_since_refresh = (
            pd.Timestamp.now() - st.session_state.last_refresh
        ).total_seconds()
        st.caption(
            f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')} ({int(time_since_refresh)}s ago)"
        )

        section_header("Market Overview")

        #  Auto-refreshing price tickers (every 30s, no full page reload)
        @st.fragment(run_every=30)
        def _live_index_tickers():
            indices = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC",
                "DOW": "^DJI",
                "VIX": "^VIX",
            }
            idx_data = {}

            # Batch fetch all index prices at once (shared cache, fewer round-trips)
            if HAS_REALTIME and get_realtime_prices_batch:
                try:
                    batch = get_realtime_prices_batch(list(indices.values()))
                    for name, symbol in indices.items():
                        result = batch.get(symbol)
                        if result:
                            price, prev = result
                            if price and price > 0 and prev and prev > 0:
                                change = (price - prev) / prev * 100
                                idx_data[name] = {
                                    "value": float(price),
                                    "change": float(change),
                                }
                except Exception:
                    pass

            # Fallback to individual fetches if batch failed
            if not idx_data and HAS_REALTIME and get_realtime_price:
                for name, symbol in indices.items():
                    try:
                        price, prev = get_realtime_price(symbol)
                        if price and price > 0 and prev and prev > 0:
                            change = (price - prev) / prev * 100
                            idx_data[name] = {
                                "value": float(price),
                                "change": float(change),
                            }
                    except Exception:
                        pass

            if idx_data:
                cols = st.columns(len(idx_data))
                for i, (name, data) in enumerate(idx_data.items()):
                    with cols[i]:
                        st.metric(
                            name, f"{data.get('value', 0):,.2f}", f"{data.get('change', 0):+.2f}%"
                        )
                st.caption(f"Live | {pd.Timestamp.now().strftime('%H:%M:%S')}")
            else:
                st.warning("Unable to fetch live prices. Retrying...")

        _live_index_tickers()

        #  Static chart section (only updates on full refresh)
        @st.cache_data(ttl=300, show_spinner=False)
        def _fetch_index_chart_data():
            """Fetch 5-day history for index overview chart (cached 5 min)."""
            _indices = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC",
                "DOW": "^DJI",
                "VIX": "^VIX",
            }
            result = {}
            # Batch download all indices at once
            try:
                import yfinance as yf

                tickers = list(_indices.values())
                data = yf.download(
                    tickers,
                    period="5d",
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                if data is not None and not data.empty:
                    for name, symbol in _indices.items():
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                if symbol in data.columns.get_level_values(0):
                                    df = data[symbol].dropna()
                                else:
                                    continue
                            else:
                                df = data.dropna()
                            if (
                                df is not None
                                and not df.empty
                                and "Close" in df.columns
                            ):
                                result[name] = df
                        except Exception:
                            pass
                if result:
                    return result
            except Exception:
                pass
            # Fallback: individual fetches
            for name, symbol in _indices.items():
                try:
                    df = None
                    if HAS_FRESH:
                        df = get_fresh_quote(symbol, period="5d")
                    if df is None or df.empty:
                        import yfinance as yf

                        df = yf.Ticker(symbol).history(period="5d")
                        if df is not None and isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                            if df.columns.duplicated().any():
                                df = df.loc[:, ~df.columns.duplicated(keep="first")]
                    if df is not None and not df.empty and "Close" in df.columns:
                        result[name] = df
                except Exception:
                    pass
            return result

        indices_chart_data = _fetch_index_chart_data()

        if indices_chart_data:
            section_header("Performance Trends -- 5 Day")
            fig_indices = go.Figure()
            chart_colors = [
                COLORS["gold"],
                COLORS["lavender"],
                COLORS["white_soft"],
                COLORS["danger"],
            ]
            for idx, (name, df) in enumerate(indices_chart_data.items()):
                close_col = df["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_col = close_col.dropna().astype(float)
                daily = close_col.copy()
                daily.index = close_col.index.normalize()
                daily = daily.groupby(daily.index).last()
                if len(daily) < 2:
                    continue
                base_val = float(daily.iloc[0])
                if base_val <= 0:
                    continue
                normalized = (daily - base_val) / base_val * 100
                fig_indices.add_trace(
                    go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode="lines+markers",
                        name=name,
                        line=dict(color=chart_colors[idx % len(chart_colors)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"<b>{name}</b><br>%{{x|%b %d}}<br>%{{y:+.2f}}%<extra></extra>",
                    )
                )
            fig_indices.add_hline(
                y=0, line_dash="dash", line_color=COLORS["border"], opacity=0.5
            )
            fig_indices.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor=COLORS["navy"],
                plot_bgcolor=COLORS["navy_light"],
                font_color=COLORS["text_primary"],
                margin=dict(l=0, r=0, t=10, b=0),
                hovermode="x unified",
                yaxis_title="5-Day Change (%)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color=COLORS["text_secondary"]),
                ),
            )
            st.plotly_chart(fig_indices, use_container_width=True)

        section_header("Quick Market Scan")
        from futures_leaderboard import futures_rank
        from sector_scanner import scan_sectors

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Top Sector Movers")
            sectors = scan_sectors()
            if not sectors.empty:
                st.dataframe(
                    sectors[["Sector", "TrendScore"]].head(5),
                    hide_index=True,
                    use_container_width=True,
                )
        with col2:
            st.caption("Top Futures Movers")
            futures = futures_rank()
            if not futures.empty:
                st.dataframe(
                    futures[["Contract", "TrendScore"]].head(5),
                    hide_index=True,
                    use_container_width=True,
                )

        # Market Movers - Top/Worst Performers
        st.markdown("---")
        from market_movers import show_market_movers

        show_market_movers()

    with dash_tabs[2]:
        st.subheader("Breaking Trades - High Confidence Setups")
        st.markdown(
            "**Professional-grade trade setups with comprehensive analysis and specifications**"
        )

        # Generate breaking trades
        if st.button(
            "Generate Breaking Trades", type="primary", key="dash_gen_breaking"
        ):
            with st.spinner("Analyzing markets for high-confidence setups..."):
                try:
                    from breaking_trades_generator import BreakingTradesGenerator

                    # Use full ticker universe instead of preset list
                    try:
                        from ticker_universe import TickerUniverse
                        tu = TickerUniverse()
                        # Get diverse sample from universe
                        watchlist = tu.get_full_universe_sample(50)  # Scan 50 random tickers from universe
                    except:
                        # Fallback to preset list if universe not available
                        watchlist = [
                            "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
                            "META", "AMZN", "SPY", "QQQ", "AMD",
                            "NFLX", "DIS", "V", "MA", "JPM",
                            "UNH", "HD", "PG", "COST", "BA",
                            "MS", "GS", "BLK", "SCHW", "C",
                            "XOM", "CVX", "COP", "SLB", "EOG",
                            "PFE", "JNJ", "UNP", "LIN", "NEE",
                            "AMT", "PLD", "CCI", "EQIX", "PSA",
                            "MCO", "SPGI", "AXP", "USB", "TFC",
                            "BKNG", "CME", "ICE", "SCHD", "VTI",
                            "VOO", "IVV", "VEA", "VWO", "BND"
                        ]

                    gen = BreakingTradesGenerator(min_confidence=55.0)
                    breaking_trades = gen.generate_breaking_trades(
                        watchlist, max_trades=5
                    )

                    if breaking_trades:
                        st.session_state["dashboard_breaking_trades"] = breaking_trades
                        st.success(
                            f"Found {len(breaking_trades)} high-confidence setups!"
                        )
                    else:
                        st.warning(
                            "No high-confidence setups found at this time. Market conditions may not be favorable."
                        )
                except Exception as e:
                    st.error(f"Error generating trades: {e}")

        # Display breaking trades
        if (
            "dashboard_breaking_trades" in st.session_state
            and st.session_state["dashboard_breaking_trades"]
        ):
            breaking_trades = st.session_state["dashboard_breaking_trades"]

            for i, trade in enumerate(breaking_trades):
                # Main card for each trade
                conf_color = (
                    "#00ff00"
                    if trade.confidence_score >= 85
                    else "#ffff00"
                    if trade.confidence_score >= 75
                    else "#ffa500"
                )

                with st.container():
                    st.markdown(
                        f'<div style="background:#161b22;border:2px solid {conf_color};border-radius:8px;padding:20px;margin:15px 0;">',
                        unsafe_allow_html=True,
                    )

                    # Header
                    hcol1, hcol2, hcol3, hcol4 = st.columns([2, 1, 1, 1])
                    with hcol1:
                        st.markdown(
                            f'<div style="font-size:1.8rem;font-weight:700;color:white;">{trade.symbol} - {trade.setup_type}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div style="font-size:0.9rem;color:#888;margin-top:-5px;">{trade.direction} | {trade.market_condition}</div>',
                            unsafe_allow_html=True,
                        )
                    with hcol2:
                        st.markdown(
                            f'<div style="text-align:center;padding:10px;background:#0d1117;border-radius:6px;">'
                            f'<div style="font-size:0.75rem;color:#aaa;">CONFIDENCE</div>'
                            f'<div style="font-size:1.8rem;font-weight:700;color:{conf_color};">{trade.confidence_score:.0f}%</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with hcol3:
                        st.metric("Current Price", f"${trade.current_price:,.2f}")
                    with hcol4:
                        st.metric("R:R Ratio", f"{trade.risk_reward_ratio:.1f}:1")

                    st.markdown("---")

                    # Primary Reason
                    st.markdown(
                        f'<div style="background:#1a1f2e;padding:15px;border-radius:6px;border-left:4px solid {conf_color};margin:10px 0;">'
                        f'<div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;">PRIMARY THESIS</div>'
                        f'<div style="font-size:1.05rem;color:white;font-weight:500;">{trade.primary_reason}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Trade Specs
                    st.markdown("### Trade Specifications")
                    spec_col1, spec_col2, spec_col3 = st.columns(3)

                    with spec_col1:
                        st.markdown("**Entry & Exit Levels**")
                        entry_color = (
                            "#00ff88" if trade.direction == "LONG" else "#ff4444"
                        )
                        st.markdown(
                            f'<div style="background:#0d1117;padding:12px;border-radius:6px;margin:5px 0;">'
                            f'<div style="color:#aaa;font-size:0.8rem;">ENTRY TRIGGER</div>'
                            f'<div style="color:{entry_color};font-size:1.3rem;font-weight:700;">${trade.entry_price:,.2f}</div>'
                            f'<div style="color:#888;font-size:0.75rem;">Enter when price reaches this level</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f'<div style="background:#0d1117;padding:12px;border-radius:6px;margin:5px 0;">'
                            f'<div style="color:#aaa;font-size:0.8rem;">STOP LOSS</div>'
                            f'<div style="color:#ff4444;font-size:1.3rem;font-weight:700;">${trade.stop_loss:,.2f}</div>'
                            f'<div style="color:#888;font-size:0.75rem;">Risk: {abs((trade.entry_price - trade.stop_loss) / trade.entry_price * 100) if trade.entry_price != 0 else 0:.1f}%</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    with spec_col2:
                        st.markdown("**Take Profit Targets**")
                        st.markdown(
                            f'<div style="background:#0d1117;padding:12px;border-radius:6px;margin:5px 0;">'
                            f'<div style="color:#aaa;font-size:0.8rem;">TARGET 1 (Scale 30%)</div>'
                            f'<div style="color:#00ff00;font-size:1.3rem;font-weight:700;">${trade.take_profit_1:,.2f}</div>'
                            f'<div style="color:#888;font-size:0.75rem;">+{abs((trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100) if trade.entry_price != 0 else 0:.1f}%</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        if trade.take_profit_2:
                            st.markdown(
                                f'<div style="background:#0d1117;padding:12px;border-radius:6px;margin:5px 0;">'
                                f'<div style="color:#aaa;font-size:0.8rem;">TARGET 2 (Scale 40%)</div>'
                                f'<div style="color:#00ff00;font-size:1.3rem;font-weight:700;">${trade.take_profit_2:,.2f}</div>'
                                f'<div style="color:#888;font-size:0.75rem;">+{abs((trade.take_profit_2 - trade.entry_price) / trade.entry_price * 100) if trade.entry_price != 0 else 0:.1f}%</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        if trade.take_profit_3:
                            st.markdown(
                                f'<div style="background:#0d1117;padding:12px;border-radius:6px;margin:5px 0;">'
                                f'<div style="color:#aaa;font-size:0.8rem;">TARGET 3 (Let run 30%)</div>'
                                f'<div style="color:#00ff00;font-size:1.3rem;font-weight:700;">${trade.take_profit_3:,.2f}</div>'
                                f'<div style="color:#888;font-size:0.75rem;">+{abs((trade.take_profit_3 - trade.entry_price) / trade.entry_price * 100):.1f}%</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    with spec_col3:
                        st.markdown("**Position Sizing**")
                        st.metric(
                            "Suggested Size",
                            f"{trade.suggested_position_size_pct}%",
                            help=f"Percentage of portfolio (based on {trade.confidence_score:.0f}% confidence)",
                        )
                        st.metric(
                            "Max Risk",
                            f"{trade.max_risk_per_trade_pct}%",
                            help="Maximum % of portfolio to risk on this trade",
                        )

                        st.markdown("**Timing**")
                        st.info(f"**Hold Time:** {trade.expected_hold_time}")

                    st.markdown("---")

                    # Component Scores
                    st.markdown("### Score Breakdown")
                    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                    with score_col1:
                        st.metric("Technical", f"{trade.technical_score:.0f}/100")
                    with score_col2:
                        st.metric("Momentum", f"{trade.momentum_score:.0f}/100")
                    with score_col3:
                        st.metric("Volatility", f"{trade.volatility_score:.0f}/100")
                    with score_col4:
                        st.metric("Volume", f"{trade.volume_score:.0f}/100")

                    # Supporting Factors
                    with st.expander("Supporting Factors & Analysis", expanded=False):
                        st.markdown("**Supporting Factors:**")
                        for factor in trade.supporting_factors:
                            st.markdown(f"- {factor}")

                        st.markdown("**Risk Factors:**")
                        for risk in trade.risk_factors:
                            st.markdown(f"- {risk}")

                        st.markdown("**Technical Analysis:**")
                        st.code(trade.technical_analysis)

                        st.markdown("**Key Levels:**")
                        for level_name, level_price in trade.key_levels.items():
                            st.text(f"{level_name}: ${level_price:,.2f}")

                    # Alert Notes
                    st.markdown("### Trade Alerts & Notes")
                    for note in trade.alert_notes:
                        st.warning(note)

                    # Invalidation
                    st.error(
                        f"TRADE INVALIDATED if price breaks ${trade.invalidation_price:,.2f}"
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

        else:
            st.info("Click 'Generate Breaking Trades' to find high-confidence setups")

elif selection == "Watchlist":
    from watchlist_dashboard import show_watchlist_dashboard

    show_watchlist_dashboard()

elif selection == "Market Scanner":
    from market_scanner import show_market_scanner

    show_market_scanner()

elif selection == "Symbol Analysis":
    st.title("Symbol Analysis")
    analysis_type = st.radio(
        "Analysis Mode",
        ["Deep Dive (Single Symbol)", "Quant Terminal (Multi-Asset)"],
        horizontal=True,
    )
    if analysis_type == "Deep Dive (Single Symbol)":
        from custom_dashboard import show_custom_dashboard

        show_custom_dashboard()
    else:
        from quant_terminal import show_quant_terminal

        show_quant_terminal()

elif selection == "Chart Analysis":
    st.title("Chart Image Analysis")
    st.caption(
        "Upload any chart screenshot for AI-powered pattern recognition and trade recommendations."
    )
    from chart_image_analyzer import show_chart_analyzer

    show_chart_analyzer()

elif selection == "Intelligence Center":
    st.title("Intelligence Center")
    # Unified Intelligence Center with News, AI Chatbot, and Counter-Trend Signals
    intel_tabs = st.tabs(
        ["News & Sentiment", "Octavian AI Assistant", "Counter-Trend Signals"]
    )
    with intel_tabs[0]:
        from news_dashboard import show_news_dashboard

        show_news_dashboard()
    with intel_tabs[1]:
        from ai_chatbot import show_octavian_chatbot

        show_octavian_chatbot()
    with intel_tabs[2]:
        st.subheader("Counter-Trend Macro Signal Dashboard")
        st.caption(
            "Identifies weaknesses and contradictions in mainstream economic narratives. "
            "Generates signals that profit from consensus mispricing."
        )

        try:
            from counter_trend_analyzer import get_counter_trend_analyzer

            ct = get_counter_trend_analyzer()

            # ── Narrative Overview Table ───────────────────────────────────────
            st.markdown("### Macro Narrative Divergence Tracker")
            st.caption(
                "Consensus Score = how strongly the market believes the narrative (0–100). "
                "Fundamental Score = how well the data actually supports it. "
                "Divergence = mispricing gap."
            )

            narratives = ct.get_all_narratives()
            if narratives:
                narr_rows = [
                    {
                        "Theme": n.theme,
                        "Consensus Score": round(n.consensus_score, 1),
                        "Fundamental Score": round(n.fundamental_score, 1),
                        "Divergence": round(n.divergence, 1),
                        "Status": (
                            "OVERCROWDED"
                            if n.divergence >= 18
                            else "UNDERHYPED"
                            if n.divergence <= -18
                            else "FAIR"
                        ),
                    }
                    for n in narratives
                ]
                narr_df = pd.DataFrame(narr_rows)
                st.dataframe(
                    narr_df[
                        [
                            "Theme",
                            "Consensus Score",
                            "Fundamental Score",
                            "Divergence",
                            "Status",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("---")

            # ── Active Counter-Trend Signals ──────────────────────────────────
            st.markdown("### Active Counter-Trend Trade Signals")
            ct_signals = ct.generate_counter_signals(
                divergence_threshold=18.0, min_strength=50.0
            )

            if ct_signals:
                st.success(
                    f"Found **{len(ct_signals)}** counter-trend opportunities across macro narratives."
                )

                for sig in ct_signals[:10]:  # Show top 10
                    dir_color = "#00ff88" if sig.direction == "LONG" else "#ff4444"
                    strength_color = (
                        "#00ff88"
                        if sig.signal_strength >= 75
                        else "#ffff00"
                        if sig.signal_strength >= 60
                        else "#ffa500"
                    )

                    with st.expander(
                        f"{sig.direction} {sig.instrument}  |  {sig.theme}  |  "
                        f"Strength: {sig.signal_strength:.0f}/100  |  {sig.time_horizon}",
                        expanded=False,
                    ):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Direction", sig.direction)
                        c2.metric("Signal Strength", f"{sig.signal_strength:.0f}/100")
                        c3.metric("Confidence", f"{sig.confidence:.0f}/100")
                        c4.metric("Position Size", f"{sig.position_size_pct:.1f}%")

                        st.markdown(
                            f"<div style='background:#1a1f2e;border-left:4px solid {dir_color};"
                            f"border-radius:6px;padding:14px;margin:8px 0;'>"
                            f"<div style='color:#aaa;font-size:0.8rem;margin-bottom:4px;'>MAINSTREAM NARRATIVE</div>"
                            f"<div style='color:#ccc;'>{sig.macro_narrative}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='background:#1a1f2e;border-left:4px solid {strength_color};"
                            f"border-radius:6px;padding:14px;margin:8px 0;'>"
                            f"<div style='color:#aaa;font-size:0.8rem;margin-bottom:4px;'>COUNTER-THESIS (WHY CROWD IS WRONG)</div>"
                            f"<div style='color:white;font-weight:500;'>{sig.counter_thesis}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(f"**Entry Rationale:** {sig.entry_rationale}")
                        st.info(f"**Catalyst Needed:** {sig.catalyst_needed}")
                        st.warning(f"**Key Risks:** {' | '.join(sig.key_risks)}")

            else:
                st.info(
                    "No strong counter-trend signals at current divergence threshold. Markets may be fairly priced."
                )

            st.markdown("---")

            # ── Narrative Contradiction Detail ────────────────────────────────
            st.markdown("### Narrative Contradictions Deep-Dive")
            all_narrative_objs = ct.get_all_narratives()
            selected_theme = st.selectbox(
                "Select Narrative to Examine",
                [n.theme for n in all_narrative_objs],
            )
            if selected_theme:
                narr_obj = next(
                    (n for n in all_narrative_objs if n.theme == selected_theme),
                    None,
                )
                if narr_obj:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(
                            f"**Consensus Score:** {narr_obj.consensus_score:.0f}/100"
                        )
                        st.markdown(
                            f"**Fundamental Score:** {narr_obj.fundamental_score:.0f}/100"
                        )
                        st.markdown(f"**Divergence:** {narr_obj.divergence:+.1f} pts")
                        if narr_obj.narrative_summary:
                            st.markdown("**Narrative Summary:**")
                            st.write(narr_obj.narrative_summary)
                    with col_b:
                        if narr_obj.contradictions:
                            st.markdown("**Key Contradictions:**")
                            for c in narr_obj.contradictions:
                                st.markdown(f"- {c}")
                        if narr_obj.supporting_data:
                            st.markdown(
                                "**Supporting the Narrative (Devil's Advocate):**"
                            )
                            for s in narr_obj.supporting_data:
                                st.markdown(f"- {s}")

            # ── Full Narrative Report ─────────────────────────────────────────
            with st.expander("Full Narrative Report (Text)", expanded=False):
                st.code(ct.get_narrative_report(), language=None)

        except Exception as e:
            st.error(f"Counter-trend analyzer unavailable: {e}")
            st.info("Ensure counter_trend_analyzer.py is present in the project root.")

elif selection == "Paper Trading":
    from paper_trading_ui import show_paper_trading

    show_paper_trading()

elif selection == "Simulation Hub":
    from simulation_viewer import render_simulation_viewer

    render_simulation_viewer()

elif selection == "Trader Profile":
    show_profile_settings()

elif selection == "Financial Model Generator":
    import plotly.express as px

    # Defensive import — clears stale .pyc cache on first ImportError then retries
    try:
        from financial_model_generator import (
            DCFAssumptions,
            get_dcf_engine,
            get_financial_generator,
        )

        _fmg_import_ok = True
    except ImportError:
        # Stale __pycache__ can cause spurious ImportErrors; invalidate and retry
        import importlib
        import sys

        for _mod in list(sys.modules.keys()):
            if "financial_model_generator" in _mod:
                del sys.modules[_mod]
        try:
            from financial_model_generator import (
                DCFAssumptions,
                get_dcf_engine,
                get_financial_generator,
            )

            _fmg_import_ok = True
        except Exception as _fmg_err:
            _fmg_import_ok = False
            st.error(
                f"Could not load Financial Model Generator: `{_fmg_err}`\n\n"
                "**Fix:** Stop Streamlit, run `find . -name '*.pyc' -delete` "
                "in the project root, then restart."
            )
    except Exception as _fmg_err:
        _fmg_import_ok = False
        st.error(f"Financial Model Generator failed to load: `{_fmg_err}`")

    if not _fmg_import_ok:
        st.stop()

    st.title("Institutional DCF Valuation Engine")
    st.caption(
        "Wall Street-grade model: Revenue → EBIT → NOPAT → FCF → DCF | "
        "Monte Carlo | Scenario Analysis | Relative Valuation | Trade Signals"
    )

    # ── Ticker input + auto-fill ──────────────────────────────────────────────
    fmg_col_ticker, fmg_col_fetch = st.columns([2, 1])
    with fmg_col_ticker:
        ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("fmg_ticker", ""),
            placeholder="e.g. MSFT, NVDA, TSLA…",
            help="Enter any ticker. Use 'Auto-Fill from Ticker' to populate assumptions from live data.",
        )
    with fmg_col_fetch:
        st.markdown("<br>", unsafe_allow_html=True)
        do_autofill = st.button(
            "Auto-Fill from Ticker",
            use_container_width=True,
            help="Fetches live financials from yfinance and populates all fields automatically.",
        )

    if do_autofill and ticker:
        with st.spinner(f"Fetching financials for {ticker.upper()}…"):
            try:
                import yfinance as _yf_fmg

                _t = _yf_fmg.Ticker(ticker.upper())
                _info = _t.info or {}

                # Price
                _price = float(
                    _info.get("currentPrice")
                    or _info.get("regularMarketPrice")
                    or _info.get("previousClose")
                    or 100.0
                )
                # Revenue (trailing 12 months, in $M)
                _rev_raw = _info.get("totalRevenue") or _info.get("revenuePerShare", 0)
                _rev = (
                    float(_rev_raw) / 1e6
                    if _rev_raw and _rev_raw > 1e6
                    else max(float(_rev_raw or 100), 100.0)
                )
                # EBIT margin
                _ebit_m = float(_info.get("operatingMargins") or 0.20)
                # Beta
                _beta = float(_info.get("beta") or 1.0)
                # Market cap ($M)
                _mktcap = float(_info.get("marketCap") or 0) / 1e6
                if _mktcap <= 0:
                    _mktcap = (
                        _price * float(_info.get("sharesOutstanding") or 1e9) / 1e6
                    )
                # Total debt ($M)
                _debt = float(_info.get("totalDebt") or 0) / 1e6
                # Cash ($M)
                _cash_val = float(_info.get("totalCash") or 0) / 1e6
                # Shares outstanding (M)
                _shares = float(_info.get("sharesOutstanding") or 1e9) / 1e6
                # Revenue growth (analyst estimate or trailing)
                _rev_growth = float(
                    _info.get("revenueGrowth") or _info.get("earningsGrowth") or 0.08
                )
                _rev_growth = max(min(_rev_growth, 0.60), -0.10)
                # Tax rate
                _tax = float(_info.get("effectiveTaxRate") or 0.21)
                _tax = max(min(_tax, 0.40), 0.05)

                st.session_state["fmg_ticker"] = ticker.upper()
                st.session_state["fmg_price"] = round(_price, 2)
                st.session_state["fmg_rev"] = round(_rev, 1)
                st.session_state["fmg_rev_growth"] = round(_rev_growth * 100, 1)
                st.session_state["fmg_ebit"] = round(_ebit_m * 100, 1)
                st.session_state["fmg_tax"] = round(_tax * 100, 1)
                st.session_state["fmg_beta"] = round(_beta, 2)
                st.session_state["fmg_mktcap"] = round(_mktcap, 0)
                st.session_state["fmg_debt"] = round(_debt, 0)
                st.session_state["fmg_cash"] = round(_cash_val, 0)
                st.session_state["fmg_shares"] = round(_shares, 1)
                st.session_state.pop("dcf_result", None)
                st.success(
                    f"Auto-filled assumptions for {ticker.upper()} — review and click Run DCF."
                )
                st.rerun()
            except Exception as _af_err:
                st.warning(
                    f"Auto-fill partial ({_af_err}). Enter assumptions manually."
                )

    # Two-column layout: inputs left, results right
    col_input, col_main = st.columns([1, 2])

    with col_input:
        st.subheader("Revenue & Margins")
        current_price = st.number_input(
            "Current Market Price ($)",
            value=float(st.session_state.get("fmg_price", 100.0)),
            min_value=0.01,
        )
        base_revenue = st.number_input(
            "Base Revenue ($M)",
            value=float(st.session_state.get("fmg_rev", 5_000.0)),
            step=100.0,
        )
        rev_growth = (
            st.slider(
                "Revenue Growth Rate (%)",
                0.0,
                50.0,
                float(st.session_state.get("fmg_rev_growth", 8.0)),
            )
            / 100
        )
        ebit_margin = (
            st.slider(
                "EBIT Margin (%)",
                0.0,
                60.0,
                float(st.session_state.get("fmg_ebit", 20.0)),
            )
            / 100
        )
        tax_rate = (
            st.slider(
                "Tax Rate (%)",
                0.0,
                40.0,
                float(st.session_state.get("fmg_tax", 21.0)),
            )
            / 100
        )

        st.subheader("Working Capital & CapEx")
        da_pct = st.slider("D&A (% Revenue)", 0.0, 15.0, 3.0) / 100
        capex_pct = st.slider("CapEx (% Revenue)", 0.0, 20.0, 5.0) / 100
        nwc_pct = st.slider("ΔNWC (% Revenue)", -5.0, 10.0, 1.0) / 100

        st.subheader("WACC / CAPM")
        risk_free = st.slider("Risk-Free Rate (%)", 2.0, 8.0, 4.25) / 100
        erp = st.slider("Equity Risk Premium (%)", 3.0, 8.0, 5.5) / 100
        beta = st.slider(
            "Beta",
            0.3,
            3.0,
            float(st.session_state.get("fmg_beta", 1.0)),
            step=0.05,
        )
        cost_of_debt = st.slider("Cost of Debt (%)", 2.0, 10.0, 4.5) / 100
        equity_val_market = st.number_input(
            "Market Cap ($M)",
            value=float(st.session_state.get("fmg_mktcap", 10_000.0)),
            step=100.0,
        )
        debt_total = st.number_input(
            "Total Debt ($M)",
            value=float(st.session_state.get("fmg_debt", 1_000.0)),
            step=100.0,
        )
        cash = st.number_input(
            "Cash ($M)",
            value=float(st.session_state.get("fmg_cash", 500.0)),
            step=100.0,
        )

        st.subheader("Terminal Value & Shares")
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0.5, 5.0, 2.5) / 100
        shares = st.number_input(
            "Shares Outstanding (M)",
            value=float(st.session_state.get("fmg_shares", 1_000.0)),
            step=10.0,
        )
        proj_years = st.selectbox("Projection Years", [3, 5, 7, 10], index=1)

        st.subheader("Peer Multiples (Relative Valuation)")
        peer_pe = st.number_input("Peer P/E", value=25.0, step=1.0)
        peer_ev_ebitda = st.number_input("Peer EV/EBITDA", value=15.0, step=0.5)
        peer_ev_fcf = st.number_input("Peer EV/FCF", value=20.0, step=0.5)
        peg = st.number_input("PEG Ratio", value=2.0, step=0.1)

        run_model = st.button(
            "Run Institutional DCF", type="primary", use_container_width=True
        )

    # ── Run engine ────────────────────────────────────────────────────────────
    if run_model:
        assumptions = DCFAssumptions(
            ticker=ticker,
            base_revenue=base_revenue,
            revenue_growth_rates=[rev_growth] * proj_years,
            ebit_margin=ebit_margin,
            tax_rate=tax_rate,
            da_pct_revenue=da_pct,
            capex_pct_revenue=capex_pct,
            nwc_change_pct_revenue=nwc_pct,
            equity_value_market=equity_val_market,
            debt_value=debt_total,
            cost_of_debt=cost_of_debt,
            risk_free_rate=risk_free,
            equity_risk_premium=erp,
            beta=beta,
            terminal_growth_rate=terminal_growth,
            cash=cash,
            shares_outstanding=shares,
            current_price=current_price,
            peer_pe=peer_pe,
            peer_ev_ebitda=peer_ev_ebitda,
            peer_ev_fcf=peer_ev_fcf,
            peg_ratio=peg,
            projection_years=proj_years,
        )
        engine = get_dcf_engine()
        with st.spinner(
            "Running institutional DCF (including Monte Carlo 10,000 paths)..."
        ):
            result = engine.run_dcf(assumptions)
        st.session_state["dcf_result"] = result

    # ── Display results ───────────────────────────────────────────────────────
    with col_main:
        if "dcf_result" not in st.session_state:
            st.info(
                "Configure assumptions on the left and click **Run Institutional DCF**."
            )
        else:
            result = st.session_state["dcf_result"]
            sig = result.trade_signal

            # ── Trade Signal Banner ───────────────────────────────────────────
            sig_colors = {
                "Strong Long": "#00ff88",
                "Long": "#7fff7f",
                "Neutral": "#aaaaaa",
                "Short": "#ff9944",
                "Strong Short": "#ff4444",
            }
            sig_color = sig_colors.get(sig.signal, "#ffffff")
            st.markdown(
                f"""
                <div style="background:#161b22;border:1px solid {sig_color};border-radius:8px;
                            padding:12px 16px;margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                    <div>
                      <div style="font-size:1.1rem;font-weight:700;color:{sig_color};">{sig.signal.upper()}</div>
                      <div style="color:#aaa;font-size:0.75rem;">{result.ticker} — {sig.risk_level} Risk</div>
                    </div>
                    <div style="text-align:center;">
                      <div style="font-size:1.3rem;font-weight:700;color:white;">${sig.fair_value:.2f}</div>
                      <div style="color:#aaa;font-size:0.7rem;">Fair Value / Share</div>
                    </div>
                    <div style="text-align:center;">
                      <div style="font-size:1.1rem;font-weight:700;color:{sig_color};">{sig.upside_pct:+.1f}%</div>
                      <div style="color:#aaa;font-size:0.7rem;">vs ${sig.market_price:.2f} Mkt</div>
                    </div>
                    <div style="text-align:center;">
                      <div style="font-size:1.1rem;font-weight:700;color:white;">{sig.confidence_pct:.0f}%</div>
                      <div style="color:#aaa;font-size:0.7rem;">Confidence</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── KPI Row ───────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Enterprise Value", f"${result.enterprise_value / 1000:.1f}B")
            k2.metric("Equity Value", f"${result.equity_value / 1000:.1f}B")
            k3.metric("WACC", f"{result.wacc:.2%}")
            k4.metric("Cost of Equity", f"{result.cost_of_equity:.2%}")
            k5.metric("Position Size", f"{sig.position_size_pct:.1f}%")

            # ── Tabs ──────────────────────────────────────────────────────────
            dcf_tabs = st.tabs(
                [
                    "20-Line DCF",
                    "Scenarios",
                    "Monte Carlo",
                    "Relative Valuation",
                    "Sensitivity",
                    "Catalysts",
                    "WACC Detail",
                ]
            )

            # Tab 1: 20-Line DCF
            with dcf_tabs[0]:
                st.subheader("Investment Banking 20-Line DCF Template")
                st.caption(
                    "All values in $M. Discount Factor = 1/(1+WACC)^t — decreases each year (correct PV factor)."
                )
                fmt_cols = {
                    c: "{:,.1f}"
                    for c in result.line_items.columns
                    if c != "Discount Factor"
                }
                fmt_cols["Discount Factor"] = "{:.4f}"
                try:
                    st.dataframe(
                        result.line_items.style.format(fmt_cols),
                        use_container_width=True,
                    )
                except Exception:
                    st.dataframe(result.line_items, use_container_width=True)

                v1, v2, v3 = st.columns(3)
                v1.metric("Sum PV(FCF)", f"${result.sum_pv_fcf:,.0f}M")
                v2.metric("PV(Terminal Value)", f"${result.pv_terminal_value:,.0f}M")
                v3.metric(
                    "TV as % of EV",
                    f"{result.pv_terminal_value / result.enterprise_value * 100:.0f}%",
                )

                # Waterfall chart
                wf_labels = [
                    "PV FCFs",
                    "PV Terminal Value",
                    "Enterprise Value",
                    "- Net Debt",
                    "Equity Value",
                ]
                wf_values = [
                    result.sum_pv_fcf / 1000,
                    result.pv_terminal_value / 1000,
                    0,
                    -result.net_debt / 1000,
                    0,
                ]
                wf_measure = ["relative", "relative", "total", "relative", "total"]
                fig_wf = go.Figure(
                    go.Waterfall(
                        name="Valuation Bridge",
                        orientation="v",
                        measure=wf_measure,
                        x=wf_labels,
                        y=wf_values,
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={
                            "marker": {"color": COLORS.get("danger", "#ff4444")}
                        },
                        increasing={"marker": {"color": COLORS.get("gold", "#e0c97f")}},
                        totals={"marker": {"color": COLORS.get("lavender", "#b39ddb")}},
                    )
                )
                fig_wf.update_layout(
                    title="Valuation Bridge ($B)",
                    template="plotly_dark",
                    height=350,
                    yaxis_title="Value ($B)",
                )
                st.plotly_chart(fig_wf, use_container_width=True)

            # Tab 2: Scenarios
            with dcf_tabs[1]:
                st.subheader("Scenario-Weighted Valuation")
                scen_cols = st.columns(3)
                colors_scen = {"Bear": "#ff4444", "Base": "#e0c97f", "Bull": "#00ff88"}
                for i, s in enumerate(result.scenarios):
                    c = colors_scen.get(s.label, "#ffffff")
                    with scen_cols[i]:
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid {c};"
                            f"border-radius:6px;padding:10px;text-align:center;'>"
                            f"<div style='color:{c};font-size:0.9rem;font-weight:700;'>{s.label}</div>"
                            f"<div style='color:#aaa;font-size:0.72rem;'>P = {s.probability:.0%}</div>"
                            f"<div style='color:white;font-size:1.3rem;font-weight:800;'>${s.fair_value:.2f}</div>"
                            f"<div style='color:{c};font-size:0.82rem;'>{s.upside:+.1%} upside</div>"
                            f"<hr style='border-color:#333;margin:4px 0;'/>"
                            f"<div style='color:#aaa;font-size:0.7rem;'>Growth: {s.revenue_growth_avg:.1%} | EBIT: {s.ebit_margin:.1%}</div>"
                            f"<div style='color:#aaa;font-size:0.7rem;'>WACC: {s.wacc:.2%}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    f"<div style='background:#1a1f2e;border:1px solid #e0c97f;"
                    f"border-radius:6px;padding:10px;text-align:center;margin-top:10px;'>"
                    f"<div style='color:#aaa;font-size:0.8rem;'>Probability-Weighted Fair Value</div>"
                    f"<div style='color:#e0c97f;font-size:1.6rem;font-weight:800;'>"
                    f"${result.scenario_weighted_value:.2f}</div>"
                    f"<div style='color:#aaa;font-size:0.72rem;'>"
                    f"Bear×25% + Base×50% + Bull×25%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Tab 3: Monte Carlo
            with dcf_tabs[2]:
                st.subheader("Monte Carlo Valuation (10,000 Simulations)")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Median Fair Value", f"${result.mc_median:.2f}")
                mc2.metric("Mean Fair Value", f"${result.mc_mean:.2f}")
                mc3.metric("Upside Probability", f"{result.mc_upside_prob:.1%}")
                mc4.metric("Downside Probability", f"{result.mc_downside_prob:.1%}")

                p = result.mc_percentiles
                pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                pc1.metric("P5", f"${p['p5']:.2f}")
                pc2.metric("P25", f"${p['p25']:.2f}")
                pc3.metric("P50", f"${p['p50']:.2f}")
                pc4.metric("P75", f"${p['p75']:.2f}")
                pc5.metric("P95", f"${p['p95']:.2f}")

                # Distribution histogram
                if result.mc_distribution:
                    fig_mc = px.histogram(
                        x=result.mc_distribution,
                        nbins=80,
                        title="Distribution of Fair Values (Monte Carlo)",
                        labels={"x": "Fair Value Per Share ($)"},
                        color_discrete_sequence=["#b39ddb"],
                    )
                    fig_mc.add_vline(
                        x=current_price,
                        line_dash="dash",
                        line_color="#ff4444",
                        annotation_text=f"Market ${current_price:.2f}",
                    )
                    fig_mc.add_vline(
                        x=result.mc_median,
                        line_dash="dash",
                        line_color="#00ff88",
                        annotation_text=f"Median ${result.mc_median:.2f}",
                    )
                    fig_mc.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_mc, use_container_width=True)

            # Tab 4: Relative Valuation
            with dcf_tabs[3]:
                st.subheader("Peer Comparison — Relative Valuation")
                rv = result.relative_valuation
                rv_methods = [
                    "P/E Implied",
                    "EV/EBITDA Implied",
                    "EV/FCF Implied",
                    "PEG Implied",
                    "DCF (This Model)",
                ]
                rv_values = [
                    rv["pe_implied_fv"],
                    rv["ev_ebitda_implied_fv"],
                    rv["ev_fcf_implied_fv"],
                    rv["peg_implied_fv"],
                    result.fair_value_per_share,
                ]
                rv_multiples = [
                    f"{rv['peer_pe']:.1f}x",
                    f"{rv['peer_ev_ebitda']:.1f}x",
                    f"{rv['peer_ev_fcf']:.1f}x",
                    f"{rv['peg_ratio']:.2f}",
                    "—",
                ]
                rv_df = pd.DataFrame(
                    {
                        "Valuation Method": rv_methods,
                        "Fair Value ($)": [f"${v:.2f}" for v in rv_values],
                        "Peer Multiple": rv_multiples,
                        "vs Market": [
                            f"{(v - current_price) / current_price:+.1%}"
                            if current_price > 0
                            else "N/A"
                            for v in rv_values
                        ],
                    }
                )
                st.dataframe(rv_df, use_container_width=True, hide_index=True)

                fig_rv = go.Figure()
                bar_colors = [
                    "#00ff88" if v > current_price else "#ff4444" for v in rv_values
                ]
                fig_rv.add_trace(
                    go.Bar(
                        x=rv_methods,
                        y=rv_values,
                        marker_color=bar_colors,
                        text=[f"${v:.2f}" for v in rv_values],
                        textposition="outside",
                    )
                )
                fig_rv.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color="#e0c97f",
                    annotation_text=f"Market ${current_price:.2f}",
                )
                fig_rv.update_layout(
                    title="Implied Fair Values by Method",
                    template="plotly_dark",
                    height=400,
                    yaxis_title="Fair Value ($)",
                )
                st.plotly_chart(fig_rv, use_container_width=True)

                # Key metrics
                r1, r2, r3 = st.columns(3)
                r1.metric("EBITDA ($M)", f"${rv['ebitda_M']:,.0f}M")
                r2.metric("FCF ($M)", f"${rv['last_fcf_M']:,.0f}M")
                r3.metric("EPS Proxy", f"${rv['eps_proxy']:.2f}")

            # Tab 5: Sensitivity
            with dcf_tabs[4]:
                st.subheader("Sensitivity Analysis — Fair Value Per Share")
                st.caption("Rows: Terminal Growth Rate | Columns: WACC")
                try:
                    st.dataframe(
                        result.sensitivity.style.format("${:.2f}"),
                        use_container_width=True,
                    )
                except Exception:
                    st.dataframe(result.sensitivity, use_container_width=True)

            # Tab 6: Catalysts
            with dcf_tabs[5]:
                st.subheader("Catalyst Tracking Dashboard")
                impact_colors = {
                    "High": "#ff4444",
                    "Medium": "#e0c97f",
                    "Low": "#aaaaaa",
                }
                dir_colors = {
                    "Positive": "#00ff88",
                    "Negative": "#ff4444",
                    "Neutral": "#aaaaaa",
                }
                for cat in result.catalysts:
                    ic = impact_colors.get(cat.impact, "#aaaaaa")
                    dc = dir_colors.get(cat.direction, "#aaaaaa")
                    st.markdown(
                        f"<div style='background:#161b22;border-left:4px solid {ic};"
                        f"border-radius:6px;padding:14px;margin:8px 0;'>"
                        f"<div style='display:flex;justify-content:space-between;'>"
                        f"<span style='font-weight:700;color:white;font-size:1.05rem;'>{cat.name}</span>"
                        f"<span style='color:{ic};font-weight:600;'>{cat.impact} Impact</span>"
                        f"</div>"
                        f"<div style='color:#888;font-size:0.8rem;margin:4px 0;'>Date: {cat.date} | "
                        f"<span style='color:{dc};'>{cat.direction}</span></div>"
                        f"<div style='color:#ccc;font-size:0.9rem;'>{cat.description}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Tab 7: WACC Detail
            with dcf_tabs[6]:
                st.subheader("WACC Decomposition")
                wb = result.wacc_breakdown
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("WACC", f"{wb['wacc']:.2%}")
                wc2.metric("Cost of Equity (CAPM)", f"{wb['cost_of_equity']:.2%}")
                wc3.metric(
                    "After-Tax Cost of Debt", f"{wb['cost_of_debt_aftertax']:.2%}"
                )
                wc4, wc5, wc6 = st.columns(3)
                wc4.metric("Weight Equity", f"{wb['weight_equity']:.1%}")
                wc5.metric("Weight Debt", f"{wb['weight_debt']:.1%}")
                wc6.metric("Beta", f"{wb['beta']:.2f}")

                st.markdown("""
                **WACC Formula:**
                `WACC = (E/(D+E)) × Re + (D/(D+E)) × Rd × (1 − T)`

                **Cost of Equity (CAPM):**
                `Re = Rf + β × (Rm − Rf)`
                """)

                wacc_data = {
                    "Component": [
                        "Risk-Free Rate (Rf)",
                        "Equity Risk Premium (ERP)",
                        "Beta (β)",
                        "Cost of Equity (Re)",
                        "Pre-Tax Cost of Debt (Rd)",
                        "After-Tax Cost of Debt",
                        "Weight Equity",
                        "Weight Debt",
                        "WACC",
                    ],
                    "Value": [
                        f"{wb['risk_free_rate']:.2%}",
                        f"{wb['equity_risk_premium']:.2%}",
                        f"{wb['beta']:.2f}",
                        f"{wb['cost_of_equity']:.2%}",
                        f"{wb['cost_of_debt_pretax']:.2%}",
                        f"{wb['cost_of_debt_aftertax']:.2%}",
                        f"{wb['weight_equity']:.1%}",
                        f"{wb['weight_debt']:.1%}",
                        f"{wb['wacc']:.2%}",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(wacc_data), use_container_width=True, hide_index=True
                )

            # ── Download ──────────────────────────────────────────────────────
            st.markdown("---")
            gen = get_financial_generator()
            # Build a minimal model_data dict wrapping the full result
            model_data_for_export = {"full_result": result}
            xls_bytes = gen.generate_excel(model_data_for_export)
            st.download_button(
                "Download Full Institutional Model (Excel)",
                data=xls_bytes,
                file_name=f"{result.ticker}_Institutional_DCF.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

elif selection == "Daily Briefing":
    import plotly.express as px

    st.title("Daily Market Intelligence Briefing")
    st.caption(
        "Institutional-grade daily analysis: macro regimes, cross-asset signals, narrative dislocations, and trade ideas"
    )

    col_gen1, col_gen2 = st.columns([1, 3])
    with col_gen1:
        if st.button(
            "Generate Full Briefing", type="primary", use_container_width=True
        ):
            from daily_intelligence import get_daily_engine

            engine = get_daily_engine()
            with st.spinner(
                "Fetching global market data, yield curve, sectors, vol surface, FX, commodities, crypto…"
            ):
                report = engine.generate_briefing()
                st.session_state["daily_report"] = report
            st.rerun()
    with col_gen2:
        if "daily_report" in st.session_state:
            rpt = st.session_state["daily_report"]
            es = rpt.get("executive_summary", {})
            tone = es.get("market_tone", "UNKNOWN")
            tone_map = {
                "RISK-ON | BULLISH": "#00ff88",
                "CAUTIOUSLY BULLISH": "#7fff7f",
                "NEUTRAL | MIXED SIGNALS": "#aaaaaa",
                "CAUTIOUSLY BEARISH": "#ffa500",
                "RISK-OFF | BEARISH": "#ff4444",
            }
            tone_color = tone_map.get(tone, "#cccccc")
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {tone_color};"
                f"border-radius:6px;padding:10px 16px;display:inline-block;margin-top:4px;'>"
                f"<span style='font-size:0.75rem;color:#aaa;'>MARKET TONE  </span>"
                f"<span style='font-size:1rem;font-weight:700;color:{tone_color};'>{tone}</span>"
                f"  <span style='font-size:0.75rem;color:#888;'>— {es.get('date', '')} {es.get('timestamp', '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if "daily_report" not in st.session_state:
        st.info(
            "Click **Generate Full Briefing** to run the complete market intelligence sweep."
        )
    else:
        rpt = st.session_state["daily_report"]
        es = rpt.get("executive_summary", {})
        indices = rpt.get("indices", {})
        yield_data = rpt.get("yield_curve", {})
        vix_data = rpt.get("volatility", {})
        fx = rpt.get("fx", {})
        commodities = rpt.get("commodities", {})
        crypto = rpt.get("crypto", {})
        sectors = rpt.get("sectors", {})
        macro = rpt.get("macro", {})
        movers = rpt.get("movers", {})
        cross_signals = rpt.get("cross_asset_signals", [])
        dislocations = rpt.get("narrative_dislocations", [])
        trade_ideas = rpt.get("trade_ideas", [])
        posture = rpt.get("posture", {})

        db_tabs = st.tabs(
            [
                "Executive Summary",
                "Global Indices",
                "Macro & Rates",
                "Volatility Surface",
                "Sector Rotation",
                "FX & Commodities",
                "Cross-Asset Signals",
                "Narrative Dislocations",
                "Trade Ideas & Posture",
            ]
        )

        # ── Tab 1: Executive Summary ─────────────────────────────────────────
        with db_tabs[0]:
            regime_label = es.get("regime", "Unknown")
            regime_conf = es.get("regime_confidence", 0)
            vix_val = es.get("vix", 0)
            spread = es.get("spread_2s10s", 0)
            sp500_1d = es.get("sp500_1d", 0)
            sp500_1m = es.get("sp500_1m", 0)

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("S&P 500 (1D)", f"{sp500_1d:+.2f}%")
            r2.metric("S&P 500 (1M)", f"{sp500_1m:+.2f}%")
            r3.metric("VIX", f"{vix_val:.1f}")
            r4.metric("2s10s Spread", f"{spread:.0f}bps")
            r5.metric(
                "HMM Regime Conf.",
                f"{regime_conf:.0%}"
                if isinstance(regime_conf, float)
                else str(regime_conf),
            )

            st.markdown("---")
            col_es1, col_es2 = st.columns(2)

            with col_es1:
                st.markdown("#### Market Regime")
                regime_bg = (
                    "#1a2e1a"
                    if "Bull" in regime_label
                    else "#2e1a1a"
                    if "Bear" in regime_label or "Crash" in regime_label
                    else "#1a1f2e"
                )
                regime_border = (
                    "#00ff88"
                    if "Bull" in regime_label
                    else "#ff4444"
                    if "Bear" in regime_label or "Crash" in regime_label
                    else "#aaaaaa"
                )
                st.markdown(
                    f"<div style='background:{regime_bg};border-left:4px solid {regime_border};"
                    f"border-radius:6px;padding:12px;margin:4px 0;'>"
                    f"<div style='color:{regime_border};font-size:1rem;font-weight:700;'>{regime_label}</div>"
                    f"<div style='color:#ccc;font-size:0.8rem;margin-top:4px;'>{es.get('regime_desc', '')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("#### Yield Curve")
                curve_color = (
                    "#ff4444"
                    if "Inverted" in es.get("yield_curve", "")
                    else "#00ff88"
                    if "Steep" in es.get("yield_curve", "")
                    else "#e0c97f"
                )
                st.markdown(
                    f"<div style='background:#161b22;border-left:4px solid {curve_color};"
                    f"border-radius:6px;padding:10px;margin:4px 0;'>"
                    f"<div style='color:{curve_color};font-size:0.85rem;font-weight:600;'>{es.get('yield_curve', '')}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem;'>2s10s: {spread:.0f}bps</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("#### VIX Regime")
                vix_regime_txt = vix_data.get("_meta", {}).get("vix_regime", "")
                vix_color_map = {
                    "green": "#00ff88",
                    "yellow": "#e0c97f",
                    "orange": "#ffa500",
                    "red": "#ff4444",
                    "darkred": "#cc0000",
                }
                vix_c = vix_color_map.get(
                    vix_data.get("_meta", {}).get("vix_color", "yellow"), "#e0c97f"
                )
                st.markdown(
                    f"<div style='background:#161b22;border-left:4px solid {vix_c};"
                    f"border-radius:6px;padding:10px;margin:4px 0;'>"
                    f"<div style='color:{vix_c};font-size:0.85rem;font-weight:600;'>VIX {vix_val:.1f} — {vix_regime_txt}</div>"
                    f"<div style='color:#aaa;font-size:0.75rem;'>Term structure: {vix_data.get('_meta', {}).get('ts_regime', '')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_es2:
                st.markdown("#### Key Risks")
                for risk in es.get("key_risks", [])[:6]:
                    st.markdown(
                        f"<div style='background:#1f1015;border-left:3px solid #ff4444;"
                        f"border-radius:4px;padding:7px 10px;margin:3px 0;font-size:0.8rem;color:#ffaaaa;'>"
                        f"[!] {risk}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("#### Key Opportunities")
                for opp in es.get("key_opportunities", [])[:5]:
                    if opp:
                        st.markdown(
                            f"<div style='background:#101f15;border-left:3px solid #00ff88;"
                            f"border-radius:4px;padding:7px 10px;margin:3px 0;font-size:0.8rem;color:#aaffcc;'>"
                            f"+ {opp[:120]}</div>",
                            unsafe_allow_html=True,
                        )

            # Portfolio posture
            st.markdown("---")
            st.markdown("#### Recommended Portfolio Posture")
            p_label = posture.get("label", "")
            p_col = (
                "#00ff88"
                if "Long" in p_label or "Risk-On" in p_label
                else "#ff4444"
                if "Defensive" in p_label or "Crisis" in p_label
                else "#e0c97f"
            )
            pcols = st.columns(5)
            pcols[0].metric("Posture", p_label)
            pcols[1].metric("Equities", posture.get("equity_pct", "—"))
            pcols[2].metric("Bonds", posture.get("bond_pct", "—"))
            pcols[3].metric("Alternatives", posture.get("alternatives", "—"))
            pcols[4].metric("Cash", posture.get("cash", "—"))
            if posture.get("notes"):
                st.info(posture["notes"])

        # ── Tab 2: Global Indices ────────────────────────────────────────────
        with db_tabs[1]:
            if indices:
                rows = []
                for name, d in indices.items():
                    if name == "VIX":
                        continue
                    above = d.get("above_200", False)
                    rows.append(
                        {
                            "Index": name,
                            "Price": f"{d.get('price', 0):,.2f}",
                            "1D %": f"{d.get('chg_1d', 0):+.2f}%",
                            "5D %": f"{d.get('chg_5d', 0):+.2f}%",
                            "1M %": f"{d.get('chg_1m', 0):+.2f}%",
                            "3M %": f"{d.get('chg_3m', 0):+.2f}%",
                            "Vol (20d)": f"{d.get('vol_20d', 0):.1f}%",
                            "RSI": f"{d.get('rsi', 50):.0f}",
                            "MACD": d.get("macd", "—"),
                            "BB Position": d.get("bb_pos", "—"),
                            "Above 200MA": "YES" if above else "NO",
                        }
                    )
                df_idx = pd.DataFrame(rows)

                def color_pct(val):
                    try:
                        v = float(val.replace("%", "").replace("+", ""))
                        return (
                            "color: #00ff88"
                            if v > 0
                            else "color: #ff4444"
                            if v < 0
                            else ""
                        )
                    except Exception:
                        return ""

                st.dataframe(
                    df_idx.style.applymap(
                        color_pct, subset=["1D %", "5D %", "1M %", "3M %"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Normalized chart
                fig_norm = go.Figure()
                chart_pairs = [
                    ("S&P 500", COLORS.get("gold", "#e0c97f")),
                    ("NASDAQ 100", COLORS.get("lavender", "#b39ddb")),
                    ("Russell 2000", "#42a5f5"),
                    ("FTSE 100", "#66bb6a"),
                    ("DAX", "#ef5350"),
                    ("Nikkei 225", "#ab47bc"),
                ]
                import yfinance as _yf_idx

                for idx_name, idx_color in chart_pairs:
                    idx_ticker = {
                        "S&P 500": "^GSPC",
                        "NASDAQ 100": "^NDX",
                        "Russell 2000": "^RUT",
                        "FTSE 100": "^FTSE",
                        "DAX": "^GDAXI",
                        "Nikkei 225": "^N225",
                    }.get(idx_name, "^GSPC")
                    try:
                        _df = _yf_idx.Ticker(idx_ticker).history(period="3mo")
                        if _df is not None and not _df.empty and "Close" in _df.columns:
                            _close = _df["Close"].dropna()
                            if len(_close) > 5:
                                _norm = (_close / float(_close.iloc[0]) - 1) * 100
                                fig_norm.add_trace(
                                    go.Scatter(
                                        x=_norm.index,
                                        y=_norm.values,
                                        mode="lines",
                                        name=idx_name,
                                        line=dict(color=idx_color, width=2),
                                    )
                                )
                    except Exception:
                        pass
                fig_norm.add_hline(y=0, line_dash="dash", line_color="#444")
                fig_norm.update_layout(
                    title=dict(
                        text="Global Indices — 3-Month Normalized Performance (%)",
                        font=dict(size=13),
                    ),
                    height=380,
                    template="plotly_dark",
                    yaxis_title="Return vs 3M ago (%)",
                    margin=dict(l=10, r=10, t=35, b=10),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_norm, use_container_width=True)
            else:
                st.warning("Index data unavailable.")

        # ── Tab 3: Macro & Rates ─────────────────────────────────────────────
        with db_tabs[2]:
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.markdown("#### US Treasury Yield Curve")
                yields = yield_data.get("yields", {})
                if yields:
                    y_rows = []
                    for tenor, d in yields.items():
                        rate = d.get("rate", 0)
                        y_rows.append(
                            {
                                "Tenor": tenor,
                                "Yield": f"{rate * 100:.2f}%"
                                if rate < 1
                                else f"{rate:.2f}%",
                                "1D Chg": f"{d.get('chg_1d', 0):+.2f}%",
                                "1M Chg": f"{d.get('chg_1m', 0):+.2f}%",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(y_rows), use_container_width=True, hide_index=True
                    )

                    # Yield curve spreads
                    spread_cols = st.columns(3)
                    spread_2s10s_val = yield_data.get("spread_2s10s_bps", 0)
                    spread_3m10y_val = yield_data.get("spread_3m10y_bps", 0)
                    spread_2s30s_val = yield_data.get("spread_2s30s_bps", 0)
                    c_2s10s = "#ff4444" if spread_2s10s_val < 0 else "#00ff88"
                    spread_cols[0].metric(
                        "2s10s Spread",
                        f"{spread_2s10s_val:.0f}bps",
                        delta="Inverted" if spread_2s10s_val < 0 else "Normal",
                    )
                    spread_cols[1].metric("3m10Y Spread", f"{spread_3m10y_val:.0f}bps")
                    spread_cols[2].metric("2s30s Spread", f"{spread_2s30s_val:.0f}bps")

                    curve_sig = yield_data.get("curve_signal", "NEUTRAL")
                    curve_color = (
                        "#00ff88"
                        if curve_sig == "BULLISH"
                        else "#ff4444"
                        if curve_sig == "BEARISH"
                        else "#e0c97f"
                    )
                    st.markdown(
                        f"<div style='background:#161b22;border-left:3px solid {curve_color};"
                        f"border-radius:5px;padding:8px 12px;font-size:0.82rem;'>"
                        f"<b style='color:{curve_color};'>Signal: {curve_sig}</b> — {yield_data.get('curve_shape', '')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with col_m2:
                st.markdown("#### Fed Policy Signals")
                fed = macro.get("fed", {})
                for k, v in fed.items():
                    st.markdown(f"**{k}:** {v}")

                st.markdown("#### Market Breadth")
                breadth = macro.get("breadth", {})
                for k, v in breadth.items():
                    if isinstance(v, float):
                        color = "#00ff88" if v > 0 else "#ff4444"
                        st.markdown(
                            f"**{k}:** <span style='color:{color};'>{v:+.2f}%</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{k}:** {v}")

            st.markdown("---")
            col_m3, col_m4 = st.columns(2)

            with col_m3:
                st.markdown("#### Credit Markets")
                credit = macro.get("credit", {})
                for k, v in credit.items():
                    if isinstance(v, float):
                        color = "#00ff88" if v > 0 else "#ff4444"
                        st.markdown(
                            f"**{k}:** <span style='color:{color};'>{v:+.2f}%</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{k}:** {v}")

            with col_m4:
                st.markdown("#### Risk Appetite Indicators")
                risk_app = macro.get("risk_appetite", {})
                for k, v in risk_app.items():
                    if isinstance(v, float):
                        color = "#00ff88" if v > 0 else "#ff4444"
                        st.markdown(
                            f"**{k}:** <span style='color:{color};'>{v:+.2f}%</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{k}:** {v}")

            st.markdown("---")
            st.markdown("#### Top Movers")
            gainers = movers.get("gainers", [])
            losers = movers.get("losers", [])
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**Top Gainers**")
                for m in gainers[:5]:
                    sym = m.get("symbol", "—")
                    chg = m.get("change", 0)
                    px_m = m.get("price", 0)
                    st.markdown(
                        f"<span style='color:#00ff88;font-weight:600;'>{sym}</span> {chg:+.2f}% @ ${px_m:.2f}",
                        unsafe_allow_html=True,
                    )
            with mc2:
                st.markdown("**Top Losers**")
                for m in losers[:5]:
                    sym = m.get("symbol", "—")
                    chg = m.get("change", 0)
                    px_m = m.get("price", 0)
                    st.markdown(
                        f"<span style='color:#ff4444;font-weight:600;'>{sym}</span> {chg:+.2f}% @ ${px_m:.2f}",
                        unsafe_allow_html=True,
                    )

        # ── Tab 4: Volatility Surface ────────────────────────────────────────
        with db_tabs[3]:
            vix_meta = vix_data.get("_meta", {})
            vm1, vm2, vm3, vm4 = st.columns(4)
            vm1.metric("VIX Spot", f"{vix_meta.get('vix_spot', 0):.2f}")
            vm2.metric("VIX 3M", f"{vix_meta.get('vix3m', 0):.2f}")
            vm3.metric("VIX TS Slope", f"{vix_meta.get('vix_ts_slope', 0):+.2f}")
            vm4.metric("SKEW Index", f"{vix_meta.get('skew', 0):.1f}")

            vr_color = (
                "#00ff88"
                if "Low" in vix_meta.get("vix_regime", "")
                or "Normal" in vix_meta.get("vix_regime", "")
                else "#ffa500"
                if "Elevated" in vix_meta.get("vix_regime", "")
                else "#ff4444"
            )
            st.markdown(
                f"<div style='background:#161b22;border-left:4px solid {vr_color};"
                f"border-radius:6px;padding:10px;margin:8px 0;font-size:0.85rem;'>"
                f"<b style='color:{vr_color};'>VIX Regime:</b> {vix_meta.get('vix_regime', '')}<br>"
                f"<b style='color:#aaa;'>Term Structure:</b> {vix_meta.get('ts_regime', '')}<br>"
                f"<b style='color:#aaa;'>SKEW Signal:</b> {vix_meta.get('skew_signal', '')}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Vol surface table
            vix_rows = []
            for vname, vd in vix_data.items():
                if vname == "_meta":
                    continue
                if isinstance(vd, dict):
                    vix_rows.append(
                        {
                            "Instrument": vname,
                            "Value": f"{vd.get('value', 0):.2f}",
                            "1D Chg": f"{vd.get('chg_1d', 0):+.2f}%",
                            "1M Chg": f"{vd.get('chg_1m', 0):+.2f}%",
                        }
                    )
            if vix_rows:
                st.dataframe(
                    pd.DataFrame(vix_rows), use_container_width=True, hide_index=True
                )

            st.markdown("#### Volatility Regime Interpretation")
            vix_s = vix_meta.get("vix_spot", 20)
            skew_val = vix_meta.get("skew", 115)
            interp_lines = [
                f"• VIX at **{vix_s:.1f}** implies ~{vix_s / 16 * 100:.0f}% annualized expected volatility for the S&P 500.",
                f"• VIX term structure in **{vix_meta.get('ts_regime', '')}** — {'hedging demand > spot fear' if 'Contango' in vix_meta.get('ts_regime', '') else 'spot fear > forward expectations (spike risk)'}.",
                f"• SKEW at **{skew_val:.1f}**: {vix_meta.get('skew_signal', '')}.",
                f"• Options strategy implication: {'Credit spreads / sell premium' if vix_s > 28 else 'Buy cheap protection' if vix_s < 15 else 'Balanced — defined-risk strategies'}.",
            ]
            for line in interp_lines:
                st.markdown(
                    f"<div style='font-size:0.83rem;color:#ccc;padding:3px 0;'>{line}</div>",
                    unsafe_allow_html=True,
                )

        # ── Tab 5: Sector Rotation ───────────────────────────────────────────
        with db_tabs[4]:
            if sectors:
                sec_rows = []
                for name, d in sectors.items():
                    sec_rows.append(
                        {
                            "Sector": name,
                            "ETF": d.get("ticker", ""),
                            "1D %": f"{d.get('chg_1d', 0):+.2f}%",
                            "5D %": f"{d.get('chg_5d', 0):+.2f}%",
                            "1M %": f"{d.get('chg_1m', 0):+.2f}%",
                            "3M %": f"{d.get('chg_3m', 0):+.2f}%",
                            "Vol": f"{d.get('vol', 0):.1f}%",
                            "RSI": f"{d.get('rsi', 50):.0f}",
                            "MACD": d.get("macd", "—"),
                            ">50MA": "YES" if d.get("above_50") else "NO",
                            ">200MA": "YES" if d.get("above_200") else "NO",
                        }
                    )
                sec_df = pd.DataFrame(sec_rows).sort_values(
                    "1M %",
                    key=lambda x: (
                        x.str.replace("%", "").str.replace("+", "").astype(float)
                    ),
                    ascending=False,
                )

                def color_pct_sec(val):
                    try:
                        v = float(str(val).replace("%", "").replace("+", ""))
                        return (
                            "color: #00ff88"
                            if v > 0
                            else "color: #ff4444"
                            if v < 0
                            else ""
                        )
                    except Exception:
                        return ""

                st.dataframe(
                    sec_df.style.applymap(
                        color_pct_sec, subset=["1D %", "5D %", "1M %", "3M %"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart: sector 1-month performance
                try:
                    sec_names = [r["Sector"] for r in sec_rows]
                    sec_1m = [
                        float(r["1M %"].replace("%", "").replace("+", ""))
                        for r in sec_rows
                    ]
                    bar_colors = ["#00ff88" if v > 0 else "#ff4444" for v in sec_1m]
                    fig_sec = go.Figure(
                        go.Bar(
                            x=sec_names,
                            y=sec_1m,
                            marker_color=bar_colors,
                            text=[f"{v:+.1f}%" for v in sec_1m],
                            textposition="outside",
                        )
                    )
                    fig_sec.update_layout(
                        title=dict(
                            text="Sector 1-Month Performance (%)", font=dict(size=13)
                        ),
                        height=320,
                        template="plotly_dark",
                        yaxis_title="Return (%)",
                        xaxis_tickangle=-30,
                        margin=dict(l=10, r=10, t=35, b=60),
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)
                except Exception:
                    pass

                # Rotation narrative
                st.markdown("#### Sector Rotation Analysis")
                top_3 = sec_df.head(3)["Sector"].tolist()
                bot_3 = sec_df.tail(3)["Sector"].tolist()
                cycle_map = {
                    "Technology": "Late expansion / AI cycle",
                    "Financials": "Rising rates / steepening curve",
                    "Energy": "Commodity upcycle / inflation",
                    "Consumer Disc.": "Consumer confidence high",
                    "Industrials": "Infrastructure / capex cycle",
                    "Materials": "Commodity demand",
                    "Healthcare": "Defensive / late cycle",
                    "Consumer Staples": "Defensive / risk-off",
                    "Utilities": "Defensive / rate decline expected",
                    "Real Estate": "Rate-sensitive / inflation hedge",
                    "Communication": "Growth / advertising recovery",
                }
                st.markdown(f"**Leading sectors (1M):** {', '.join(top_3)}")
                for s in top_3:
                    signal = cycle_map.get(s, "—")
                    st.markdown(f"  → {s}: _{signal}_")
                st.markdown(f"**Lagging sectors (1M):** {', '.join(bot_3)}")
            else:
                st.warning("Sector data unavailable.")

        # ── Tab 6: FX & Commodities ──────────────────────────────────────────
        with db_tabs[5]:
            col_fx, col_comm = st.columns(2)

            with col_fx:
                st.markdown("#### FX Rates")
                if fx:
                    fx_rows = []
                    for name, d in fx.items():
                        fx_rows.append(
                            {
                                "Pair": name,
                                "Rate": f"{d.get('rate', 0):.4f}",
                                "1D %": f"{d.get('chg_1d', 0):+.3f}%",
                                "5D %": f"{d.get('chg_5d', 0):+.2f}%",
                                "1M %": f"{d.get('chg_1m', 0):+.2f}%",
                                "Vol": f"{d.get('vol', 0):.1f}%",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(fx_rows), use_container_width=True, hide_index=True
                    )

            with col_comm:
                st.markdown("#### Commodities")
                if commodities:
                    comm_rows = []
                    for name, d in commodities.items():
                        comm_rows.append(
                            {
                                "Commodity": name,
                                "Price": f"{d.get('price', 0):.2f}",
                                "1D %": f"{d.get('chg_1d', 0):+.2f}%",
                                "5D %": f"{d.get('chg_5d', 0):+.2f}%",
                                "1M %": f"{d.get('chg_1m', 0):+.2f}%",
                                "Vol": f"{d.get('vol', 0):.1f}%",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(comm_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Crypto
            st.markdown("---")
            st.markdown("#### Crypto")
            if crypto:
                crypto_rows = []
                for name, d in crypto.items():
                    crypto_rows.append(
                        {
                            "Asset": name,
                            "Price": f"${d.get('price', 0):,.2f}",
                            "1D %": f"{d.get('chg_1d', 0):+.2f}%",
                            "5D %": f"{d.get('chg_5d', 0):+.2f}%",
                            "1M %": f"{d.get('chg_1m', 0):+.2f}%",
                            "Vol": f"{d.get('vol', 0):.1f}%",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(crypto_rows), use_container_width=True, hide_index=True
                )

        # ── Tab 7: Cross-Asset Signals ───────────────────────────────────────
        with db_tabs[6]:
            if cross_signals:
                for sig_item in cross_signals:
                    sev = sig_item.get("severity", "LOW")
                    sev_color = (
                        "#ff4444"
                        if sev == "HIGH"
                        else "#ffa500"
                        if sev == "MEDIUM"
                        else "#e0c97f"
                    )
                    sev_bg = (
                        "#2e1010"
                        if sev == "HIGH"
                        else "#2e1e10"
                        if sev == "MEDIUM"
                        else "#2e2a10"
                    )
                    st.markdown(
                        f"<div style='background:{sev_bg};border-left:4px solid {sev_color};"
                        f"border-radius:6px;padding:10px 14px;margin:6px 0;'>"
                        f"<div style='display:flex;justify-content:space-between;'>"
                        f"<span style='color:{sev_color};font-weight:700;font-size:0.9rem;'>{sig_item.get('title', '')}</span>"
                        f"<span style='background:{sev_color};color:#000;font-size:0.7rem;font-weight:700;"
                        f"border-radius:3px;padding:2px 6px;'>{sev}</span>"
                        f"</div>"
                        f"<div style='color:#ccc;font-size:0.8rem;margin-top:5px;'>{sig_item.get('detail', '')}</div>"
                        f"<div style='color:#aaa;font-size:0.78rem;margin-top:4px;'>"
                        f"<b>Implication:</b> {sig_item.get('implication', '')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No significant cross-asset divergences detected at this time.")

        # ── Tab 8: Narrative Dislocations ────────────────────────────────────
        with db_tabs[7]:
            st.markdown("#### Narrative vs. Price Dislocation Analysis")
            st.caption(
                "Where the consensus story contradicts what the data actually shows — potential mispricing opportunities."
            )
            if dislocations:
                for dis in dislocations:
                    sev = dis.get("severity", "MEDIUM")
                    d_color = (
                        "#ff4444"
                        if sev == "HIGH"
                        else "#ffa500"
                        if sev == "MEDIUM"
                        else "#e0c97f"
                    )
                    with st.expander(
                        f"[{sev}] {dis.get('title', '')}",
                        expanded=True,
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                f"<div style='background:#1f1015;border-left:3px solid #ff6666;"
                                f"border-radius:5px;padding:8px;margin:4px 0;font-size:0.82rem;'>"
                                f"<b style='color:#ff9999;'>NARRATIVE:</b><br>{dis.get('narrative', '')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='background:#101f15;border-left:3px solid #66ff99;"
                                f"border-radius:5px;padding:8px;margin:4px 0;font-size:0.82rem;'>"
                                f"<b style='color:#99ffcc;'>REALITY / DATA:</b><br>{dis.get('reality', '')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        with c2:
                            st.markdown(
                                f"<div style='background:#1a1f2e;border-left:3px solid {d_color};"
                                f"border-radius:5px;padding:8px;margin:4px 0;font-size:0.82rem;'>"
                                f"<b style='color:{d_color};'>IMPLICATION:</b><br>{dis.get('implication', '')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='background:#161b22;border-left:3px solid #aaa;"
                                f"border-radius:5px;padding:8px;margin:4px 0;font-size:0.82rem;color:#ccc;'>"
                                f"<b>BASE RATE / PROBABILITY:</b><br>{dis.get('probability', '')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.success(
                    "No major narrative dislocations detected. Markets appear broadly consistent with macro fundamentals."
                )

        # ── Tab 9: Trade Ideas & Posture ─────────────────────────────────────
        with db_tabs[8]:
            st.markdown("#### Actionable Trade Ideas")
            st.caption(
                "Derived from regime detection, cross-asset analysis, and macro signals. Not financial advice."
            )
            if trade_ideas:
                for i, idea in enumerate(trade_ideas):
                    t_type = idea.get("type", "")
                    t_color = (
                        "#00ff88"
                        if "LONG" in t_type
                        else "#ff4444"
                        if "SHORT" in t_type
                        else "#e0c97f"
                    )
                    with st.expander(
                        f"{t_type}: {idea.get('instrument', '')}", expanded=(i < 3)
                    ):
                        ic1, ic2 = st.columns([2, 1])
                        with ic1:
                            st.markdown(
                                f"<div style='background:#161b22;border-left:4px solid {t_color};"
                                f"border-radius:6px;padding:10px;font-size:0.83rem;'>"
                                f"<b style='color:{t_color};'>THESIS:</b> {idea.get('thesis', '')}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if idea.get("risk"):
                                st.markdown(
                                    f"<div style='font-size:0.8rem;color:#ffaaaa;margin-top:5px;'><span style='display:inline-block;width:8px;height:8px;background:#ff4444;border-radius:50%;margin-right:5px;animation:pulse-dot 1.5s ease-in-out infinite;'></span>Risk: {idea['risk']}</div>",
                                    unsafe_allow_html=True,
                                )
                        with ic2:
                            st.metric("Timeframe", idea.get("timeframe", "—"))
                            st.metric("Suggested Size", idea.get("sizing", "—"))
            else:
                st.info("No high-conviction trade ideas generated at this time.")

            st.markdown("---")
            st.markdown("#### Portfolio Posture Summary")
            if posture:
                p_cols = st.columns(5)
                p_cols[0].metric("Posture", posture.get("label", "—"))
                p_cols[1].metric("Equities", posture.get("equity_pct", "—"))
                p_cols[2].metric("Bonds", posture.get("bond_pct", "—"))
                p_cols[3].metric("Alternatives", posture.get("alternatives", "—"))
                p_cols[4].metric("Cash", posture.get("cash", "—"))
                if posture.get("notes"):
                    st.info(posture["notes"])

elif selection == "Quant Portal":
    from quant_portal import render_quant_portal

    render_quant_portal()

elif selection == "Strategy Research Lab":
    from strategy_research_lab import render_strategy_research_lab

    render_strategy_research_lab()

elif selection == "Settings & Analytics":
    st.title("Analytics & Settings")
    from analytics_dashboard import show_analytics_dashboard

    show_analytics_dashboard()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<p style="color:{COLORS["text_secondary"]};font-size:0.65rem;text-align:center;'
    f'letter-spacing:0.5px;line-height:1.8;">'
    f"v4.0.0  |  Octavian AI<br>"
    f'<span style="color:{COLORS["gold_dark"]};font-weight:500;">by APB</span></p>',
    unsafe_allow_html=True,
)
