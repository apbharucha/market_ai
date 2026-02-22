"""
Octavian Trader Profile System — Comprehensive Personalization Engine
Author: APB - Octavian Team

Collects detailed trader preferences and tailors the entire app experience.
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from timeframe_analysis_engine import TimeframeScope, TimeframeAnalysisEngine

# --- Profile data management ---

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "trader_profiles")
os.makedirs(PROFILE_PATH, exist_ok=True)

# 
# PROFILE SCHEMA & DEFAULTS
# 

EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced", "Professional", "Institutional"]

TRADING_STYLES = {
    "Scalper": {"timeframe": "1m-15m", "hold": "seconds to minutes", "icon": ""},
    "Day Trader": {"timeframe": "5m-1h", "hold": "minutes to hours", "icon": ""},
    "Swing Trader": {"timeframe": "1h-1d", "hold": "days to weeks", "icon": ""},
    "Position Trader": {"timeframe": "1d-1w", "hold": "weeks to months", "icon": ""},
    "Long-Term Investor": {"timeframe": "1w-1mo", "hold": "months to years", "icon": ""},
}

RISK_PROFILES = {
    "Conservative": {"max_risk_pct": 1.0, "max_positions": 3, "color": "", "desc": "Capital preservation first"},
    "Moderate": {"max_risk_pct": 2.0, "max_positions": 5, "color": "[NOTE]", "desc": "Balanced risk/reward"},
    "Aggressive": {"max_risk_pct": 5.0, "max_positions": 10, "color": "#ff9800", "desc": "Growth-focused, higher drawdowns OK"},
    "Very Aggressive": {"max_risk_pct": 10.0, "max_positions": 20, "color": "", "desc": "Maximum returns, high volatility tolerance"},
}

ASSET_CLASSES = {
    "US Stocks": {"symbols": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"], "icon": ""},
    "International Stocks": {"symbols": ["BABA", "TSM", "ASML", "NVO", "SHOP", "SE"], "icon": ""},
    "ETFs": {"symbols": ["SPY", "QQQ", "IWM", "VTI", "ARKK", "XLK", "XLF"], "icon": ""},
    "Forex": {"symbols": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"], "icon": ""},
    "Crypto": {"symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"], "icon": "₿"},
    "Futures": {"symbols": ["ES=F", "NQ=F", "CL=F", "GC=F", "SI=F"], "icon": ""},
    "Options": {"symbols": ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"], "icon": ""},
    "Commodities": {"symbols": ["GLD", "SLV", "USO", "UNG", "DBA"], "icon": "[COIN]"},
    "Bonds/Fixed Income": {"symbols": ["TLT", "IEF", "LQD", "HYG", "TIP"], "icon": ""},
}

SECTORS_OF_INTEREST = [
    "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Materials", "Real Estate", "Utilities",
    "Communication Services", "AI & Robotics", "Clean Energy", "Biotech",
    "Semiconductors", "Cybersecurity", "EV & Autonomous", "Space & Defense",
]

PREFERRED_INDICATORS = [
    "RSI", "MACD", "Bollinger Bands", "Moving Averages (SMA/EMA)",
    "Volume Profile", "VWAP", "Fibonacci Retracements", "Ichimoku Cloud",
    "Stochastic Oscillator", "ATR", "OBV", "ADX", "Pivot Points",
    "Keltner Channels", "Parabolic SAR", "Williams %R",
]

TRADING_GOALS = [
    "Consistent income (weekly/monthly)",
    "Long-term wealth building",
    "Quick profits (short-term trades)",
    "Portfolio hedging",
    "Learning & skill development",
    "Retirement savings",
    "Beat the S&P 500",
    "Generate alpha with options",
    "Diversify across asset classes",
    "Build a dividend portfolio",
]

NEWS_PREFERENCES = [
    "Earnings reports", "Fed/Central bank decisions", "Macro economics",
    "Geopolitical events", "Sector rotation", "Insider trading",
    "Options flow", "Short interest", "IPOs & SPACs", "Crypto regulation",
    "AI/Tech developments", "Commodity supply/demand",
]

DEFAULT_PROFILE = {
    "name": "",
    "experience_level": "Intermediate",
    "trading_style": "Swing Trader",
    "risk_profile": "Moderate",
    "asset_classes": ["US Stocks", "ETFs"],
    "sectors_of_interest": ["Technology"],
    "watchlist": ["SPY", "QQQ", "AAPL", "NVDA"],
    "preferred_indicators": ["RSI", "MACD", "Moving Averages (SMA/EMA)"],
    "trading_goals": ["Long-term wealth building"],
    "news_preferences": ["Earnings reports", "Fed/Central bank decisions"],
    # Expanded Fields
    "portfolio_size": 10000.0,
    "time_horizon": "1-3 Years",
    "current_holdings": "", # JSON string or text
    "model_sensitivity": "Medium", # Low, Medium, High
    "notification_rules": {
        "price_alerts": True,
        "regime_change": True,
        "model_signals": False,
        "news_impact": False
    },
    # Existing
    "capital_range": "$10,000 - $50,000",
    "max_loss_per_trade_pct": 2.0,
    "profit_target_pct": 5.0,
    "preferred_session": "US Market Hours",
    "alerts_enabled": True,
    "show_advanced_metrics": False,
    "dark_pool_alerts": False,
    "options_greeks": False,
    "ai_commentary_style": "balanced",
    "created_at": None,
    "updated_at": None,
    "interaction_count": 0,
    "queries_history": [],
}

CAPITAL_RANGES = [
    "Under $1,000",
    "$1,000 - $5,000",
    "$5,000 - $10,000",
    "$10,000 - $50,000",
    "$50,000 - $100,000",
    "$100,000 - $500,000",
    "$500,000 - $1,000,000",
    "$1,000,000+",
]

TIME_HORIZONS = [
    "Scalp (Minutes)",
    "Intraday (Hours)",
    "Swing (Days-Weeks)",
    "Position (Weeks-Months)",
    "1-3 Years",
    "5+ Years",
    "Retirement"
]

MODEL_SENSITIVITIES = ["Low (High Confidence Only)", "Medium (Balanced)", "High (More Signals)"]

TRADING_SESSIONS = [
    "Pre-Market (4:00-9:30 ET)",
    "US Market Hours (9:30-16:00 ET)",
    "After-Hours (16:00-20:00 ET)",
    "Asian Session (19:00-4:00 ET)",
    "European Session (3:00-12:00 ET)",
    "24/7 (Crypto)",
    "All Sessions",
]

AI_STYLES = {
    "conservative": "Focus on risk management, downside protection, and high-probability setups only.",
    "balanced": "Balanced view of risk and reward. Present both bull and bear cases.",
    "aggressive": "Focus on highest-return opportunities. Emphasize momentum and breakout trades.",
    "educational": "Explain reasoning in detail. Great for learning.",
    "institutional": "Concise, data-dense. Assume advanced knowledge.",
}

WATCHLIST_PRESETS = {
    " Top Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"],
    " Major ETFs": ["SPY", "QQQ", "IWM", "VTI", "DIA", "GLD", "TLT"],
    "₿ Crypto Majors": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"],
    " Forex Majors": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"],
    " Futures": ["ES=F", "NQ=F", "CL=F", "GC=F", "SI=F"],
    " Biotech": ["LLY", "MRNA", "PFE", "ABBV", "GILD", "REGN"],
    " Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
    " Financials": ["JPM", "BAC", "GS", "MS", "BLK", "SCHW"],
    " AI & Semis": ["NVDA", "AMD", "AVGO", "SMCI", "AI", "PLTR", "CRWD"],
}


# 
# PROFILE PERSISTENCE
# 

def _profile_file(profile_id: str = "default") -> str:
    return os.path.join(PROFILE_PATH, f"{profile_id}.json")


def save_profile(profile: Dict, profile_id: str = "default"):
    profile["updated_at"] = datetime.now().isoformat()
    if not profile.get("created_at"):
        profile["created_at"] = profile["updated_at"]
    with open(_profile_file(profile_id), "w") as f:
        json.dump(profile, f, indent=2)
    st.session_state["trader_profile"] = profile


def load_profile(profile_id: str = "default") -> Dict:
    path = _profile_file(profile_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            # Merge with defaults for any new fields
            merged = {**DEFAULT_PROFILE, **loaded}
            return merged
        except Exception:
            pass
    return {**DEFAULT_PROFILE}


def get_trader_profile() -> Dict:
    if "trader_profile" not in st.session_state:
        st.session_state["trader_profile"] = load_profile()
    return st.session_state["trader_profile"]


def list_profiles() -> List[str]:
    profiles = []
    for f in os.listdir(PROFILE_PATH):
        if f.endswith(".json"):
            profiles.append(f.replace(".json", ""))
    return profiles or ["default"]


# 
# INTERACTION LEARNING
# 

def learn_from_user_interaction(query: str, response: Any = None):
    """Track user queries to learn preferences over time."""
    profile = get_trader_profile()
    profile["interaction_count"] = profile.get("interaction_count", 0) + 1
    history = profile.get("queries_history", [])
    history.append({
        "query": query[:200],
        "timestamp": datetime.now().isoformat(),
    })
    # Keep last 100 queries
    profile["queries_history"] = history[-100:]
    save_profile(profile)


def get_timeframe_context_for_analysis() -> Dict:
    """Get timeframe context based on trader profile."""
    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])

    from timeframe_analysis_engine import TimeframeScope
    style_to_scope = {
        "Scalper": TimeframeScope.SCALPING,
        "Day Trader": TimeframeScope.INTRADAY,
        "Swing Trader": TimeframeScope.SWING,
        "Position Trader": TimeframeScope.POSITION,
        "Long-Term Investor": TimeframeScope.INVESTMENT,
    }
    return {
        "primary_timeframe": style_to_scope.get(style, TimeframeScope.SWING),
        "style": style,
        "hold_period": style_info["hold"],
    }


def get_recommendation_style() -> str:
    """Get the AI commentary style preference."""
    profile = get_trader_profile()
    return profile.get("ai_commentary_style", "balanced")


def get_watchlist() -> List[str]:
    """Get user's watchlist symbols."""
    profile = get_trader_profile()
    return profile.get("watchlist", ["SPY", "QQQ", "AAPL"])


def get_risk_params() -> Dict:
    """Get risk parameters for position sizing and alerts."""
    profile = get_trader_profile()
    risk_name = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk_name, RISK_PROFILES["Moderate"])
    return {
        "profile": risk_name,
        "max_risk_pct": risk_info["max_risk_pct"],
        "max_positions": risk_info["max_positions"],
        "max_loss_per_trade_pct": profile.get("max_loss_per_trade_pct", 2.0),
        "profit_target_pct": profile.get("profit_target_pct", 5.0),
        "capital_range": profile.get("capital_range", "$10,000 - $50,000"),
    }


def get_preferred_assets() -> List[str]:
    """Get flat list of preferred symbols based on selected asset classes."""
    profile = get_trader_profile()
    classes = profile.get("asset_classes", ["US Stocks"])
    symbols = []
    for cls in classes:
        if cls in ASSET_CLASSES:
            symbols.extend(ASSET_CLASSES[cls]["symbols"])
    # Add watchlist
    symbols.extend(profile.get("watchlist", []))
    return list(dict.fromkeys(symbols))  # Dedupe preserving order


def should_show_advanced() -> bool:
    profile = get_trader_profile()
    return profile.get("show_advanced_metrics", False) or \
           profile.get("experience_level", "Intermediate") in ["Advanced", "Professional", "Institutional"]


# 
# SIDEBAR WIDGET (compact, for every page)
# 

def show_trader_selection(key_suffix: str = ""):
    """Compact sidebar widget showing current profile summary."""
    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])
    risk = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk, RISK_PROFILES["Moderate"])
    exp = profile.get("experience_level", "Intermediate")

    st.sidebar.markdown(
        f"**{style_info['icon']} {style}** | {risk_info['color']} {risk} | {exp}"
    )
    wl = profile.get("watchlist", [])
    if wl:
        st.sidebar.caption(f"Watchlist: {', '.join(wl[:6])}")


# 
# FULL PROFILE SETTINGS PAGE
# 

def _safe_index(lst: list, value, default: int = 0) -> int:
    """Safely get index of value in list, returning default if not found."""
    try:
        return lst.index(value)
    except ValueError:
        # Try partial match (e.g. "US Market Hours" matches "US Market Hours (9:30-16:00 ET)")
        val_lower = str(value).lower()
        for i, item in enumerate(lst):
            if val_lower in str(item).lower() or str(item).lower() in val_lower:
                return i
        return default


def _render_watchlist_editor(profile: Dict) -> List[str]:
    """Rich watchlist editor with categories, presets, and validation."""
    st.subheader("[PIN] Watchlist Management")

    current_wl = list(profile.get("watchlist", ["SPY", "QQQ", "AAPL"]))

    # Show current watchlist with remove buttons
    if current_wl:
        st.markdown(f"**Current Watchlist** ({len(current_wl)} symbols)")
        # Display in rows of 6
        rows = [current_wl[i:i+6] for i in range(0, len(current_wl), 6)]
        remove_syms = []
        for row_idx, row in enumerate(rows):
            cols = st.columns(len(row))
            for col, sym in zip(cols, row):
                with col:
                    # Color code by type
                    if sym.endswith("-USD"):
                        icon = "₿"
                    elif "/" in sym:
                        icon = ""
                    elif "=F" in sym:
                        icon = ""
                    elif sym.startswith("^"):
                        icon = ""
                    else:
                        icon = ""
                    if st.button(f" {icon} {sym}", key=f"wl_rm_{sym}_{row_idx}",
                                 use_container_width=True):
                        remove_syms.append(sym)
        for sym in remove_syms:
            if sym in current_wl:
                current_wl.remove(sym)
    else:
        st.info("Your watchlist is empty. Add symbols below.")

    st.markdown("---")

    # Add symbols
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_syms = st.text_input(
            "Add Symbols (comma-separated)",
            placeholder="AAPL, USD/JPY, BTC-USD, ES=F",
            key="wl_add_input",
            help="Stocks (AAPL), Forex (USD/JPY or EURUSD=X), Crypto (BTC-USD), Futures (ES=F)"
        )
    with col_add2:
        st.markdown("<br>", unsafe_allow_html=True)
        add_clicked = st.button(" Add", key="wl_add_btn", type="primary", use_container_width=True)

    if add_clicked and new_syms:
        for s in new_syms.split(","):
            s = s.strip().upper()
            if s and s not in current_wl:
                current_wl.append(s)

    # Preset quick-add
    st.markdown("**Quick Add Presets:**")
    preset_cols = st.columns(3)
    for idx, (preset_name, preset_syms) in enumerate(WATCHLIST_PRESETS.items()):
        with preset_cols[idx % 3]:
            if st.button(preset_name, key=f"wl_preset_{idx}", use_container_width=True):
                for s in preset_syms:
                    if s not in current_wl:
                        current_wl.append(s)

    # Clear all
    if current_wl and st.button(" Clear Entire Watchlist", key="wl_clear"):
        current_wl.clear()

    return current_wl


def show_profile_settings():
    """Full profile configuration page."""
    st.title(" Trader Profile & Preferences")
    st.markdown("Configure your trading profile to personalize your Octavian experience.")

    profile = get_trader_profile()

    # Profile selector
    profiles = list_profiles()
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        active = st.selectbox("Active Profile", profiles, index=0, key="profile_select")
    with col_p2:
        new_name = st.text_input("New Profile", key="new_profile_name", placeholder="my_profile")
        if st.button(" Create", key="create_profile") and new_name:
            save_profile({**DEFAULT_PROFILE, "name": new_name}, new_name.strip().lower().replace(" ", "_"))
            st.success(f"Profile '{new_name}' created!")
            st.rerun()

    if active != "default":
        profile = load_profile(active)
        st.session_state["trader_profile"] = profile

    st.markdown("---")

    #  Section 1: Identity & Experience 
    st.subheader(" Experience & Style")
    col1, col2, col3 = st.columns(3)
    with col1:
        profile["name"] = st.text_input("Display Name", value=profile.get("name", ""), key="prof_name")
    with col2:
        profile["experience_level"] = st.selectbox(
            "Experience Level", EXPERIENCE_LEVELS,
            index=_safe_index(EXPERIENCE_LEVELS, profile.get("experience_level", "Intermediate"), 1),
            key="prof_exp"
        )
    with col3:
        styles_list = list(TRADING_STYLES.keys())
        profile["trading_style"] = st.selectbox(
            "Trading Style", styles_list,
            index=_safe_index(styles_list, profile.get("trading_style", "Swing Trader"), 2),
            key="prof_style",
            format_func=lambda x: f"{TRADING_STYLES[x]['icon']} {x} ({TRADING_STYLES[x]['hold']})"
        )

    col4, col5 = st.columns(2)
    with col4:
        profile["time_horizon"] = st.selectbox(
            "Time Horizon", TIME_HORIZONS,
            index=_safe_index(TIME_HORIZONS, profile.get("time_horizon", "1-3 Years"), 4),
            key="prof_horizon"
        )
    with col5:
        # Portfolio Size (Numeric Input)
        profile["portfolio_size"] = st.number_input(
            "Portfolio Size ($)", min_value=0.0, step=1000.0,
            value=float(profile.get("portfolio_size", 10000.0)),
            format="%.2f", key="prof_size_val"
        )

    # Show style details
    style_info = TRADING_STYLES[profile["trading_style"]]
    st.info(f"**{style_info['icon']} {profile['trading_style']}** — "
            f"Typical timeframe: {style_info['timeframe']} | Hold period: {style_info['hold']}")

    #  Section 2: Risk Management 
    st.markdown("---")
    st.subheader(" Risk Management")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        risk_list = list(RISK_PROFILES.keys())
        profile["risk_profile"] = st.selectbox(
            "Risk Tolerance", risk_list,
            index=_safe_index(risk_list, profile.get("risk_profile", "Moderate"), 1),
            key="prof_risk",
            format_func=lambda x: f"{RISK_PROFILES[x]['color']} {x} — {RISK_PROFILES[x]['desc']}"
        )
    with col_r2:
        profile["capital_range"] = st.selectbox(
            "Trading Capital", CAPITAL_RANGES,
            index=_safe_index(CAPITAL_RANGES, profile.get("capital_range", "$10,000 - $50,000"), 3),
            key="prof_capital"
        )
    with col_r3:
        profile["preferred_session"] = st.selectbox(
            "Trading Session", TRADING_SESSIONS,
            index=_safe_index(TRADING_SESSIONS, profile.get("preferred_session", "US Market Hours"), 1),
            key="prof_session"
        )

    col_r4, col_r5 = st.columns(2)
    with col_r4:
        profile["max_loss_per_trade_pct"] = st.slider(
            "Max Loss Per Trade (%)", 0.5, 20.0,
            value=float(profile.get("max_loss_per_trade_pct", 2.0)),
            step=0.5, key="prof_max_loss"
        )
    with col_r5:
        profile["profit_target_pct"] = st.slider(
            "Profit Target Per Trade (%)", 1.0, 50.0,
            value=float(profile.get("profit_target_pct", 5.0)),
            step=1.0, key="prof_target"
        )

    risk_info = RISK_PROFILES[profile["risk_profile"]]
    rr_ratio = profile["profit_target_pct"] / max(profile["max_loss_per_trade_pct"], 0.1)
    st.caption(f"Risk/Reward Ratio: **{rr_ratio:.1f}:1** | "
               f"Max concurrent positions: **{risk_info['max_positions']}** | "
               f"Max risk per trade: **{risk_info['max_risk_pct']}%**")

    #  Section 3: Watchlist (NEW ENHANCED) 
    st.markdown("---")
    profile["watchlist"] = _render_watchlist_editor(profile)

    #  Section 4: Asset Preferences 
    st.markdown("---")
    st.subheader(" Asset Classes & Sectors")

    profile["asset_classes"] = st.multiselect(
        "Preferred Asset Classes",
        list(ASSET_CLASSES.keys()),
        default=profile.get("asset_classes", ["US Stocks", "ETFs"]),
        format_func=lambda x: f"{ASSET_CLASSES[x]['icon']} {x}",
        key="prof_assets"
    )

    profile["sectors_of_interest"] = st.multiselect(
        "Sectors of Interest",
        SECTORS_OF_INTEREST,
        default=profile.get("sectors_of_interest", ["Technology"]),
        key="prof_sectors"
    )

    #  Section 5: Analysis Preferences 
    st.markdown("---")
    st.subheader(" Analysis & Indicator Preferences")

    profile["preferred_indicators"] = st.multiselect(
        "Preferred Technical Indicators",
        PREFERRED_INDICATORS,
        default=profile.get("preferred_indicators", ["RSI", "MACD", "Moving Averages (SMA/EMA)"]),
        key="prof_indicators"
    )

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        profile["show_advanced_metrics"] = st.toggle(
            "Show Advanced Metrics (Greeks, Sharpe, Sortino, etc.)",
            value=profile.get("show_advanced_metrics", False),
            key="prof_advanced"
        )
        profile["dark_pool_alerts"] = st.toggle(
            "Dark Pool / Institutional Flow Alerts",
            value=profile.get("dark_pool_alerts", False),
            key="prof_darkpool"
        )
    with col_a2:
        profile["options_greeks"] = st.toggle(
            "Show Options Greeks & Chain",
            value=profile.get("options_greeks", False),
            key="prof_greeks"
        )
        profile["alerts_enabled"] = st.toggle(
            "Enable Price & Signal Alerts",
            value=profile.get("alerts_enabled", True),
            key="prof_alerts"
        )

    #  Section 6: Goals & AI Style 
    st.markdown("---")
    st.subheader(" Goals, Sensitivity & Holdings")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        profile["model_sensitivity"] = st.selectbox(
            "Model Sensitivity", MODEL_SENSITIVITIES,
            index=_safe_index(MODEL_SENSITIVITIES, profile.get("model_sensitivity", "Medium"), 1),
            key="prof_sensitivity"
        )
    with col_g2:
        profile["trading_goals"] = st.multiselect(
            "Trading Goals", TRADING_GOALS,
            default=profile.get("trading_goals", ["Long-term wealth building"]),
            key="prof_goals"
        )

    # Current Holdings (Simple Text Area for now)
    profile["current_holdings"] = st.text_area(
        "Current Holdings (Optional - for Context)",
        value=profile.get("current_holdings", ""),
        placeholder="e.g. 100 AAPL @ 150, 50 MSFT @ 300...",
        help="Paste a summary of your positions here so the AI can give context-aware advice."
    )

    st.markdown("---")
    st.subheader(" Notification Rules")
    
    notif_rules = profile.get("notification_rules", {})
    c_n1, c_n2, c_n3, c_n4 = st.columns(4)
    with c_n1:
        notif_rules["price_alerts"] = st.checkbox("Price Alerts", value=notif_rules.get("price_alerts", True), key="n_price")
    with c_n2:
        notif_rules["regime_change"] = st.checkbox("Regime Change", value=notif_rules.get("regime_change", True), key="n_regime")
    with c_n3:
        notif_rules["model_signals"] = st.checkbox("Model Signals", value=notif_rules.get("model_signals", False), key="n_model")
    with c_n4:
        notif_rules["news_impact"] = st.checkbox("High Impact News", value=notif_rules.get("news_impact", False), key="n_news")
    profile["notification_rules"] = notif_rules

    profile["news_preferences"] = st.multiselect(
        "News & Event Preferences",
        NEWS_PREFERENCES,
        default=profile.get("news_preferences", ["Earnings reports", "Fed/Central bank decisions"]),
        key="prof_news"
    )

    ai_styles_list = list(AI_STYLES.keys())
    profile["ai_commentary_style"] = st.selectbox(
        "AI Commentary Style",
        ai_styles_list,
        index=_safe_index(ai_styles_list, profile.get("ai_commentary_style", "balanced"), 1),
        key="prof_ai_style",
        format_func=lambda x: f"{x.title()} — {AI_STYLES[x]}"
    )

    #  Save 
    st.markdown("---")
    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        if st.button(" Save Profile", type="primary", key="save_profile", use_container_width=True):
            save_profile(profile, active)
            st.success(" Profile saved!")
            st.rerun()
    with col_s2:
        if profile.get("updated_at"):
            st.caption(f"Last saved: {profile['updated_at'][:19]} | "
                       f"Interactions: {profile.get('interaction_count', 0)}")

    #  Profile Summary Card 
    st.markdown("---")
    st.subheader(" Profile Summary")
    _render_profile_card(profile)


def _render_profile_card(profile: Dict):
    """Render a visual profile summary card."""
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])
    risk = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk, RISK_PROFILES["Moderate"])

    name = profile.get("name") or "Trader"
    wl = profile.get("watchlist", [])
    goals = profile.get("trading_goals", [])

    st.markdown(f"""
| Field | Value |
|-------|-------|
| **Name** | {name} |
| **Experience** | {profile.get('experience_level', 'Intermediate')} |
| **Style** | {style_info['icon']} {style} ({style_info['hold']}) |
| **Risk** | {risk_info['color']} {risk} — {risk_info['desc']} |
| **Capital** | {profile.get('capital_range', 'N/A')} |
| **Session** | {profile.get('preferred_session', 'US Market Hours')} |
| **Max Loss/Trade** | {profile.get('max_loss_per_trade_pct', 2.0)}% |
| **Profit Target** | {profile.get('profit_target_pct', 5.0)}% |
| **Watchlist** | {', '.join(wl[:8]) if wl else 'None'} |
| **Goals** | {', '.join(goals[:3]) if goals else 'None set'} |
| **AI Style** | {profile.get('ai_commentary_style', 'balanced').title()} |
""")


# 
# PERSONALIZED DASHBOARD (used by main.py)
# 

def show_personalized_dashboard():
    """Show a personalized 'My Dashboard' based on trader profile."""
    import concurrent.futures

    profile = get_trader_profile()
    style = profile.get("trading_style", "Swing Trader")
    style_info = TRADING_STYLES.get(style, TRADING_STYLES["Swing Trader"])
    risk = profile.get("risk_profile", "Moderate")
    risk_info = RISK_PROFILES.get(risk, RISK_PROFILES["Moderate"])
    name = profile.get("name") or "Trader"

    st.markdown(f"### {style_info['icon']} Welcome back, **{name}**")
    st.caption(f"{style} | {risk_info['color']} {risk} | {profile.get('experience_level', '')} | "
               f"Capital: {profile.get('capital_range', 'N/A')}")

    #  Watchlist Quick View 
    watchlist = profile.get("watchlist", ["SPY", "QQQ", "AAPL"])
    if watchlist:
        st.subheader("[PIN] Your Watchlist")
        from watchlist_dashboard import _fetch_symbol_data, _safe_close, _compute_technicals

        def _quick_analyze(sym):
            try:
                df, at = _fetch_symbol_data(sym, "1mo")
                if df is None or df.empty:
                    return None
                close = _safe_close(df)
                if close is None or len(close) < 5:
                    return None
                tech = _compute_technicals(close, df)
                price = float(close.iloc[-1])
                prev = float(close.iloc[-2]) if len(close) >= 2 else price
                change = (price / prev - 1) * 100
                return {"symbol": sym, "price": price, "change": change, **tech}
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_quick_analyze, s): s for s in watchlist[:12]}
            wl_results = []
            done, _ = concurrent.futures.wait(futures, timeout=15)
            for f in done:
                try:
                    r = f.result(timeout=1)
                    if r:
                        wl_results.append(r)
                except Exception:
                    pass

        if wl_results:
            cols_per_row = min(len(wl_results), 4)
            rows = [wl_results[i:i + cols_per_row] for i in range(0, len(wl_results), cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for col, r in zip(cols, row):
                    with col:
                        sig_icon = "" if r.get("signal") == "BULLISH" else "" if r.get("signal") == "BEARISH" else ""
                        chg = r.get("change", 0)
                        st.metric(
                            f"{r['symbol']} {sig_icon}",
                            f"${r['price']:,.2f}",
                            f"{chg:+.2f}%"
                        )
                        rsi = r.get("metrics", {}).get("rsi")
                        if rsi:
                            st.caption(f"RSI: {rsi:.0f} | Conf: {r.get('confidence', 0):.0%}")

    #  Tailored Tips Based on Profile 
    st.markdown("---")
    st.subheader(" Personalized Insights")

    tips = []
    exp = profile.get("experience_level", "Intermediate")
    if exp == "Beginner":
        tips.append(" **Tip:** Start with paper trading before risking real capital. "
                     "Focus on 1-2 setups and master them.")
        tips.append(" **Risk Rule:** Never risk more than 1-2% of your account on a single trade.")
    elif exp == "Intermediate":
        tips.append(" **Tip:** Review your trade journal weekly. Look for patterns in your winners and losers.")
        tips.append(" **Strategy:** Consider adding sector rotation to your analysis toolkit.")
    elif exp in ["Advanced", "Professional"]:
        tips.append("[CALC] **Advanced:** Use the Quant Terminal for multi-factor analysis and backtesting.")
        tips.append(" **Edge:** Monitor dark pool activity and options flow for institutional sentiment.")
    elif exp == "Institutional":
        tips.append(" **Institutional:** Use correlation analysis to manage portfolio-level risk.")
        tips.append(" **Execution:** Consider time-weighted positioning for larger orders.")

    if style == "Scalper":
        tips.append(" **Scalper Focus:** Monitor the 1m/5m charts. Use Level 2 data and tight stops.")
    elif style == "Day Trader":
        tips.append(" **Day Trade Focus:** Key hours are 9:30-11:30 and 14:00-16:00 ET for highest volume.")
    elif style == "Swing Trader":
        tips.append(" **Swing Focus:** Look for daily chart breakouts with volume confirmation. Hold 2-10 days.")
    elif style == "Position Trader":
        tips.append(" **Position Focus:** Weekly chart trends matter most. Let winners run.")
    elif style == "Long-Term Investor":
        tips.append(" **Investor Focus:** Focus on fundamentals and macro trends. DCA on dips.")

    if risk == "Conservative":
        tips.append(" **Risk:** Stick to large-cap, liquid names. Avoid leverage and meme stocks.")
    elif risk == "Very Aggressive":
        tips.append(" **Risk:** High risk tolerance noted. Ensure position sizing discipline even on high-conviction trades.")

    for tip in tips[:4]:
        st.info(tip)

    #  Recommended Actions 
    st.markdown("---")
    st.subheader(" Recommended Actions")

    goals = profile.get("trading_goals", [])
    actions = []
    if "Consistent income (weekly/monthly)" in goals:
        actions.append(("", "Review covered call opportunities on your watchlist", "Navigate to Quant Terminal → Backtest"))
    if "Beat the S&P 500" in goals:
        actions.append(("", "Compare your watchlist performance vs SPY", "Go to Symbol Analysis → Deep Dive SPY"))
    if "Learning & skill development" in goals:
        actions.append(("", "Run a simulation to practice without risk", "Go to Simulation Hub → Run Simulation"))
    if "Generate alpha with options" in goals:
        actions.append(("", "Check unusual options activity on your watchlist", "Ask Octavian: 'Options flow for AAPL'"))
    if "Diversify across asset classes" in goals:
        actions.append(("", "Check correlation between your positions", "Go to Quant Terminal → Correlation"))

    if not actions:
        actions.append(("[SCAN]", "Scan the market for opportunities matching your style", "Go to Market Scanner"))

    for icon, action, how in actions[:4]:
        st.markdown(f"{icon} **{action}**")
        st.caption(f"→ {how}")

    #  Quick Stats 
    st.markdown("---")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        st.metric("Interactions", profile.get("interaction_count", 0))
    with col_q2:
        st.metric("Watchlist Size", len(profile.get("watchlist", [])))
    with col_q3:
        st.metric("Asset Classes", len(profile.get("asset_classes", [])))
    with col_q4:
        updated = profile.get("updated_at", "Never")
        if updated and updated != "Never":
            updated = updated[:10]
        st.metric("Last Updated", updated)

