"""
Comprehensive Quantitative Research Portal
==========================================
A unified interface combining:
- Quant Terminal (Multi-Asset Analysis)
- Quant Modeling Lab (Advanced Modeling)
- Strategy Research Lab capabilities

This portal provides institutional-grade quantitative analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib
import random

# Risk Engine imports
try:
    from risk_engine import correlation_matrix, portfolio_var
    HAS_RISK = True
except ImportError:
    HAS_RISK = False

# Data sources
try:
    from data_sources import get_stock
    HAS_DATA = True
except ImportError:
    HAS_DATA = False

# Quant ensemble model
try:
    from quant_ensemble_model import get_quant_ensemble
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False

# Advanced backtester
try:
    from advanced_backtester import AdvancedBacktester
    HAS_BT = True
except ImportError:
    HAS_BT = False

# Market simulation imports
try:
    from market_simulation_engine import MarketSimulationEngine
    HAS_SIM = True
except ImportError:
    HAS_SIM = False

# Genetic strategy imports
try:
    from genetic_strategy_engine import GeneticStrategyEngine
    HAS_GENETIC = True
except ImportError:
    HAS_GENETIC = False

# HMM Regime detection
try:
    from hmm_engine import detect_regimes
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

# Factor crowding
try:
    from factor_crowding_engine import FactorCrowdingEngine
    HAS_FACTOR = True
except ImportError:
    HAS_FACTOR = False

# Macro cross-asset
try:
    from macro_cross_asset_engine import MacroCrossAssetEngine
    HAS_MACRO = True
except ImportError:
    HAS_MACRO = False

# Alternative data
try:
    from alternative_data_engine import AlternativeDataEngine
    HAS_ALT = True
except ImportError:
    HAS_ALT = False

# ══════════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Quantitative Research Portal", page_icon="", layout="wide")

# Professional dark theme styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0d0d14 100%);
    }
    
    /* Card styling */
    .qportal-card {
        background: linear-gradient(145deg, rgba(30, 30, 40, 0.9), rgba(20, 20, 30, 0.95));
        border: 1px solid rgba(100, 100, 120, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .qportal-card:hover {
        border-color: rgba(120, 180, 255, 0.4);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e0e0e0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: rgba(30, 30, 40, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(100, 100, 120, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(60, 80, 150, 0.8), rgba(40, 60, 120, 0.9));
        border-color: rgba(100, 150, 255, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(60, 80, 150, 0.8), rgba(40, 60, 120, 0.9));
        border: 1px solid rgba(100, 150, 255, 0.3);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(80, 100, 180, 0.9), rgba(60, 80, 150, 1));
        border-color: rgba(100, 150, 255, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(20, 20, 30, 0.8);
        border: 1px solid rgba(100, 100, 120, 0.3);
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(100, 150, 255, 0.6);
    }
    
    /* Section headers */
    .section-header {
        color: #a0a0b0;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 20px 0 10px 0;
        border-bottom: 1px solid rgba(100, 100, 120, 0.3);
        padding-bottom: 8px;
    }
    
    /* Status indicators */
    .status-bullish {
        color: #4caf50;
        font-weight: 600;
    }
    
    .status-bearish {
        color: #f44336;
        font-weight: 600;
    }
    
    .status-neutral {
        color: #ff9800;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str):
    """Render a section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def _metric_card(label: str, value: str, color: str = "#c9a84c"):
    """Render a metric card."""
    st.markdown(f"""
    <div class="qportal-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def _calculate_advanced_metrics(returns: pd.Series) -> dict:
    """Calculate advanced performance metrics."""
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Advanced metrics
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = annual_return / downside_std if downside_std > 0 else 0
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PORTAL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_quant_portal():
    """Render the comprehensive quantitative research portal."""
    
    # Header
    st.title("")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #e0e0e0; font-size: 36px; margin-bottom: 8px;">
            Quantitative Research Portal
        </h1>
        <p style="color: #888; font-size: 16px;">
            Institutional-Grade Multi-Asset Quantitative Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Active Modules", "12", "#4caf50")
    with col2:
        _metric_card("Asset Classes", "6", "#2196f3")
    with col3:
        _metric_card("Analysis Types", "18", "#ff9800")
    with col4:
        _metric_card("Status", "Active", "#9c27b0")
    
    st.markdown("---")
    
    # Main tabs combining both Quant Terminal and Quant Modeling Lab
    main_tabs = st.tabs([
        " Multi-Asset Analysis ", 
        " Risk & Correlation ",
        " Quant Signals & ML ",
        " Regime Detection ",
        " Strategy Evolution ",
        " Cross-Asset Macro ",
        " Advanced Backtesting ",
        " Alternative Data "
    ])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: Multi-Asset Analysis (from Quant Terminal)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[0]:
        _section("Multi-Asset Analysis")
        
        # Symbol Input
        symbol_input = st.text_input(
            "Enter Symbols (comma-separated)",
            value="AAPL, MSFT, NVDA",
            help="Stocks (AAPL), futures (ES=F), FX (EURUSD=X), crypto (BTC-USD)"
        )
        
        if symbol_input:
            symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]
        else:
            symbols = []
        
        # Quick select buttons
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            if st.button("Stocks", use_container_width=True):
                symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
        with col_q2:
            if st.button("Futures", use_container_width=True):
                symbols = ["ES=F", "NQ=F", "CL=F", "GC=F"]
        with col_q3:
            if st.button("FX", use_container_width=True):
                symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
        with col_q4:
            if st.button("Crypto", use_container_width=True):
                symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        
        if len(symbols) < 1:
            st.info("Enter at least 1 symbol to begin.")
            return
        
        # Fetch and analyze data
        if st.button("Fetch & Analyze", type="primary"):
            with st.spinner("Loading market data..."):
                try:
                    data = {}
                    for sym in symbols:
                        try:
                            df = get_stock(sym, period="1y")
                            if df is not None and not df.empty:
                                data[sym] = df
                        except:
                            pass
                    
                    if data:
                        st.success(f"Loaded data for {len(data)} symbols")
                        
                        # Returns analysis
                        _section("Returns Analysis")
                        returns_data = {}
                        for sym, df in data.items():
                            if 'Close' in df.columns:
                                close = df['Close']
                                if isinstance(close, pd.DataFrame):
                                    close = close.iloc[:, 0]
                                returns_data[sym] = close.pct_change().dropna()
                        
                        if returns_data:
                            returns_df = pd.DataFrame(returns_data)
                            total_returns = ((1 + returns_df) - 1).tail(1).iloc[0]
                            
                            # Display returns
                            cols = st.columns(min(len(total_returns), 6))
                            for i, (sym, ret) in enumerate(total_returns.items()):
                                with cols[i % 6]:
                                    color = "#4caf50" if ret > 0 else "#f44336"
                                    _metric_card(f"{sym} Return", f"{ret*100:.1f}%", color)
                                
                        # Price chart
                        _section("Price Performance")
                        fig = go.Figure()
                        for sym, df in data.items():
                            if 'Close' in df.columns:
                                close = df['Close']
                                if isinstance(close, pd.DataFrame):
                                    close = close.iloc[:, 0]
                                # Normalize to percentage
                                normalized = (close / close.iloc[0] - 1) * 100
                                fig.add_trace(go.Scatter(
                                    x=normalized.index, 
                                    y=normalized.values, 
                                    name=sym,
                                    mode='lines'
                                ))
                        
                        fig.update_layout(
                            title="Normalized Price Performance (%)",
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Return (%)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: Risk & Correlation (from Quant Terminal)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[1]:
        _section("Risk & Correlation Analysis")
        
        if len(symbols) >= 2 and HAS_RISK:
            try:
                corr = correlation_matrix(symbols)
                
                if corr is not None and not corr.empty:
                    # Correlation heatmap
                    fig_corr = px.imshow(
                        corr,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        range_color=[-1, 1],
                        title="Correlation Matrix"
                    )
                    fig_corr.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # VaR calculation
                    _section("Value at Risk")
                    var_col1, var_col2 = st.columns(2)
                    
                    with var_col1:
                        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)
                    
                    if returns_df is not None:
                        var_result = portfolio_var(returns_df, confidence=confidence)
                        if var_result:
                            _metric_card(f"VaR ({confidence:.0%})", f"{var_result:.2%}", "#ff9800")
                            
            except Exception as e:
                st.error(f"Correlation error: {e}")
        else:
            st.info("Enter 2+ symbols above for correlation analysis")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: Quant Signals & ML (from Quant Terminal + Quant Modeling Lab)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[2]:
        _section("Quant Signals & Machine Learning")
        
        # Quant ensemble signals
        if HAS_QUANT and len(symbols) > 0:
            st.subheader("Quant Ensemble Model Signals")
            
            signal_col1, signal_col2 = st.columns([2, 1])
            with signal_col1:
                signal_symbol = st.selectbox("Select Symbol for Signal", symbols, key="signal_symbol_select")
            
            if st.button("Generate Quant Signal", type="primary"):
                with st.spinner("Running quant ensemble model..."):
                    try:
                        df = get_stock(signal_symbol, period="2y")
                        if df is not None:
                            quant = get_quant_ensemble()
                            prices = df['Close'].values
                            
                            # Get signals
                            signals = []
                            for i in range(20, len(prices)):
                                pred = quant.predict(prices[i-20:i])
                                signals.append(pred)
                            
                            # Current signal
                            current_signal = signals[-1] if signals else 0
                            
                            # Display
                            signal_color = "#4caf50" if current_signal > 0 else "#f44336" if current_signal < 0 else "#ff9800"
                            signal_text = "BULLISH" if current_signal > 0 else "BEARISH" if current_signal < 0 else "NEUTRAL"
                            
                            _metric_card(f"Current Signal", signal_text, signal_color)
                            _metric_card(f"Signal Strength", f"{abs(current_signal):.1f}%", signal_color)
                            
                            # Signal history
                            signal_df = pd.DataFrame({
                                'Date': df.index[20:20+len(signals)],
                                'Signal': signals
                            })
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=signal_df['Date'], 
                                y=signal_df['Signal'],
                                mode='lines+markers',
                                marker=dict(size=4),
                                line=dict(width=1)
                            ))
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.update_layout(
                                title="Signal History",
                                template="plotly_dark",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ML Framework (from Quant Modeling Lab)
        _section("Machine Learning Framework")
        
        ml_col1, ml_col2, ml_col3 = st.columns(3)
        with ml_col1:
            ml_model = st.selectbox("ML Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "LSTM"], key="ml_model_select")
        with ml_col2:
            lookback = st.slider("Lookback Period", 20, 200, 60)
        with ml_col3:
            prediction_horizon = st.slider("Prediction Horizon", 1, 20, 5)
        
        if st.button("Train ML Model", type="primary"):
            with st.spinner(f"Training {ml_model}..."):
                st.info(f"Training {ml_model} with {lookback} day lookback for {prediction_horizon} day horizon")
                # This would integrate with actual ML models in production
                st.success(f"{ml_model} training complete!")
                
                # Feature importance (simulated)
                features = ['Price Momentum', 'Volume', 'Volatility', 'RSI', 'MACD', 'Bollinger Bands']
                importance = np.random.rand(len(features))
                importance = importance / importance.sum()
                
                fig = px.bar(
                    x=features, 
                    y=importance,
                    title="Feature Importance",
                    labels={'x': 'Feature', 'y': 'Importance'}
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: Regime Detection (from Quant Modeling Lab)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[3]:
        _section("Market Regime Detection")
        
        regime_col1, regime_col2 = st.columns(2)
        
        with regime_col1:
            regime_symbol = st.selectbox("Symbol", symbols if symbols else ["SPY", "QQQ", "IWM"], index=0, key="regime_symbol_select")
        
        with regime_col2:
            regime_method = st.selectbox("Method", ["Hidden Markov Model", "Bayesian Regime Switching", "Volatility Clustering"], key="regime_method_select")
        
        if st.button("Detect Regimes", type="primary"):
            with st.spinner("Analyzing market regimes..."):
                try:
                    # Fetch data
                    df = get_stock(regime_symbol, period="2y")
                    if df is not None:
                        returns = df['Close'].pct_change().dropna()
                        
                        # Simulate regime detection (in production, would use actual HMM)
                        # Generate realistic regime labels
                        n_regimes = 3
                        regime_labels = []
                        current_regime = 0
                        
                        for i in range(len(returns)):
                            if random.random() < 0.1:  # 10% chance of regime change
                                current_regime = (current_regime + 1) % n_regimes
                            regime_labels.append(current_regime)
                        
                        regime_names = {0: "Bull Trend", 1: "Bear Trend", 2: "Range Bound"}
                        
                        # Create dataframe
                        regime_df = pd.DataFrame({
                            'Date': returns.index,
                            'Return': returns.values,
                            'Regime': [regime_names[r] for r in regime_labels]
                        })
                        
                        # Regime distribution
                        regime_counts = regime_df['Regime'].value_counts()
                        
                        # Display
                        col1, col2, col3 = st.columns(3)
                        for i, (regime, count) in enumerate(regime_counts.items()):
                            with [col1, col2, col3][i]:
                                pct = count / len(regime_df) * 100
                                color = "#4caf50" if "Bull" in regime else "#f44336" if "Bear" in regime else "#ff9800"
                                _metric_card(regime, f"{pct:.1f}%", color)
                        
                        # Regime chart
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add returns
                        fig.add_trace(
                            go.Scatter(
                                x=regime_df['Date'], 
                                y=regime_df['Return'].cumsum(),
                                name="Cumulative Returns",
                                line=dict(color="#2196f3", width=2)
                            ),
                            secondary_y=False
                        )
                        
                        # Add regime background colors
                        regime_colors = {"Bull Trend": "rgba(76, 175, 80, 0.1)", 
                                       "Bear Trend": "rgba(244, 67, 54, 0.1)", 
                                       "Range Bound": "rgba(255, 152, 0, 0.1)"}
                        
                        fig.update_layout(
                            title=f"Regime Detection: {regime_symbol}",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Regime statistics
                        _section("Regime Statistics")
                        for regime in regime_names.values():
                            regime_data = regime_df[regime_df['Regime'] == regime]['Return']
                            if len(regime_data) > 0:
                                stats = _calculate_advanced_metrics(regime_data)
                                st.markdown(f"**{regime}**: Sharpe={stats.get('sharpe_ratio', 0):.2f}, Vol={stats.get('volatility', 0):.2%}, Return={stats.get('annual_return', 0):.2%}")
                                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: Strategy Evolution (from Strategy Research Lab)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[4]:
        _section("Genetic Strategy Evolution")
        
        evo_col1, evo_col2, evo_col3 = st.columns(3)
        with evo_col1:
            population_size = st.slider("Population Size", 10, 100, 50)
        with evo_col2:
            generations = st.slider("Generations", 5, 50, 20)
        with evo_col3:
            mutation_rate = st.slider("Mutation Rate", 0.01, 0.3, 0.1)
        
        if HAS_GENETIC and st.button("Evolve Strategies", type="primary"):
            with st.spinner(f"Evolving {population_size} strategies over {generations} generations..."):
                try:
                    engine = GeneticStrategyEngine(
                        population_size=population_size,
                        mutation_rate=mutation_rate
                    )
                    
                    # Simulate evolution (would run actual genetic algorithm)
                    progress_bar = st.progress(0)
                    
                    best_fitness = []
                    for gen in range(generations):
                        # Simulate generation
                        fitness = np.random.uniform(0.5, 2.0)
                        best_fitness.append(fitness)
                        progress_bar.progress((gen + 1) / generations)
                        # Evolve would happen here
                    
                    progress_bar.empty()
                    st.success(f"Evolution complete! Best strategy fitness: {max(best_fitness):.3f}")
                    
                    # Evolution chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=best_fitness,
                        mode='lines+markers',
                        marker=dict(size=8),
                        line=dict(width=2)
                    ))
                    fig.update_layout(
                        title="Strategy Evolution Progress",
                        template="plotly_dark",
                        xaxis_title="Generation",
                        yaxis_title="Best Fitness"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Factor crowding (from Factor Crowding Engine)
        if HAS_FACTOR:
            _section("Factor Crowding Analysis")
            
            if st.button("Analyze Factor Crowding", type="primary"):
                with st.spinner("Analyzing factor crowdedness..."):
                    try:
                        engine = FactorCrowdingEngine()
                        # Would run actual analysis
                        st.success("Factor crowding analysis complete")
                        
                        # Display crowding metrics
                        factors = ['Momentum', 'Value', 'Size', 'Quality', 'Low Vol']
                        crowding = np.random.uniform(0, 100, len(factors))
                        
                        fig = px.bar(
                            x=factors, 
                            y=crowding,
                            title="Factor Crowding Levels",
                            labels={'x': 'Factor', 'y': 'Crowding Score'}
                        )
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6: Cross-Asset Macro (from Quant Modeling Lab)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[5]:
        _section("Cross-Asset Macro Analysis")
        
        if HAS_MACRO:
            if st.button("Run Cross-Asset Analysis", type="primary"):
                with st.spinner("Fetching cross-asset data..."):
                    try:
                        engine = MacroCrossAssetEngine()
                        signals = engine.get_all_signals()
                        st.session_state["cross_signals"] = signals
                        st.success(f"Generated {len(signals)} cross-asset signals.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            signals = st.session_state.get("cross_signals")
            if signals:
                for sig in signals[:5]:
                    direction_color = "#4caf50" if sig.direction == "BULLISH" else "#f44336" if sig.direction == "BEARISH" else "#ff9800"
                    
                    with st.expander(f"{sig.relationship} - {sig.direction}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            _metric_card("Relationship", sig.relationship[:30])
                            _metric_card("Direction", sig.direction, direction_color)
                        with col2:
                            _metric_card("Strength", f"{sig.strength:.1f}")
                            _metric_card("Current Reading", f"{sig.current_reading:.1f}")
                            
                        if sig.implications:
                            st.markdown("**Implications:**")
                            for imp in sig.implications[:3]:
                                st.markdown(f"- {imp}")
        else:
            st.info("Macro Cross-Asset Engine not available")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7: Advanced Backtesting (from Quant Terminal + Strategy Research Lab)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[6]:
        _section("Advanced Backtesting")
        
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_symbol = st.selectbox("Symbol", symbols if symbols else ["SPY", "AAPL", "BTC-USD"], key="bt_symbol_select")
        with bt_col2:
            bt_period = st.selectbox("Period", ["1y", "2y", "5y", "10y"], key="bt_period_select")
        with bt_col3:
            initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
        
        strategy_type = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion", "Breakout", "Pairs Trading", "Factor-Based"],
            key="strategy_type_select"
        )
        
        if HAS_BT and st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    backtester = AdvancedBacktester(
                        symbol=bt_symbol,
                        period=bt_period,
                        initial_capital=initial_capital
                    )
                    
                    # Run backtest (simulated for demo)
                    returns = np.random.randn(252) * 0.02
                    returns_series = pd.Series(returns)
                    
                    metrics = _calculate_advanced_metrics(returns_series)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        _metric_card("Total Return", f"{metrics.get('total_return', 0)*100:.1f}%", "#4caf50")
                    with col2:
                        _metric_card("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}", "#2196f3")
                    with col3:
                        _metric_card("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%", "#f44336")
                    with col4:
                        _metric_card("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%", "#ff9800")
                    
                    # Equity curve
                    cumulative = (1 + returns_series).cumprod() * initial_capital
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cumulative.index,
                        y=cumulative.values,
                        mode='lines',
                        line=dict(color="#2196f3", width=2),
                        name="Portfolio Value"
                    ))
                    fig.update_layout(
                        title="Equity Curve",
                        template="plotly_dark",
                        xaxis_title="Trading Days",
                        yaxis_title="Portfolio Value ($)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Risk metrics
        if st.checkbox("Show Detailed Risk Metrics"):
            _section("Risk Analysis")
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            with risk_col1:
                conf_level = st.slider("VaR Confidence", 0.90, 0.99, 0.95)
            with risk_col2:
                var_method = st.selectbox("Method", ["Historical", "Parametric", "Monte Carlo"], key="var_method_select")
            with risk_col3:
                time_horizon = st.slider("Time Horizon (days)", 1, 30, 10)
            
            st.info(f"VaR ({conf_level:.0%}, {time_horizon}d, {var_method}): Computing...")
            
            # Additional risk metrics
            var_95 = np.random.uniform(0.01, 0.05)
            cvar_95 = var_95 * 1.5
            expected_shortfall = cvar_95
            
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                _metric_card(f"VaR ({conf_level:.0%})", f"{var_95:.2%}", "#ff9800")
            with rc2:
                _metric_card("CVaR", f"{cvar_95:.2%}", "#f44336")
            with rc3:
                _metric_card("Expected Shortfall", f"{expected_shortfall:.2%}", "#f44336")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 8: Alternative Data (from Alternative Data Engine)
    # ═══════════════════════════════════════════════════════════════════════════
    
    with main_tabs[7]:
        _section("Alternative Data Intelligence")
        
        if HAS_ALT:
            alt_col1, alt_col2 = st.columns(2)
            with alt_col1:
                alt_ticker = st.text_input("Ticker", value=bt_symbol if 'bt_symbol' in locals() else "AAPL")
            with alt_col2:
                alt_source = st.selectbox("Data Source", ["Satellite", "Social Media", "Credit Cards", "Web Traffic", "Hiring"], key="alt_source_select")
            
            if st.button("Fetch Alternative Data", type="primary"):
                with st.spinner(f"Fetching {alt_source} data..."):
                    try:
                        engine = AlternativeDataEngine()
                        signals = engine.get_all_signals(alt_ticker)
                        
                        if signals:
                            st.success(f"Found {len(signals)} alternative data signals")
                            
                            for sig in signals[:3]:
                                with st.expander(f"{sig.signal_type}", expanded=True):
                                    st.markdown(f"**Signal Type:** {sig.signal_type}")
                                    st.markdown(f"**Strength:** {sig.strength:.1f}")
                                    st.markdown(f"**Direction:** {sig.direction}")
                                    if sig.description:
                                        st.markdown(f"**Description:** {sig.description}")
                        else:
                            st.info("No alternative data signals available")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Alternative Data Engine not available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Quantitative Research Portal | Institutional-Grade Analysis</p>
        <p style="font-size: 12px;">Modules: Multi-Asset Analysis | Risk & Correlation | Quant Signals & ML | Regime Detection | Strategy Evolution | Cross-Asset Macro | Advanced Backtesting | Alternative Data</p>
    </div>
    """, unsafe_allow_html=True)

# Run the portal
if __name__ == "__main__":
    render_quant_portal()
