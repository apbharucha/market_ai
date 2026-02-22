import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# New Modules
try:
    from hmm_engine import get_regime_detector
    from daily_intelligence import get_daily_engine
    HAS_ADVANCED_INTEL = True
except ImportError:
    HAS_ADVANCED_INTEL = False

def show_dashboard(df, signal=None, prob=None):
    st.subheader("Price Action with Indicators")

    # --- Regimes & Daily Intel (New Section) ---
    if HAS_ADVANCED_INTEL:
        _render_regime_strip(df)

    # Create subplots: 2 rows, 1 column
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))

    # Add EMAs if available
    if 'ema20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], mode='lines', name='EMA 20', line=dict(color='blue')))
    if 'ema50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema50'], mode='lines', name='EMA 50', line=dict(color='red')))

    # Volume bar chart
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3))

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        hovermode='x unified',
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    if signal is not None and prob is not None:
        col1, col2 = st.columns(2)
        col1.metric("Signal", signal)
        col2.metric("Confidence", f"{round(prob * 100, 2)}%")


def show_sector_scanner(df):
    st.subheader("Sector Trend Scanner")
    
    if df.empty:
        st.warning("No sector data available")
        return
    
    # Interactive bar chart
    fig_bar = px.bar(
        df.reset_index(),
        x='Sector',
        y='TrendScore',
        color='TrendScore',
        color_continuous_scale=['red', 'yellow', 'green'],
        title="Sector Performance Ranking",
        labels={'TrendScore': 'Trend Score (%)', 'Sector': 'Sector'},
        text='TrendScore',
        hover_data=['AssetsUsed']
    )
    fig_bar.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Trend Score: %{y:.2f}%<br>Assets Analyzed: %{customdata[0]}<extra></extra>'
    )
    fig_bar.update_layout(
        height=400,
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Trend Score (%)",
        showlegend=False,
        xaxis={'categoryorder': 'total descending'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Breakdown metrics
    st.caption(" **Breakdown:** Higher scores indicate stronger upward momentum. Scores are calculated as average returns across sector constituents.")
    
    # Detailed table with color coding
    df_display = df.copy()
    df_display['Status'] = df_display['TrendScore'].apply(
        lambda x: ' Strong' if x > 2 else '[NOTE] Moderate' if x > -2 else ' Weak'
    )
    df_display['TrendScore'] = df_display['TrendScore'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        df_display[['Sector', 'TrendScore', 'Status', 'AssetsUsed']].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )


def show_fx_radar(df):
    st.subheader("FX Strength Radar")
    
    if df.empty:
        st.warning("No FX data available")
        return
    
    df_reset = df.reset_index()
    
    # Enhanced polar chart with better visualization
    fig_polar = px.bar_polar(
        df_reset,
        r="TrendScore",
        theta="Pair",
        color="TrendScore",
        color_continuous_scale="RdYlGn",
        title="FX Pair Strength Analysis",
        labels={'TrendScore': 'Trend Score (%)'},
        hover_data=['Pair']
    )
    fig_polar.update_traces(
        hovertemplate='<b>%{theta}</b><br>Trend Score: %{r:.2f}%<extra></extra>'
    )
    fig_polar.update_layout(
        height=500,
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(
                title=dict(text="Trend Score (%)", font=dict(size=12)),
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        ),
        showlegend=True
    )
    st.plotly_chart(fig_polar, use_container_width=True)
    
    # Additional breakdown with bar chart
    st.caption(" **Breakdown:** Radial distance shows trend strength. Green = Strong Up, Red = Strong Down")
    
    # Horizontal bar chart for easier comparison
    fig_bar = px.bar(
        df_reset.sort_values('TrendScore', ascending=True),
        x='TrendScore',
        y='Pair',
        orientation='h',
        color='TrendScore',
        color_continuous_scale="RdYlGn",
        title="FX Pair Comparison (Horizontal View)",
        labels={'TrendScore': 'Trend Score (%)', 'Pair': 'Currency Pair'},
        text='TrendScore'
    )
    fig_bar.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Trend Score: %{x:.2f}%<extra></extra>'
    )
    fig_bar.update_layout(
        height=400,
        template="plotly_dark",
        xaxis_title="Trend Score (%)",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Key insights
    col_fx1, col_fx2, col_fx3 = st.columns(3)
    strongest = df_reset.loc[df_reset['TrendScore'].idxmax()]
    weakest = df_reset.loc[df_reset['TrendScore'].idxmin()]
    avg_score = df_reset['TrendScore'].mean()
    
    with col_fx1:
        st.metric("Strongest Pair", strongest['Pair'], f"{strongest['TrendScore']:.2f}%")
    with col_fx2:
        st.metric("Weakest Pair", weakest['Pair'], f"{weakest['TrendScore']:.2f}%")
    with col_fx3:
        st.metric("Average Score", f"{avg_score:.2f}%", "All Pairs")


def show_futures_leaderboard(df):
    st.subheader("Futures Leaderboard")
    
    if df.empty:
        st.warning("No futures data available")
        return
    
    # Interactive bar chart
    fig_bar = px.bar(
        df.reset_index(),
        x='Contract',
        y='TrendScore',
        color='TrendScore',
        color_continuous_scale=['red', 'yellow', 'green'],
        title="Futures Contract Performance Ranking",
        labels={'TrendScore': 'Trend Score (%)', 'Contract': 'Contract'},
        text='TrendScore',
        orientation='v'
    )
    fig_bar.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Trend Score: %{y:.2f}%<extra></extra>'
    )
    fig_bar.update_layout(
        height=400,
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Trend Score (%)",
        showlegend=False,
        xaxis={'categoryorder': 'total descending'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Breakdown by asset class
    st.caption(" **Breakdown:** Performance ranking of major futures contracts. Higher scores indicate stronger momentum.")
    
    # Categorize contracts
    df_display = df.copy()
    df_display['Category'] = df_display['Contract'].apply(
        lambda x: 'Equity Index' if any(y in x for y in ['ES', 'NQ']) 
        else 'Energy' if 'CL' in x 
        else 'Metals' if any(y in x for y in ['GC', 'SI']) 
        else 'Fixed Income' if 'ZB' in x 
        else 'Other'
    )
    
    # Grouped visualization
    fig_grouped = px.bar(
        df_display.reset_index(),
        x='Category',
        y='TrendScore',
        color='Contract',
        title="Futures Performance by Asset Class",
        labels={'TrendScore': 'Trend Score (%)', 'Category': 'Asset Class'},
        barmode='group'
    )
    fig_grouped.update_layout(
        height=400,
        template="plotly_dark",
        xaxis_title="Asset Class",
        yaxis_title="Trend Score (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_grouped, use_container_width=True)
    
    # Detailed table
    df_display['Status'] = df_display['TrendScore'].apply(
        lambda x: ' Strong' if x > 2 else '[NOTE] Moderate' if x > -2 else ' Weak'
    )
    df_display['TrendScore'] = df_display['TrendScore'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        df_display[['Contract', 'Category', 'TrendScore', 'Status']].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )


def _render_regime_strip(df):
    """Render a visual strip showing the current market regime from HMM."""
    if df is None or len(df) < 50:
        return
        
    try:
        from hmm_engine import get_regime_detector
        detector = get_regime_detector()
        regime = detector.predict_regime(df)
        
        label = regime.get('regime', 'Unknown')
        conf = regime.get('confidence', 0)
        desc = regime.get('desc', '')
        
        # Color coding
        color = "gray"
        if "Bull" in label: color = "green"
        elif "Bear" in label: color = "red"
        elif "Volatile" in label: color = "orange"
        
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: rgba(255,255,255,0.05); border-left: 5px solid {color}; margin-bottom: 20px;">
            <h4 style="margin:0; color: {color};"> Market Regime: {label} <span style="font-size:0.8em; color:gray;">(Confidence: {conf:.0%})</span></h4>
            <p style="margin:5px 0 0 0; font-size:0.9em; opacity: 0.8;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"HMM Error: {e}")
