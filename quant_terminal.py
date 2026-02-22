import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from risk_engine import correlation_matrix, portfolio_var
from data_sources import get_stock

try:
    from quant_ensemble_model import get_quant_ensemble
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False

try:
    from advanced_backtester import AdvancedBacktester
    HAS_BT = True
except ImportError:
    HAS_BT = False


def _run_quant_ensemble_backtest(df: pd.DataFrame, symbol: str, initial_capital: float = 100000) -> dict:
    """Run a backtest driven EXCLUSIVELY by the quant ensemble model signals.
    Returns a dict with all backtest metrics, or None on failure.
    """
    if not HAS_QUANT:
        return None

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)

    if len(close) < 60:
        return None

    volumes = None
    if "Volume" in df.columns:
        v = df["Volume"]
        if isinstance(v, pd.DataFrame):
            v = v.iloc[:, 0]
        v = v.dropna().astype(float)
        if len(v) > 0 and v.sum() > 0 and v.max() > 1:
            volumes = v.values

    quant = get_quant_ensemble()
    prices = close.values
    dates = close.index

    # Walk-forward: at each bar from warmup onward, predict using data up to that bar
    warmup = 50
    if len(prices) <= warmup + 10:
        return None

    # Generate signals for each bar
    signals = []  # list of (date_idx, direction, probability, confidence)
    for i in range(warmup, len(prices)):
        try:
            window = prices[:i + 1]
            vol_window = volumes[:i + 1] if volumes is not None and len(volumes) >= i + 1 else None
            sig = quant.predict(window, vol_window)
            signals.append({
                "idx": i,
                "direction": sig.direction,
                "probability": sig.probability,
                "confidence": sig.confidence,
                "expected_return": sig.expected_return,
                "stop_loss_pct": sig.stop_loss_pct,
                "take_profit_pct": sig.take_profit_pct,
            })
        except Exception:
            signals.append({
                "idx": i,
                "direction": "NEUTRAL",
                "probability": 0.5,
                "confidence": 0.0,
                "expected_return": 0.0,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            })

    # Simulate trades based on quant signals
    cash = initial_capital
    position = 0  # number of shares (positive = long, negative = short)
    entry_price = 0.0
    entry_idx = 0
    entry_confidence = 0.0
    trades = []
    equity_curve = [initial_capital]
    position_state = "FLAT"  # FLAT, LONG, SHORT

    min_confidence = 0.25  # minimum confidence to enter
    min_probability = 0.52  # minimum directional probability

    for s in signals:
        i = s["idx"]
        price = float(prices[i])
        direction = s["direction"]
        prob = s["probability"]
        conf = s["confidence"]
        sl_pct = s["stop_loss_pct"]
        tp_pct = s["take_profit_pct"]

        # Check exits first if in a position
        if position_state == "LONG" and position > 0:
            pnl_pct = (price - entry_price) / entry_price
            # Exit conditions: stop loss, take profit, or signal reversal
            should_exit = False
            exit_reason = ""
            if pnl_pct <= -sl_pct:
                should_exit = True
                exit_reason = f"Stop loss ({pnl_pct:+.2%})"
            elif pnl_pct >= tp_pct:
                should_exit = True
                exit_reason = f"Take profit ({pnl_pct:+.2%})"
            elif direction == "BEARISH" and conf >= min_confidence:
                should_exit = True
                exit_reason = f"Signal reversal to BEARISH (conf={conf:.0%})"
            elif direction == "NEUTRAL" and conf >= 0.4 and pnl_pct > 0:
                should_exit = True
                exit_reason = f"Signal neutral, locking profit ({pnl_pct:+.2%})"

            if should_exit:
                pnl_dollar = position * (price - entry_price)
                cash += position * price
                trades.append({
                    "entry_time": str(dates[entry_idx])[:10],
                    "exit_time": str(dates[i])[:10],
                    "side": "LONG",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_dollar,
                    "confidence": entry_confidence,
                    "exit_reason": exit_reason,
                    "hold_bars": i - entry_idx,
                })
                position = 0
                position_state = "FLAT"

        elif position_state == "SHORT" and position < 0:
            pnl_pct = (entry_price - price) / entry_price
            should_exit = False
            exit_reason = ""
            if pnl_pct <= -sl_pct:
                should_exit = True
                exit_reason = f"Stop loss ({pnl_pct:+.2%})"
            elif pnl_pct >= tp_pct:
                should_exit = True
                exit_reason = f"Take profit ({pnl_pct:+.2%})"
            elif direction == "BULLISH" and conf >= min_confidence:
                should_exit = True
                exit_reason = f"Signal reversal to BULLISH (conf={conf:.0%})"
            elif direction == "NEUTRAL" and conf >= 0.4 and pnl_pct > 0:
                should_exit = True
                exit_reason = f"Signal neutral, locking profit ({pnl_pct:+.2%})"

            if should_exit:
                pnl_dollar = abs(position) * (entry_price - price)
                cash += abs(position) * (2 * entry_price - price)  # return collateral + pnl
                position = 0
                position_state = "FLAT"
                trades.append({
                    "entry_time": str(dates[entry_idx])[:10],
                    "exit_time": str(dates[i])[:10],
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_dollar,
                    "confidence": entry_confidence,
                    "exit_reason": exit_reason,
                    "hold_bars": i - entry_idx,
                })

        # Check entries if flat
        if position_state == "FLAT":
            if direction == "BULLISH" and conf >= min_confidence and prob >= min_probability:
                # Size position based on confidence (5-15% of capital)
                alloc_pct = 0.05 + (conf - min_confidence) * 0.2
                alloc_pct = min(alloc_pct, 0.15)
                alloc_dollar = cash * alloc_pct
                shares = int(alloc_dollar / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    entry_idx = i
                    entry_confidence = conf
                    cash -= shares * price
                    position_state = "LONG"

            elif direction == "BEARISH" and conf >= min_confidence and (1 - prob) >= min_probability:
                alloc_pct = 0.05 + (conf - min_confidence) * 0.2
                alloc_pct = min(alloc_pct, 0.12)
                alloc_dollar = cash * alloc_pct
                shares = int(alloc_dollar / price)
                if shares > 0:
                    position = -shares
                    entry_price = price
                    entry_idx = i
                    entry_confidence = conf
                    cash -= shares * price  # collateral
                    position_state = "SHORT"

        # Calculate equity
        if position_state == "LONG":
            equity = cash + position * price
        elif position_state == "SHORT":
            equity = cash + abs(position) * (2 * entry_price - price)
        else:
            equity = cash
        equity_curve.append(max(equity, 0))

    # Close any open position at end
    if position_state == "LONG" and position > 0:
        final_price = float(prices[-1])
        pnl_pct = (final_price - entry_price) / entry_price
        pnl_dollar = position * (final_price - entry_price)
        cash += position * final_price
        trades.append({
            "entry_time": str(dates[entry_idx])[:10],
            "exit_time": str(dates[-1])[:10],
            "side": "LONG",
            "entry_price": entry_price,
            "exit_price": final_price,
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
            "confidence": entry_confidence,
            "exit_reason": "End of period",
            "hold_bars": len(prices) - 1 - entry_idx,
        })
    elif position_state == "SHORT" and position < 0:
        final_price = float(prices[-1])
        pnl_pct = (entry_price - final_price) / entry_price
        pnl_dollar = abs(position) * (entry_price - final_price)
        trades.append({
            "entry_time": str(dates[entry_idx])[:10],
            "exit_time": str(dates[-1])[:10],
            "side": "SHORT",
            "entry_price": entry_price,
            "exit_price": final_price,
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
            "confidence": entry_confidence,
            "exit_reason": "End of period",
            "hold_bars": len(prices) - 1 - entry_idx,
        })

    # Compute metrics
    equity_arr = np.array(equity_curve)
    total_return_pct = (equity_arr[-1] / initial_capital) - 1
    total_return_dollar = equity_arr[-1] - initial_capital

    # Daily returns from equity curve
    eq_returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-10)

    sharpe = 0.0
    sortino = 0.0
    if len(eq_returns) > 20:
        mean_r = np.mean(eq_returns)
        std_r = np.std(eq_returns)
        if std_r > 0:
            sharpe = (mean_r / std_r) * np.sqrt(252)
        downside = eq_returns[eq_returns < 0]
        if len(downside) > 0:
            down_std = np.std(downside)
            if down_std > 0:
                sortino = (mean_r / down_std) * np.sqrt(252)

    # Max drawdown
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / (running_max + 1e-10)
    max_drawdown_pct = float(np.min(drawdown))
    drawdown_curve = drawdown.tolist()

    # Trade statistics
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    avg_win_pct = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_win_dollar = np.mean([t["pnl_dollar"] for t in wins]) if wins else 0
    avg_loss_dollar = np.mean([t["pnl_dollar"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl_dollar"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_dollar"] for t in losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    largest_win_pct = max((t["pnl_pct"] for t in trades), default=0)
    largest_loss_pct = min((t["pnl_pct"] for t in trades), default=0)

    avg_hold = np.mean([t["hold_bars"] for t in trades]) if trades else 0

    # Benchmark (buy & hold)
    benchmark_return = (float(prices[-1]) / float(prices[warmup]) - 1) if prices[warmup] > 0 else 0
    alpha = total_return_pct - benchmark_return

    # Annualized return
    n_bars = len(prices) - warmup
    ann_factor = 252 / max(n_bars, 1)
    annualized_return = (1 + total_return_pct) ** ann_factor - 1 if total_return_pct > -1 else -1

    calmar = annualized_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    cur_wins = 0
    cur_losses = 0
    for t in trades:
        if t["pnl_pct"] > 0:
            cur_wins += 1
            cur_losses = 0
            max_consec_wins = max(max_consec_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_consec_losses = max(max_consec_losses, cur_losses)

    # Expectancy
    expectancy = (win_rate * avg_win_dollar + (1 - win_rate) * avg_loss_dollar) if total_trades > 0 else 0

    # Rolling sharpe
    rolling_sharpe = []
    if len(eq_returns) >= 30:
        for j in range(30, len(eq_returns)):
            window = eq_returns[j - 30:j]
            s = np.std(window)
            if s > 0:
                rolling_sharpe.append(float((np.mean(window) / s) * np.sqrt(252)))
            else:
                rolling_sharpe.append(0.0)

    # Monthly returns
    monthly_returns = {}
    if len(equity_arr) > 20 and len(dates) > warmup:
        eq_series = pd.Series(equity_arr[1:], index=dates[warmup:])
        monthly = eq_series.resample('M').last()
        for i_m in range(1, len(monthly)):
            month_key = monthly.index[i_m].strftime('%Y-%m')
            monthly_returns[month_key] = (monthly.iloc[i_m] / monthly.iloc[i_m - 1]) - 1

    # Max DD duration
    dd_start = 0
    max_dd_duration = 0
    for j in range(1, len(equity_arr)):
        if equity_arr[j] >= running_max[j]:
            dd_dur = j - dd_start
            max_dd_duration = max(max_dd_duration, dd_dur)
            dd_start = j
    max_dd_duration = max(max_dd_duration, len(equity_arr) - 1 - dd_start)

    return {
        "total_return_pct": total_return_pct,
        "total_return_dollar": total_return_dollar,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "alpha": alpha,
        "annualized_return": annualized_return,
        "calmar_ratio": calmar,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "avg_win_dollar": avg_win_dollar,
        "avg_loss_dollar": avg_loss_dollar,
        "largest_win_pct": largest_win_pct,
        "largest_loss_pct": largest_loss_pct,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "avg_hold_bars": avg_hold,
        "expectancy": expectancy,
        "benchmark_return": benchmark_return,
        "max_drawdown_duration_bars": max_dd_duration,
        "beta": 1.0,
        "kelly_fraction": (win_rate - (1 - win_rate) / (abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 1)) if total_trades > 0 else 0,
        "information_ratio": (total_return_pct - benchmark_return) / (np.std(eq_returns) * np.sqrt(252)) if len(eq_returns) > 0 and np.std(eq_returns) > 0 else 0,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "rolling_sharpe": rolling_sharpe,
        "monthly_returns": monthly_returns,
        "trades": trades,
        "initial_capital": initial_capital,
        "model_source": "Quant Ensemble Only",
    }


def show_quant_terminal():
    st.header("Quant Terminal -- Professional Multi-Asset Analysis")
    st.caption("Correlation, risk, quant signals, and backtesting for any asset combination")

    # ═══ Symbol Input ═══
    symbol_input = st.text_input(
        "Enter Symbols (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="Stocks (AAPL), futures (ES=F), FX (EURUSD=X), crypto (BTC-USD)"
    )
    
    if symbol_input:
        symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]
    else:
        symbols = []
    
    st.caption("Quick Select:")
    col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
    with col_quick1:
        if st.button("Stocks", use_container_width=True, key="quick_select_stocks"):
            st.session_state['quant_symbols'] = "AAPL, MSFT, NVDA, GOOGL"
            st.rerun()
    with col_quick2:
        if st.button("Futures", use_container_width=True, key="quick_select_futures"):
            st.session_state['quant_symbols'] = "ES=F, NQ=F, CL=F, GC=F"
            st.rerun()
    with col_quick3:
        if st.button("FX", use_container_width=True, key="quick_select_fx"):
            st.session_state['quant_symbols'] = "EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X"
            st.rerun()
    with col_quick4:
        if st.button("Crypto", use_container_width=True, key="quick_select_crypto"):
            st.session_state['quant_symbols'] = "BTC-USD, ETH-USD, SOL-USD"
            st.rerun()
    
    if 'quant_symbols' in st.session_state:
        if not symbol_input or symbol_input == "AAPL, MSFT, NVDA":
            symbol_input = st.session_state['quant_symbols']
            symbols = [s.strip().upper() for s in symbol_input.split(',') if s.strip()]

    if len(symbols) < 1:
        st.info("Enter at least 1 symbol to begin. For correlation analysis, enter 2+.")
        return

    # ═══ Tabs ═══
    qt_tabs = st.tabs(["Correlation & Risk", "Quant Signals", "Returns Analysis", "Backtest"])

    # ─── Tab 1: Correlation & Risk ───
    with qt_tabs[0]:
        if len(symbols) >= 2:
            try:
                corr = correlation_matrix(symbols)
            except Exception as e:
                st.error(f"Correlation error: {e}")
                corr = None
            
            if corr is not None and not corr.empty:
                st.subheader("Interactive Correlation Matrix")
                fig_corr = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu", color_continuous_midpoint=0,
                    title="Asset Correlation Heatmap"
                )
                fig_corr.update_layout(height=500, template="plotly_dark",
                                       margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_corr, use_container_width=True)
                
                corr_no_diag = corr.copy()
                for i in range(len(corr_no_diag)):
                    corr_no_diag.iloc[i, i] = np.nan
                
                col_corr1, col_corr2 = st.columns(2)
                with col_corr1:
                    max_corr = corr_no_diag.max().max()
                    if not pd.isna(max_corr):
                        st.metric("Strongest Positive", f"{max_corr:.3f}")
                with col_corr2:
                    min_corr = corr_no_diag.min().min()
                    if not pd.isna(min_corr):
                        st.metric("Strongest Negative", f"{min_corr:.3f}")

                st.subheader("Portfolio Risk Analysis")
                weights_input = st.text_input(
                    "Weights (sum to 1.0)",
                    value=",".join([str(round(1/len(symbols), 2)) for _ in symbols])
                )
                try:
                    custom_weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
                    if len(custom_weights) != len(symbols):
                        custom_weights = [1 / len(symbols)] * len(symbols)
                except Exception:
                    custom_weights = [1 / len(symbols)] * len(symbols)

                var, vol = portfolio_var(symbols, custom_weights)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("VaR (95%)", f"{abs(var):.2f}%")
                with c2:
                    st.metric("Portfolio Volatility", f"{vol:.2f}%")
                with c3:
                    st.metric("Risk-Adj Return", f"{abs(var/vol):.2f}" if vol > 0 else "N/A")
        else:
            st.info("Enter 2+ symbols for correlation and portfolio risk analysis.")

    # ─── Tab 2: Quant Ensemble Signals ───
    with qt_tabs[1]:
        st.subheader("Quant Ensemble Model Signals")
        
        if not HAS_QUANT:
            st.warning("Quant Ensemble Model not available.")
        else:
            quant = get_quant_ensemble()
            
            weights = quant.get_model_weights()
            accuracies = quant.get_model_accuracies()
            
            st.markdown("### Model Weights & Accuracy")
            weight_df = pd.DataFrame([
                {"Model": name.replace("_", " ").title(), 
                 "Weight": f"{w:.1%}", 
                 "Accuracy": f"{accuracies.get(name, 0.5):.1%}"}
                for name, w in weights.items()
            ])
            st.dataframe(weight_df, use_container_width=True, hide_index=True)
            
            st.markdown("### Per-Symbol Signals")
            for sym in symbols:
                with st.expander(f"{sym}", expanded=len(symbols) <= 3):
                    try:
                        df = get_stock(sym, period="6mo")
                        if df is None or df.empty:
                            st.warning(f"No data for {sym}")
                            continue
                        close = df["Close"]
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        prices = close.dropna().astype(float).values
                        
                        volumes = None
                        if "Volume" in df.columns:
                            v = df["Volume"]
                            if isinstance(v, pd.DataFrame):
                                v = v.iloc[:, 0]
                            v = v.dropna().astype(float)
                            if len(v) > 0 and v.sum() > 0 and v.max() > 1:
                                volumes = v.values
                        
                        if len(prices) < 40:
                            st.warning(f"Insufficient data ({len(prices)} bars)")
                            continue
                        
                        signal = quant.predict(prices, volumes)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Direction", signal.direction)
                        with c2:
                            st.metric("Probability", f"{signal.probability:.1%}")
                        with c3:
                            st.metric("Confidence", f"{signal.confidence:.1%}")
                        with c4:
                            st.metric("Expected Return", f"{signal.expected_return:+.2%}")
                        
                        c5, c6, c7, c8 = st.columns(4)
                        with c5:
                            st.metric("Sharpe Est.", f"{signal.sharpe_estimate:.2f}")
                        with c6:
                            st.metric("Kelly Size", f"{signal.optimal_position_size:.1%}")
                        with c7:
                            st.metric("Stop Loss", f"{signal.stop_loss_pct:.2%}")
                        with c8:
                            st.metric("Take Profit", f"{signal.take_profit_pct:.2%}")
                        
                        st.markdown("**Sub-Model Breakdown:**")
                        sub_data = []
                        for name, data in signal.sub_model_signals.items():
                            prob = data["probability"]
                            sub_data.append({
                                "Model": name.replace("_", " ").title(),
                                "Signal": "Bullish" if prob > 0.55 else "Bearish" if prob < 0.45 else "Neutral",
                                "Prob": f"{prob:.1%}",
                                "Weight": f"{data['weight']:.1%}",
                                "Contribution": f"{data['weighted_contribution']:.3f}",
                            })
                        st.dataframe(pd.DataFrame(sub_data), use_container_width=True, hide_index=True)
                        
                        st.markdown("**Risk Metrics:**")
                        risk_cols = st.columns(4)
                        rm = signal.risk_metrics
                        with risk_cols[0]:
                            st.metric("Volatility", f"{rm.get('volatility', 0):.1%}")
                        with risk_cols[1]:
                            st.metric("VaR 95%", f"{rm.get('var_95', 0):.2%}")
                        with risk_cols[2]:
                            st.metric("Max DD", f"{rm.get('max_drawdown', 0):.1%}")
                        with risk_cols[3]:
                            st.metric("Tail Ratio", f"{rm.get('tail_ratio', 1):.2f}")
                        
                        if signal.reasoning:
                            st.markdown("**Reasoning:**")
                            for r in signal.reasoning[:8]:
                                st.caption(r)
                    except Exception as e:
                        st.error(f"Error analyzing {sym}: {e}")

    # ─── Tab 3: Returns Analysis ───
    with qt_tabs[2]:
        st.subheader("Asset Returns Analysis")
        returns_data = {}
        prices_data = {}
        for sym in symbols:
            df = get_stock(sym)
            if df is not None and not df.empty:
                close = df['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                returns_data[sym] = close.pct_change().dropna()
                prices_data[sym] = close
        
        if returns_data:
            fig_returns = go.Figure()
            for sym, rets in returns_data.items():
                cum_rets = (1 + rets).cumprod() - 1
                fig_returns.add_trace(go.Scatter(
                    x=cum_rets.index, y=cum_rets.values * 100,
                    mode='lines', name=sym
                ))
            fig_returns.update_layout(
                title=dict(text="Cumulative Returns Comparison", x=0.01, font=dict(size=13)),
                height=380, template="plotly_dark", hovermode='x unified',
                yaxis_title="Cumulative Return (%)",
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            summary_data = []
            for sym, rets in returns_data.items():
                if len(rets) == 0: continue
                cum_rets = (1 + rets).cumprod()
                running_max = cum_rets.expanding().max()
                drawdown = (cum_rets - running_max) / running_max
                summary_data.append({
                    'Asset': sym,
                    'Avg Daily (%)': f"{rets.mean() * 100:.3f}",
                    'Vol (%)': f"{rets.std() * 100:.3f}",
                    'Sharpe': f"{(rets.mean() / rets.std()) * np.sqrt(252):.2f}" if rets.std() > 0 else "0",
                    'Max DD (%)': f"{drawdown.min() * 100:.2f}",
                    'Skew': f"{rets.skew():.2f}",
                    'Kurtosis': f"{rets.kurtosis():.2f}",
                })
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ─── Tab 4: Backtest (Quant Ensemble Only) ───
    with qt_tabs[3]:
        st.subheader("Quant Ensemble Backtest")

        # Disclosure box
        st.info(
            "**Model Disclosure:** This backtest is driven **exclusively by the Quant Ensemble Model** — "
            "a multi-model system combining momentum, mean-reversion, volatility, statistical, and ML sub-models. "
            "Signals are generated via walk-forward prediction at each bar. "
            "No other model stack (unbiased analyzer, ML engine, etc.) is used here.\n\n"
            "For a full-stack backtest using all available models, use the **Simulation Hub** instead."
        )
        
        if not HAS_QUANT:
            st.warning("Quant Ensemble Model not available. Cannot run quant-specific backtest.")
        else:
            bt_sym = st.selectbox("Select Symbol to Backtest", symbols, key="qt_bt_sym")
            bt_col1, bt_col2 = st.columns(2)
            with bt_col1:
                bt_period = st.selectbox("Historical Period", ["3mo", "6mo", "1y", "2y", "5y"],
                                          index=2, key="qt_bt_period")
            with bt_col2:
                bt_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000, 
                                              step=10000, key="qt_bt_capital")
            
            if st.button("Run Quant Ensemble Backtest", key="qt_run_bt", type="primary"):
                with st.spinner(f"Backtesting {bt_sym} using Quant Ensemble over {bt_period}..."):
                    try:
                        df = get_stock(bt_sym, period=bt_period)
                        if df is None or df.empty:
                            st.error(f"Could not fetch data for {bt_sym}.")
                        else:
                            close_check = df["Close"]
                            if isinstance(close_check, pd.DataFrame):
                                close_check = close_check.iloc[:, 0]
                            close_check = close_check.dropna()
                            
                            st.caption(f"Data: {len(close_check)} bars | "
                                      f"Range: ${float(close_check.min()):.2f} — ${float(close_check.max()):.2f} | "
                                      f"Period: {close_check.index[0].strftime('%Y-%m-%d')} to {close_check.index[-1].strftime('%Y-%m-%d')}")
                            
                            if len(close_check) < 60:
                                st.error(f"Only {len(close_check)} bars — need at least 60 for quant ensemble. Try a longer period.")
                            else:
                                result = _run_quant_ensemble_backtest(df, bt_sym, bt_capital)
                                
                                if result is None:
                                    st.error("Backtest failed. Ensure the quant ensemble model is properly configured.")
                                elif result["total_trades"] == 0:
                                    st.warning("No trades generated. The quant model did not produce signals with "
                                              "sufficient confidence in this period. Try a different symbol or longer period.")
                                else:
                                    st.success(f"Backtest complete: {result['total_trades']} trades | Model: {result['model_source']}")
                                    
                                    c1, c2, c3, c4 = st.columns(4)
                                    with c1:
                                        color = "normal" if result["total_return_pct"] >= 0 else "inverse"
                                        st.metric("Total Return", f"{result['total_return_pct']:+.2%}",
                                                  delta=f"${result['total_return_dollar']:+,.2f}")
                                    with c2:
                                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                                    with c3:
                                        st.metric("Win Rate", f"{result['win_rate']:.0%}")
                                    with c4:
                                        st.metric("Total Trades", result["total_trades"])
                                    
                                    c5, c6, c7, c8 = st.columns(4)
                                    with c5:
                                        st.metric("Max Drawdown", f"{result['max_drawdown_pct']:.2%}")
                                    with c6:
                                        pf = result["profit_factor"]
                                        st.metric("Profit Factor", f"{pf:.2f}" if pf < 99 else "Inf")
                                    with c7:
                                        st.metric("Sortino", f"{result['sortino_ratio']:.2f}")
                                    with c8:
                                        st.metric("Alpha vs B&H", f"{result['alpha']:+.2%}")
                                    
                                    # Equity curve
                                    bt_tabs = st.tabs(["Equity Curve", "Drawdown", "Analytics", "Trade Log"])
                                    
                                    with bt_tabs[0]:
                                        fig_eq = go.Figure()
                                        fig_eq.add_trace(go.Scatter(
                                            x=list(range(len(result["equity_curve"]))),
                                            y=result["equity_curve"],
                                            mode='lines', name='Quant Ensemble Strategy',
                                            line=dict(color='#00d4aa', width=2)
                                        ))
                                        fig_eq.add_hline(y=bt_capital, line_dash="dash", line_color="gray",
                                                         annotation_text=f"Start ${bt_capital:,.0f}")
                                        fig_eq.update_layout(
                                            title=dict(text=f"{bt_sym} — Quant Ensemble Equity Curve", x=0.01, font=dict(size=13)),
                                            height=380, template="plotly_dark", yaxis_title="Portfolio Value ($)",
                                            margin=dict(l=10, r=10, t=35, b=10),
                                        )
                                        st.plotly_chart(fig_eq, use_container_width=True)
                                    
                                    with bt_tabs[1]:
                                        fig_dd = go.Figure()
                                        fig_dd.add_trace(go.Scatter(
                                            x=list(range(len(result["drawdown_curve"]))),
                                            y=[d * 100 for d in result["drawdown_curve"]],
                                            mode='lines', name='Drawdown', fill='tozeroy',
                                            fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=1)
                                        ))
                                        fig_dd.update_layout(
                                            title=dict(text="Drawdown (%)", x=0.01, font=dict(size=13)),
                                            height=280, template="plotly_dark", yaxis_title="Drawdown %",
                                            margin=dict(l=10, r=10, t=35, b=10),
                                        )
                                        st.plotly_chart(fig_dd, use_container_width=True)
                                        
                                        if result["rolling_sharpe"]:
                                            fig_rs = go.Figure()
                                            fig_rs.add_trace(go.Scatter(
                                                x=list(range(len(result["rolling_sharpe"]))),
                                                y=result["rolling_sharpe"],
                                                mode='lines', name='Rolling Sharpe (30-bar)',
                                                line=dict(color='#42a5f5', width=1.5)
                                            ))
                                            fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
                                            fig_rs.update_layout(
                                                title=dict(text="Rolling Sharpe Ratio (30-bar)", x=0.01, font=dict(size=13)),
                                                height=240, template="plotly_dark",
                                                margin=dict(l=10, r=10, t=35, b=10),
                                            )
                                            st.plotly_chart(fig_rs, use_container_width=True)
                                    
                                    with bt_tabs[2]:
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.markdown("### Win/Loss Distribution")
                                            pnls = [t["pnl_pct"] * 100 for t in result["trades"]]
                                            if pnls:
                                                fig_hist = px.histogram(x=pnls, nbins=max(10, len(pnls) // 3),
                                                                        title="Trade P&L Distribution (%)",
                                                                        color_discrete_sequence=['#636EFA'])
                                                fig_hist.update_layout(template="plotly_dark", height=280,
                                                                       margin=dict(l=10, r=10, t=35, b=10))
                                                st.plotly_chart(fig_hist, use_container_width=True)
                                        
                                        with col_b:
                                            st.markdown("### Monthly Returns")
                                            if result["monthly_returns"]:
                                                months = list(result["monthly_returns"].keys())
                                                rets = [v * 100 for v in result["monthly_returns"].values()]
                                                colors = ['green' if r > 0 else 'red' for r in rets]
                                                fig_monthly = go.Figure(go.Bar(x=months, y=rets, marker_color=colors))
                                                fig_monthly.update_layout(title=dict(text="Monthly Returns (%)", x=0.01, font=dict(size=13)),
                                                                          height=280, template="plotly_dark",
                                                                          margin=dict(l=10, r=10, t=35, b=10))
                                                st.plotly_chart(fig_monthly, use_container_width=True)
                                            else:
                                                st.caption("Insufficient data for monthly breakdown.")
                                        
                                        st.markdown("### Detailed Statistics")
                                        stats = {
                                            "Annualized Return": f"{result['annualized_return']:+.2%}",
                                            "Calmar Ratio": f"{result['calmar_ratio']:.2f}",
                                            "Avg Win": f"{result['avg_win_pct']:+.2%} (${result['avg_win_dollar']:+,.2f})",
                                            "Avg Loss": f"{result['avg_loss_pct']:+.2%} (${result['avg_loss_dollar']:+,.2f})",
                                            "Largest Win": f"{result['largest_win_pct']:+.2%}",
                                            "Largest Loss": f"{result['largest_loss_pct']:+.2%}",
                                            "Max Consec. Wins": result["max_consecutive_wins"],
                                            "Max Consec. Losses": result["max_consecutive_losses"],
                                            "Avg Hold (bars)": f"{result['avg_hold_bars']:.1f}",
                                            "Expectancy ($/trade)": f"${result['expectancy']:+,.2f}",
                                            "Information Ratio": f"{result['information_ratio']:.2f}",
                                            "Benchmark Return": f"{result['benchmark_return']:+.2%}",
                                            "Kelly Fraction": f"{result['kelly_fraction']:.1%}",
                                            "Max DD Duration (bars)": result["max_drawdown_duration_bars"],
                                            "Model Source": result["model_source"],
                                        }
                                        st.dataframe(pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]),
                                                     use_container_width=True, hide_index=True)
                                    
                                    with bt_tabs[3]:
                                        if result["trades"]:
                                            trade_rows = []
                                            for t in result["trades"]:
                                                trade_rows.append({
                                                    "Entry": t["entry_time"],
                                                    "Exit": t["exit_time"],
                                                    "Side": t["side"],
                                                    "Entry$": f"${t['entry_price']:,.2f}",
                                                    "Exit$": f"${t['exit_price']:,.2f}",
                                                    "P&L%": f"{t['pnl_pct']:+.2%}",
                                                    "P&L$": f"${t['pnl_dollar']:+,.2f}",
                                                    "Conf": f"{t['confidence']:.0%}",
                                                    "Reason": t["exit_reason"],
                                                    "Hold": f"{t['hold_bars']}d",
                                                })
                                            st.dataframe(pd.DataFrame(trade_rows),
                                                         use_container_width=True, hide_index=True, height=400)
                                        else:
                                            st.info("No trades to display.")
                    except Exception as e:
                        st.error(f"Backtest error: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")
