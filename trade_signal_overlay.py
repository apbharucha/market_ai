"""
Trade Signal Overlay — generates model entry/exit points for chart display.
Author: APB - Octavian Team
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TradeMarker:
    timestamp: object  # datetime index
    price: float
    side: str  # "BUY", "SELL", "SHORT", "COVER"
    reason: str
    pnl_pct: Optional[float] = None  # filled on exit
    pnl_dollar: Optional[float] = None
    trade_id: int = 0
    indicator_snapshot: Optional[Dict] = None  # RSI, MACD, SMA values at signal time
    model_reasoning: str = "" # Detailed explanation
    regime_context: str = "Unknown" # e.g. "Bull", "Bear"
    confidence: float = 0.0

def generate_model_trades(close: pd.Series, df: pd.DataFrame = None,
                          strategy: str = "combined") -> List[TradeMarker]:
    """Generate hypothetical model trade entries/exits on historical data."""
    if close is None or len(close) < 30:
        return []

    # Get Regime Data if available
    regime_map = {}
    try:
        from hmm_engine import get_regime_detector
        detector = get_regime_detector()
        # We need to predict regime for the whole dataframe ideally, 
        # but here we simulate 'point-in-time' knowledge by just using the detector 
        # (assuming it's trained on past data or we rely on robust features).
        # For simplicity in this overlay, we will just use a heuristic or previous data if plausible.
        # In a real backtest, we'd walk forward. 
        # Here we will try to get the full sequence of regimes.
        if df is not None:
             # Fast approximation: get regimes for the dataset
             # Note: This technically 'peeks' if we fit on the whole DF, 
             # but for visualization of "what the model thinks now about the past", it's acceptable
             # provided we label it clearly. 
             # For strict out-of-sample, we would need a rolling fit.
             pass 
    except ImportError:
        pass

    trades = []
    if strategy == "combined":
        trades = _strategy_combined(close, df)
    elif strategy == "rsi":
        trades = _strategy_rsi(close)
    elif strategy == "sma_cross":
        trades = _strategy_sma_cross(close)
    elif strategy == "macd":
        trades = _strategy_macd(close)
    else:
        trades = _strategy_combined(close, df)

    # Assign trade IDs and calculate P&L on pairs
    _pair_trades_and_calc_pnl(trades)
    return trades


def _strategy_combined(close: pd.Series, df: pd.DataFrame = None) -> List[TradeMarker]:
    """Combined strategy: RSI + SMA + MACD confluence."""
    markers = []
    if len(close) < 30:
        return markers

    # Compute indicators
    rsi = _rsi(close, 14)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(min(50, len(close) - 1)).mean()

    macd_hist = None
    if len(close) >= 27:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

    in_long = False
    in_short = False
    entry_price = 0.0
    cooldown = 0

    for i in range(30, len(close)):
        if cooldown > 0:
            cooldown -= 1
            continue

        price = float(close.iloc[i])
        r = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else 50
        s20 = float(sma20.iloc[i]) if pd.notna(sma20.iloc[i]) else price
        s50 = float(sma50.iloc[i]) if pd.notna(sma50.iloc[i]) else price
        mh = float(macd_hist.iloc[i]) if macd_hist is not None and pd.notna(macd_hist.iloc[i]) else 0
        mh_prev = float(macd_hist.iloc[i-1]) if macd_hist is not None and i > 0 and pd.notna(macd_hist.iloc[i-1]) else 0
        
        # Determine Context (Simple Moving Average Slope / Relative Position)
        context = "Neutral"
        if price > s50 and s20 > s50: context = "Bullish Trend"
        elif price < s50 and s20 < s50: context = "Bearish Trend"
        
        snap = {
            "rsi": round(r, 1),
            "sma20": round(s20, 2),
            "sma50": round(s50, 2),
            "macd_hist": round(mh, 4),
            "price_vs_sma20_pct": round((price - s20) / s20 * 100, 2) if s20 > 0 else 0,
        }

        #  LONG entry 
        if not in_long and not in_short:
            bull_signals = 0
            reasons = []
            detailed_reasoning = f"The model detected a potential long entry. Market context appears {context}."
            
            if r < 35:
                bull_signals += 1; reasons.append(f"RSI oversold ({r:.0f})")
                detailed_reasoning += f" RSI is oversold at {r:.0f}, indicating potential mean reversion."
            if price > s20 and s20 > s50:
                bull_signals += 1; reasons.append("Price > SMA20 > SMA50")
                detailed_reasoning += " Price is above short-term and medium-term moving averages, confirming momentum."
            if mh > 0 and mh_prev <= 0:
                bull_signals += 1; reasons.append("MACD bullish cross")
                detailed_reasoning += " MACD Histogram crossed above zero, signaling increasing bullish momentum."

            if bull_signals >= 2:
                markers.append(TradeMarker(
                    timestamp=close.index[i], price=price,
                    side="BUY", reason=" + ".join(reasons),
                    indicator_snapshot=snap.copy(),
                    model_reasoning=detailed_reasoning,
                    regime_context=context,
                    confidence=0.6 + (0.1 * bull_signals)
                ))
                in_long = True
                entry_price = price
                cooldown = 3
                continue

        #  SHORT entry 
        if not in_long and not in_short:
            bear_signals = 0
            reasons = []
            if r > 65:
                bear_signals += 1; reasons.append(f"RSI overbought ({r:.0f})")
            if price < s20 and s20 < s50:
                bear_signals += 1; reasons.append("Price < SMA20 < SMA50")
            if mh < 0 and mh_prev >= 0:
                bear_signals += 1; reasons.append("MACD bearish cross")

            if bear_signals >= 2:
                markers.append(TradeMarker(
                    timestamp=close.index[i], price=price,
                    side="SHORT", reason=" + ".join(reasons),
                    indicator_snapshot=snap.copy(),
                ))
                in_short = True
                entry_price = price
                cooldown = 3
                continue

        #  LONG exit 
        if in_long:
            exit_reason = None
            if r > 70:
                exit_reason = f"RSI overbought exit ({r:.0f})"
            elif price < s20:
                exit_reason = "Price fell below SMA20"
            elif mh < 0 and mh_prev >= 0:
                exit_reason = "MACD bearish cross"
            elif (price - entry_price) / entry_price < -0.03:
                exit_reason = "Stop loss (-3%)"
            elif (price - entry_price) / entry_price > 0.08:
                exit_reason = "Profit target (+8%)"

            if exit_reason:
                markers.append(TradeMarker(
                    timestamp=close.index[i], price=price,
                    side="SELL", reason=exit_reason,
                    indicator_snapshot=snap.copy(),
                ))
                in_long = False
                cooldown = 3

        #  SHORT exit 
        if in_short:
            exit_reason = None
            if r < 30:
                exit_reason = f"RSI oversold exit ({r:.0f})"
            elif price > s20:
                exit_reason = "Price rose above SMA20"
            elif mh > 0 and mh_prev <= 0:
                exit_reason = "MACD bullish cross"
            elif (entry_price - price) / entry_price < -0.03:
                exit_reason = "Stop loss (-3%)"
            elif (entry_price - price) / entry_price > 0.08:
                exit_reason = "Profit target (+8%)"

            if exit_reason:
                markers.append(TradeMarker(
                    timestamp=close.index[i], price=price,
                    side="COVER", reason=exit_reason,
                    indicator_snapshot=snap.copy(),
                ))
                in_short = False
                cooldown = 3

    # Close any open position at the end
    if in_long or in_short:
        i_last = len(close) - 1
        p_last = float(close.iloc[-1])
        r_last = float(rsi.iloc[i_last]) if pd.notna(rsi.iloc[i_last]) else 50
        s20_last = float(sma20.iloc[i_last]) if pd.notna(sma20.iloc[i_last]) else p_last
        s50_last = float(sma50.iloc[i_last]) if pd.notna(sma50.iloc[i_last]) else p_last
        mh_last = float(macd_hist.iloc[i_last]) if macd_hist is not None and pd.notna(macd_hist.iloc[i_last]) else 0
        snap_last = {
            "rsi": round(r_last, 1), "sma20": round(s20_last, 2), "sma50": round(s50_last, 2),
            "macd_hist": round(mh_last, 4),
            "price_vs_sma20_pct": round((p_last - s20_last) / s20_last * 100, 2) if s20_last > 0 else 0,
        }
    if in_long:
        markers.append(TradeMarker(
            timestamp=close.index[-1], price=float(close.iloc[-1]),
            side="SELL", reason="End of period (forced exit)",
            indicator_snapshot=snap_last,
        ))
    if in_short:
        markers.append(TradeMarker(
            timestamp=close.index[-1], price=float(close.iloc[-1]),
            side="COVER", reason="End of period (forced exit)",
            indicator_snapshot=snap_last,
        ))

    return markers


def _strategy_rsi(close: pd.Series) -> List[TradeMarker]:
    """Pure RSI strategy."""
    markers = []
    rsi = _rsi(close, 14)
    in_long = False
    entry_price = 0.0
    for i in range(14, len(close)):
        price = float(close.iloc[i])
        r = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else 50
        snap = {"rsi": round(r, 1)}
        if not in_long and r < 30:
            markers.append(TradeMarker(close.index[i], price, "BUY", f"RSI oversold ({r:.0f})",
                                       indicator_snapshot=snap.copy()))
            in_long = True; entry_price = price
        elif in_long and (r > 70 or (price - entry_price) / entry_price < -0.04):
            reason = f"RSI overbought ({r:.0f})" if r > 70 else "Stop loss"
            markers.append(TradeMarker(close.index[i], price, "SELL", reason,
                                       indicator_snapshot=snap.copy()))
            in_long = False
    if in_long:
        r_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50
        markers.append(TradeMarker(close.index[-1], float(close.iloc[-1]), "SELL", "End of period",
                                   indicator_snapshot={"rsi": round(r_last, 1)}))
    return markers


def _strategy_sma_cross(close: pd.Series) -> List[TradeMarker]:
    """SMA 20/50 crossover strategy."""
    markers = []
    if len(close) < 50:
        return markers
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    in_long = False
    for i in range(50, len(close)):
        s20 = float(sma20.iloc[i]); s20p = float(sma20.iloc[i-1])
        s50 = float(sma50.iloc[i]); s50p = float(sma50.iloc[i-1])
        price = float(close.iloc[i])
        snap = {"sma20": round(s20, 2), "sma50": round(s50, 2)}
        if not in_long and s20 > s50 and s20p <= s50p:
            markers.append(TradeMarker(close.index[i], price, "BUY", "Golden cross (SMA20 > SMA50)",
                                       indicator_snapshot=snap.copy()))
            in_long = True
        elif in_long and s20 < s50 and s20p >= s50p:
            markers.append(TradeMarker(close.index[i], price, "SELL", "Death cross (SMA20 < SMA50)",
                                       indicator_snapshot=snap.copy()))
            in_long = False
    if in_long:
        s20_last = float(sma20.iloc[-1]); s50_last = float(sma50.iloc[-1])
        markers.append(TradeMarker(close.index[-1], float(close.iloc[-1]), "SELL", "End of period",
                                   indicator_snapshot={"sma20": round(s20_last, 2), "sma50": round(s50_last, 2)}))
    return markers


def _strategy_macd(close: pd.Series) -> List[TradeMarker]:
    """MACD crossover strategy."""
    markers = []
    if len(close) < 27:
        return markers
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    in_long = False
    for i in range(27, len(close)):
        m = float(macd_line.iloc[i]); s = float(signal_line.iloc[i])
        mp = float(macd_line.iloc[i-1]); sp = float(signal_line.iloc[i-1])
        price = float(close.iloc[i])
        snap = {"macd_line": round(m, 4), "signal_line": round(s, 4)}
        if not in_long and m > s and mp <= sp:
            markers.append(TradeMarker(close.index[i], price, "BUY", "MACD bullish cross",
                                       indicator_snapshot=snap.copy()))
            in_long = True
        elif in_long and m < s and mp >= sp:
            markers.append(TradeMarker(close.index[i], price, "SELL", "MACD bearish cross",
                                       indicator_snapshot=snap.copy()))
            in_long = False
    if in_long:
        m_last = float(macd_line.iloc[-1]); s_last = float(signal_line.iloc[-1])
        markers.append(TradeMarker(close.index[-1], float(close.iloc[-1]), "SELL", "End of period",
                                   indicator_snapshot={"macd_line": round(m_last, 4), "signal_line": round(s_last, 4)}))
    return markers


def _pair_trades_and_calc_pnl(markers: List[TradeMarker]):
    """Pair entries with exits and calculate P&L."""
    trade_id = 0
    entry = None
    for m in markers:
        if m.side in ("BUY", "SHORT"):
            trade_id += 1
            entry = m
            m.trade_id = trade_id
        elif m.side in ("SELL", "COVER") and entry is not None:
            m.trade_id = trade_id
            if entry.side == "BUY":
                m.pnl_pct = (m.price - entry.price) / entry.price
                m.pnl_dollar = m.price - entry.price
            elif entry.side == "SHORT":
                m.pnl_pct = (entry.price - m.price) / entry.price
                m.pnl_dollar = entry.price - m.price
            entry = None


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _format_snapshot(snap: Optional[Dict]) -> str:
    """Format indicator snapshot dict into HTML lines for hover tooltip."""
    if not snap:
        return ""
    parts = []
    if "rsi" in snap:
        parts.append(f"RSI: {snap['rsi']}")
    if "sma20" in snap:
        parts.append(f"SMA20: ${snap['sma20']:,.2f}")
    if "sma50" in snap:
        parts.append(f"SMA50: ${snap['sma50']:,.2f}")
    if "macd_hist" in snap:
        parts.append(f"MACD Hist: {snap['macd_hist']:+.4f}")
    if "macd_line" in snap:
        parts.append(f"MACD: {snap['macd_line']:+.4f}")
    if "signal_line" in snap:
        parts.append(f"Signal: {snap['signal_line']:+.4f}")
    if "price_vs_sma20_pct" in snap:
        parts.append(f"vs SMA20: {snap['price_vs_sma20_pct']:+.2f}%")
    return "<br>".join(parts)


def _build_entry_hover(m: 'TradeMarker') -> str:
    """Build hover text for an entry marker (BUY/SHORT)."""
    lines = [f"<b>{m.side}</b> ${m.price:,.2f}", f"{m.reason}"]
    
    if m.regime_context and m.regime_context != "Unknown":
        lines.append(f"Context: {m.regime_context}")
    if m.confidence > 0:
        lines.append(f"Confidence: {m.confidence:.0%}")
        
    snap_text = _format_snapshot(m.indicator_snapshot)
    if snap_text:
        lines.append(f"<br><i>Indicators:</i><br>{snap_text}")
        
    if m.model_reasoning:
        # Wrap detailed reasoning
        reasoning = m.model_reasoning
        # rough wrap
        wrapped = "<br>".join([reasoning[i:i+60] for i in range(0, len(reasoning), 60)])
        lines.append(f"<br><i>Model Logic:</i><br><span style='font-size:0.8em'>{wrapped}</span>")
        
    return "<br>".join(lines)


def _build_exit_hover(m: 'TradeMarker', entry: Optional['TradeMarker'] = None) -> str:
    """Build hover text for an exit marker (SELL/COVER)."""
    lines = [f"<b>{m.side}</b> ${m.price:,.2f}", f"{m.reason}"]
    if m.pnl_pct is not None:
        color = "lime" if m.pnl_pct > 0 else "red"
        lines.append(f"<span style='color:{color}'>P&L: {m.pnl_pct:+.2%} (${m.pnl_dollar:+,.2f})</span>")
    if entry is not None:
        try:
            days = (m.timestamp - entry.timestamp).days
            lines.append(f"Held: {days} day{'s' if days != 1 else ''}")
        except Exception:
            pass
    snap_text = _format_snapshot(m.indicator_snapshot)
    if snap_text:
        lines.append(f"<br><i>Indicators:</i><br>{snap_text}")
    return "<br>".join(lines)


def add_trade_markers_to_fig(fig, markers: List[TradeMarker], row: int = 1, col: int = 1):
    """Add trade entry/exit markers to a plotly figure."""
    if not markers:
        return fig

    # Build entry lookup for holding-period calculation on exits
    _entry_by_tid = {}
    for m in markers:
        if m.side in ("BUY", "SHORT"):
            _entry_by_tid[m.trade_id] = m

    buys = [m for m in markers if m.side == "BUY"]
    sells = [m for m in markers if m.side == "SELL"]
    shorts = [m for m in markers if m.side == "SHORT"]
    covers = [m for m in markers if m.side == "COVER"]

    if buys:
        fig.add_trace(go.Scatter(
            x=[m.timestamp for m in buys],
            y=[m.price for m in buys],
            mode='markers', name='Buy Entry',
            marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=1, color='white')),
            text=[_build_entry_hover(m) for m in buys],
            hoverinfo='text',
        ), row=row, col=col)

    if sells:
        fig.add_trace(go.Scatter(
            x=[m.timestamp for m in sells],
            y=[m.price for m in sells],
            mode='markers', name='Sell Exit',
            marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=1, color='white')),
            text=[_build_exit_hover(m, _entry_by_tid.get(m.trade_id)) for m in sells],
            hoverinfo='text',
        ), row=row, col=col)

    if shorts:
        fig.add_trace(go.Scatter(
            x=[m.timestamp for m in shorts],
            y=[m.price for m in shorts],
            mode='markers', name='Short Entry',
            marker=dict(symbol='triangle-down', size=14, color='orange', line=dict(width=1, color='white')),
            text=[_build_entry_hover(m) for m in shorts],
            hoverinfo='text',
        ), row=row, col=col)

    if covers:
        fig.add_trace(go.Scatter(
            x=[m.timestamp for m in covers],
            y=[m.price for m in covers],
            mode='markers', name='Cover Exit',
            marker=dict(symbol='triangle-up', size=14, color='cyan', line=dict(width=1, color='white')),
            text=[_build_exit_hover(m, _entry_by_tid.get(m.trade_id)) for m in covers],
            hoverinfo='text',
        ), row=row, col=col)

    # Draw lines connecting entry to exit
    for m in markers:
        if m.side in ("SELL", "COVER") and m.pnl_pct is not None:
            # Find matching entry
            entry = None
            for e in markers:
                if e.trade_id == m.trade_id and e.side in ("BUY", "SHORT"):
                    entry = e
                    break
            if entry:
                color = "rgba(0,255,0,0.3)" if m.pnl_pct > 0 else "rgba(255,0,0,0.3)"
                fig.add_trace(go.Scatter(
                    x=[entry.timestamp, m.timestamp],
                    y=[entry.price, m.price],
                    mode='lines', line=dict(color=color, width=1, dash='dot'),
                    showlegend=False, hoverinfo='skip',
                ), row=row, col=col)

    return fig


def get_trade_summary(markers: List[TradeMarker]) -> Dict:
    """Compute summary stats for the trades."""
    exits = [m for m in markers if m.pnl_pct is not None]
    if not exits:
        return {"total_trades": 0}
    pnls = [m.pnl_pct for m in exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total_trades": len(exits),
        "win_rate": len(wins) / len(exits) if exits else 0,
        "avg_win": float(np.mean(wins)) if wins else 0,
        "avg_loss": float(np.mean(losses)) if losses else 0,
        "total_return": float(np.sum(pnls)),
        "best_trade": float(max(pnls)) if pnls else 0,
        "worst_trade": float(min(pnls)) if pnls else 0,
        "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
    }


# Need this import for type hints in add_trade_markers_to_fig
import plotly.graph_objs as go
