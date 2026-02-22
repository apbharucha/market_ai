from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from data_sources import get_stock  # For price checks
from trading_system.confidence_score import ConfidenceScorer
from trading_system.interest_rate_manager import InterestRateManager
from trading_system.macro_analysis import MacroAnalyzer
from trading_system.risk_manager import AdvancedRiskManager
from trading_system.trading_data_types import (
    Direction,
    OrderType,
    PositionSpec,
    TradeAlert,
    TradeType,
)

try:
    from counter_trend_analyzer import (
        CounterTrendAnalyzer,
        CounterTrendSignal,
        get_counter_trend_analyzer,
    )

    HAS_COUNTER_TREND = True
except ImportError:
    HAS_COUNTER_TREND = False


class TradeGenerator:
    """
    Central engine for generating proactive trade alerts.

    Orchestrates three signal sources:
      1. Trend-following signals (breakout / breakdown technicals)
      2. Macro-fundamental signals (relative currency strength)
      3. Counter-trend signals (narrative fade / consensus mispricing)

    All three streams are scored via ConfidenceScorer and filtered
    through AdvancedRiskManager before producing TradeAlerts.
    """

    # Minimum confidence score to emit a TradeAlert
    CONFIDENCE_THRESHOLD: float = 60.0
    # Minimum counter-trend signal strength to consider for alert generation
    COUNTER_TREND_MIN_STRENGTH: float = 55.0

    def __init__(self, risk_manager: AdvancedRiskManager):
        self.macro_analyzer = MacroAnalyzer()
        self.rate_manager = InterestRateManager()
        self.confidence_scorer = ConfidenceScorer()
        self.risk_manager = risk_manager

        # Counter-trend engine (optional — degrades gracefully if unavailable)
        self._counter_trend: Optional["CounterTrendAnalyzer"] = None
        if HAS_COUNTER_TREND:
            try:
                self._counter_trend = get_counter_trend_analyzer()
            except Exception:
                self._counter_trend = None

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def scan_for_setups(self, symbols: List[str]) -> List[TradeAlert]:
        """
        Full scan: technical trend-following + counter-trend narrative fades.

        Returns a deduplicated, scored list of TradeAlerts sorted by
        confidence descending.
        """
        alerts: List[TradeAlert] = []

        # 1. Trend-following scan (per-symbol technicals + macro filter)
        alerts.extend(self._scan_trend_following(symbols))

        # 2. Counter-trend / narrative fade scan (macro-level, not per-symbol)
        alerts.extend(self._scan_counter_trend())

        # Deduplicate by symbol+direction, keep highest confidence
        alerts = self._deduplicate(alerts)

        # Sort by confidence descending
        alerts.sort(key=lambda a: a.confidence_score, reverse=True)

        return alerts

    def scan_trend_only(self, symbols: List[str]) -> List[TradeAlert]:
        """Run only the trend-following scanner (legacy interface)."""
        return self._scan_trend_following(symbols)

    def scan_counter_trend_only(self) -> List[TradeAlert]:
        """Run only the counter-trend / narrative fade scanner."""
        return self._scan_counter_trend()

    def get_macro_narrative_report(self) -> str:
        """Return a human-readable macro narrative / contradiction report."""
        if self._counter_trend is not None:
            return self._counter_trend.get_narrative_report()
        return "Counter-trend analyzer not available."

    def get_narrative_signals(self) -> List[Dict]:
        """
        Return a structured list of narrative divergence signals for display
        in dashboards.  Each entry is a plain dict suitable for DataFrame construction.
        """
        if self._counter_trend is None:
            return []

        out = []
        for sig in self._counter_trend.generate_counter_signals(
            divergence_threshold=15.0, min_strength=45.0
        ):
            out.append(
                {
                    "Theme": sig.theme,
                    "Instrument": sig.instrument,
                    "Direction": sig.direction,
                    "Asset Class": sig.asset_class,
                    "Signal Strength": round(sig.signal_strength, 1),
                    "Confidence": round(sig.confidence, 1),
                    "Divergence": round(sig.divergence_score, 1),
                    "Time Horizon": sig.time_horizon,
                    "Position Size %": sig.position_size_pct,
                    "Macro Narrative": sig.macro_narrative,
                    "Counter-Thesis": sig.counter_thesis,
                    "Catalyst": sig.catalyst_needed,
                    "Key Risks": " | ".join(sig.key_risks),
                }
            )
        return out

    def get_all_narratives(self) -> List[Dict]:
        """
        Return all tracked narratives (both overcrowded and underhyped)
        as plain dicts for display.
        """
        if self._counter_trend is None:
            return []

        out = []
        for n in self._counter_trend.get_all_narratives():
            out.append(
                {
                    "Theme": n.theme,
                    "Consensus Score": round(n.consensus_score, 1),
                    "Fundamental Score": round(n.fundamental_score, 1),
                    "Divergence": round(n.divergence, 1),
                    "Status": (
                        "OVERCROWDED"
                        if n.divergence > 10
                        else ("UNDERHYPED" if n.divergence < -10 else "FAIR")
                    ),
                    "Summary": n.narrative_summary,
                    "Top Contradiction": n.contradictions[0]
                    if n.contradictions
                    else "",
                }
            )
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # TREND-FOLLOWING SCANNER
    # ─────────────────────────────────────────────────────────────────────────

    def _scan_trend_following(self, symbols: List[str]) -> List[TradeAlert]:
        """Per-symbol technical breakout / breakdown scanner with macro overlay."""
        alerts: List[TradeAlert] = []

        for symbol in symbols:
            trade_type = self._determine_trade_type(symbol)

            # ── Fetch market data ─────────────────────────────────────────────
            try:
                df = get_stock(symbol, period="1mo")
                if df is None or df.empty:
                    continue
                current_price = float(df["Close"].iloc[-1])
            except Exception:
                continue

            # ── Macro score (FX only) ─────────────────────────────────────────
            macro_score = 50.0
            if trade_type == TradeType.FX:
                base = symbol[:3]
                quote = symbol[3:6]
                try:
                    macro_analysis = self.macro_analyzer.analyze_pair(base, quote)
                    net_strength = macro_analysis["net_relative_strength"]
                    macro_score = max(0.0, min(100.0, 50.0 + net_strength * 2.5))
                except Exception:
                    macro_score = 50.0

            # ── Technical signal ──────────────────────────────────────────────
            high_20d = float(df["High"].max())
            low_20d = float(df["Low"].min())

            tech_score = 50.0
            direction: Optional[Direction] = None
            trigger_price = 0.0
            stop_loss = 0.0

            if current_price >= high_20d * 0.99:  # Near 20d high → breakout
                tech_score = 80.0
                direction = Direction.LONG
                trigger_price = high_20d * 1.001
                stop_loss = current_price * 0.98
            elif current_price <= low_20d * 1.01:  # Near 20d low → breakdown
                tech_score = 80.0
                direction = Direction.SHORT
                trigger_price = low_20d * 0.999
                stop_loss = current_price * 1.02

            if direction is None:
                continue

            # ── Risk sizing ───────────────────────────────────────────────────
            position_spec = self.risk_manager.calculate_position_sizing(
                entry_price=trigger_price,
                stop_loss=stop_loss,
            )

            # ── Confidence scoring ────────────────────────────────────────────
            rates_conviction = 50.0
            ml_prob = 65.0
            score_result = self.confidence_scorer.calculate_score(
                technical_score=tech_score,
                macro_score=macro_score,
                rates_conviction=rates_conviction,
                risk_reward_ratio=position_spec.risk_reward_ratio,
                ml_probability=ml_prob,
            )

            if score_result["total_score"] < self.CONFIDENCE_THRESHOLD:
                continue

            alert = TradeAlert(
                symbol=symbol,
                trade_type=trade_type,
                direction=direction,
                trigger_price=trigger_price,
                order_type=OrderType.STOP_ENTRY,
                position_spec=position_spec,
                confidence_score=score_result["total_score"],
                confidence_breakdown=score_result["breakdown"],
                reasoning=[
                    f"Technical {'Breakout' if direction == Direction.LONG else 'Breakdown'} "
                    f"near {trigger_price:.2f}",
                    f"Macro Score: {macro_score:.1f}",
                    f"Rating: {score_result['rating']}",
                    "Signal type: TREND-FOLLOWING",
                ],
                macro_score=macro_score,
                technical_score=tech_score,
            )
            alerts.append(alert)

        return alerts

    # ─────────────────────────────────────────────────────────────────────────
    # COUNTER-TREND / NARRATIVE FADE SCANNER
    # ─────────────────────────────────────────────────────────────────────────

    def _scan_counter_trend(self) -> List[TradeAlert]:
        """
        Generate TradeAlerts for narrative-fade opportunities.

        For each strong counter-trend signal produced by CounterTrendAnalyzer,
        we:
          1. Look up the current price of the instrument.
          2. Calculate appropriate entry / stop levels.
          3. Score the signal through ConfidenceScorer.
          4. Emit a TradeAlert tagged as COUNTER_TREND.
        """
        if self._counter_trend is None:
            return []

        alerts: List[TradeAlert] = []

        try:
            ct_signals = self._counter_trend.generate_counter_signals(
                divergence_threshold=18.0,
                min_strength=self.COUNTER_TREND_MIN_STRENGTH,
            )
        except Exception:
            return []

        seen_instruments: set = set()

        for sig in ct_signals:
            instrument = sig.instrument

            # Deduplicate within this scan pass
            if instrument in seen_instruments:
                continue
            seen_instruments.add(instrument)

            # ── Fetch current price ───────────────────────────────────────────
            try:
                df = get_stock(instrument, period="1mo")
                if df is None or df.empty:
                    continue
                current_price = float(df["Close"].iloc[-1])
                if current_price <= 0:
                    continue
            except Exception:
                continue

            # ── Direction and entry/stop levels ───────────────────────────────
            if sig.direction == "LONG":
                direction = Direction.LONG
                # Enter at market; stop 3% below (counter-trend = wider stop)
                trigger_price = current_price
                stop_loss = current_price * 0.97
            else:  # SHORT
                direction = Direction.SHORT
                trigger_price = current_price
                stop_loss = current_price * 1.03

            # ── Risk sizing ───────────────────────────────────────────────────
            try:
                position_spec = self.risk_manager.calculate_position_sizing(
                    entry_price=trigger_price,
                    stop_loss=stop_loss,
                )
            except Exception:
                continue

            # ── Translate signal strength into scorer inputs ───────────────────
            # Counter-trend trades get a reduced technical score and elevated macro score
            tech_score = max(
                40.0, sig.signal_strength * 0.6
            )  # Lower — no trend support
            macro_score = min(95.0, sig.signal_strength * 1.1)  # Higher — macro driven
            rates_conviction = 55.0  # Slightly bullish conviction (rates fading)
            ml_prob = min(80.0, sig.confidence)

            try:
                score_result = self.confidence_scorer.calculate_score(
                    technical_score=tech_score,
                    macro_score=macro_score,
                    rates_conviction=rates_conviction,
                    risk_reward_ratio=position_spec.risk_reward_ratio,
                    ml_probability=ml_prob,
                )
            except Exception:
                continue

            # Counter-trend trades require a slightly lower threshold
            # (macro conviction compensates for lack of technical confirmation)
            adjusted_threshold = self.CONFIDENCE_THRESHOLD - 5.0
            if score_result["total_score"] < adjusted_threshold:
                continue

            # ── Trade type ────────────────────────────────────────────────────
            trade_type = self._determine_trade_type(instrument)

            # ── Reasoning ─────────────────────────────────────────────────────
            reasoning = [
                f"COUNTER-TREND: Fading '{sig.theme}' narrative",
                f"Divergence Score: {sig.divergence_score:+.0f} pts "
                f"(consensus {self._counter_trend._narratives[0].consensus_score:.0f} "
                f"vs fundamentals {self._counter_trend._narratives[0].fundamental_score:.0f})"
                if self._counter_trend._narratives
                else "",
                f"Counter-Thesis: {sig.counter_thesis[:120]}...",
                f"Catalyst needed: {sig.catalyst_needed}",
                f"Time Horizon: {sig.time_horizon}",
                f"Signal Strength: {sig.signal_strength:.0f}/100 | "
                f"Confidence: {sig.confidence:.0f}/100",
                "Signal type: COUNTER-TREND (Narrative Fade)",
            ]
            reasoning = [r for r in reasoning if r]  # drop empty strings

            alert = TradeAlert(
                symbol=instrument,
                trade_type=trade_type,
                direction=direction,
                trigger_price=trigger_price,
                order_type=OrderType.MARKET,  # Counter-trend: enter at market
                position_spec=position_spec,
                confidence_score=score_result["total_score"],
                confidence_breakdown=score_result["breakdown"],
                reasoning=reasoning,
                macro_score=macro_score,
                technical_score=tech_score,
            )
            alerts.append(alert)

        return alerts

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _determine_trade_type(self, symbol: str) -> TradeType:
        """Classify a symbol into a TradeType."""
        s = symbol.upper()
        if "=X" in s:
            return TradeType.FX
        if "=F" in s:
            return TradeType.FUTURES
        if s.endswith("USD") or s.endswith("USDT") or s.endswith("BTC"):
            return TradeType.CRYPTO
        if len(s) == 6 and s.isalpha():
            return TradeType.FX  # Bare 6-char FX codes like EURUSD
        return TradeType.SPOT  # Default: equity

    @staticmethod
    def _deduplicate(alerts: List[TradeAlert]) -> List[TradeAlert]:
        """
        Deduplicate alerts by (symbol, direction).
        Keeps the entry with the highest confidence score.
        """
        best: Dict[str, TradeAlert] = {}
        for alert in alerts:
            key = f"{alert.symbol}_{alert.direction.value}"
            if key not in best or alert.confidence_score > best[key].confidence_score:
                best[key] = alert
        return list(best.values())
