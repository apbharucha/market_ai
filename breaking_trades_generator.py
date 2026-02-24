"""
Octavian Breaking Trades Generator
Generates high-confidence, comprehensive trade setups with full specifications.

Author: APB - Octavian Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from data_sources import get_stock
from quant_ensemble_model import get_quant_ensemble


@dataclass
class BreakingTradeSetup:
    """Complete trade setup with all specifications"""
    # Identification
    symbol: str
    trade_id: str
    generated_at: datetime

    # Trade Direction & Type
    direction: str  # LONG or SHORT
    setup_type: str  # "Breakout", "Momentum", "Reversal", "Trend", "Range"

    # Price Levels
    current_price: float
    entry_price: float  # Trigger price
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]

    # Position Sizing
    risk_reward_ratio: float
    suggested_position_size_pct: float  # % of portfolio
    max_risk_per_trade_pct: float  # % of portfolio to risk

    # Confidence & Scoring
    confidence_score: float  # 0-100
    technical_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float

    # Comprehensive Reasoning
    primary_reason: str
    supporting_factors: List[str]
    risk_factors: List[str]
    technical_analysis: str

    # Timing
    expected_hold_time: str  # "1-3 days", "Intraday", "1-2 weeks"
    market_condition: str

    # Technical Indicators
    key_levels: Dict[str, float]
    indicators: Dict[str, any]

    # Alerts
    invalidation_price: float  # Price that invalidates setup
    alert_notes: List[str]


class BreakingTradesGenerator:
    """
    Generates high-confidence breaking trade setups with comprehensive analysis.
    Only outputs trades with confidence > 55%.
    """

    def __init__(self, min_confidence: float = 55.0):
        self.min_confidence = min_confidence
        self.quant = get_quant_ensemble()

    def generate_breaking_trades(self, symbols: List[str],
                                 max_trades: int = 5) -> List[BreakingTradeSetup]:
        """
        Generate breaking trades for given symbols.
        Returns only high-confidence setups.
        """
        setups = []

        for symbol in symbols:
            try:
                setup = self._analyze_symbol(symbol)
                if setup and setup.confidence_score >= self.min_confidence:
                    setups.append(setup)

                if len(setups) >= max_trades:
                    break
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by confidence
        setups.sort(key=lambda x: x.confidence_score, reverse=True)
        return setups[:max_trades]

    def _analyze_symbol(self, symbol: str) -> Optional[BreakingTradeSetup]:
        """Perform comprehensive analysis on a single symbol"""
        # Get data
        df = get_stock(symbol, period="3mo", interval="1d")
        if df is None or df.empty:
            return None

        # Extract price data
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float)

        if len(close) < 50:
            return None

        prices = close.values
        current_price = float(prices[-1])

        # Get volume if available
        volume = None
        if "Volume" in df.columns:
            vol = df["Volume"]
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
            volume = vol.dropna().astype(float).values

        # Get ensemble signal
        signal = self.quant.predict(prices)

        if signal.confidence < 0.40:  # Minimum threshold
            return None

        # Calculate technical indicators
        indicators = self._calculate_indicators(close, volume)

        # Determine setup
        setup_analysis = self._determine_setup(prices, signal, indicators)
        if not setup_analysis:
            return None

        # Calculate price levels
        levels = self._calculate_price_levels(
            current_price, signal.direction, indicators, setup_analysis
        )

        # Calculate scores
        scores = self._calculate_scores(signal, indicators, setup_analysis, prices)

        # Overall confidence
        confidence = self._calculate_confidence(scores, signal)

        if confidence < self.min_confidence:
            return None

        # Generate reasoning
        reasoning = self._generate_reasoning(
            symbol, signal, indicators, setup_analysis, scores
        )

        # Create setup
        setup = BreakingTradeSetup(
            symbol=symbol,
            trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            generated_at=datetime.now(),
            direction="LONG" if signal.direction == "BULLISH" else "SHORT",
            setup_type=setup_analysis['type'],
            current_price=current_price,
            entry_price=levels['entry'],
            stop_loss=levels['stop_loss'],
            take_profit_1=levels['tp1'],
            take_profit_2=levels.get('tp2'),
            take_profit_3=levels.get('tp3'),
            risk_reward_ratio=levels['risk_reward'],
            suggested_position_size_pct=self._calculate_position_size(confidence, levels['risk_reward']),
            max_risk_per_trade_pct=2.0,
            confidence_score=confidence,
            technical_score=scores['technical'],
            momentum_score=scores['momentum'],
            volatility_score=scores['volatility'],
            volume_score=scores['volume'],
            primary_reason=reasoning['primary'],
            supporting_factors=reasoning['supporting'],
            risk_factors=reasoning['risks'],
            technical_analysis=reasoning['technical_summary'],
            expected_hold_time=setup_analysis['hold_time'],
            market_condition=setup_analysis['market_condition'],
            key_levels=levels['key_levels'],
            indicators=indicators,
            invalidation_price=levels['invalidation'],
            alert_notes=reasoning['alerts']
        )

        return setup

    def _calculate_indicators(self, close: pd.Series, volume: Optional[np.ndarray]) -> Dict:
        """Calculate comprehensive technical indicators"""
        indicators = {}

        prices = close.values

        # Moving Averages
        indicators['sma_20'] = float(np.mean(prices[-20:])) if len(prices) >= 20 else None
        indicators['sma_50'] = float(np.mean(prices[-50:])) if len(prices) >= 50 else None
        indicators['ema_12'] = float(close.ewm(span=12).mean().iloc[-1]) if len(close) >= 12 else None

        # RSI
        if len(prices) >= 15:
            recent_prices = prices[-15:]
            if len(recent_prices) == 15:
                delta = np.diff(recent_prices)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                rs = avg_gain / (avg_loss + 1e-10)
                indicators['rsi'] = float(100 - (100 / (1 + rs)))
            else:
                indicators['rsi'] = 50.0
        else:
            indicators['rsi'] = 50.0

        # ATR (Average True Range)
        if len(prices) >= 14:
            high_low = np.std(prices[-14:])
            indicators['atr'] = float(high_low * 1.5)
            indicators['atr_pct'] = float(indicators['atr'] / prices[-1] * 100)
        else:
            indicators['atr'] = float(np.std(prices[-7:]) * 1.5) if len(prices) >= 7 else 0
            indicators['atr_pct'] = float(indicators['atr'] / prices[-1] * 100) if prices[-1] > 0 else 0

        # Volatility
        if len(prices) >= 21:
            price_slice = prices[-20:]
            if len(price_slice) > 1:
                returns = np.diff(price_slice) / price_slice[:-1]
                indicators['volatility'] = float(np.std(returns) * np.sqrt(252))
            else:
                indicators['volatility'] = 0.20
        else:
            indicators['volatility'] = 0.20

        # Price momentum
        if len(prices) >= 10:
            indicators['momentum_10d'] = float((prices[-1] / prices[-10] - 1) * 100)
        else:
            indicators['momentum_10d'] = 0.0

        if len(prices) >= 20:
            indicators['momentum_20d'] = float((prices[-1] / prices[-20] - 1) * 100)
        else:
            indicators['momentum_20d'] = 0.0

        # Volume analysis
        if volume is not None and len(volume) >= 20:
            avg_volume = np.mean(volume[-20:])
            recent_volume = np.mean(volume[-5:])
            indicators['volume_ratio'] = float(recent_volume / avg_volume) if avg_volume > 0 else 1.0
        else:
            indicators['volume_ratio'] = 1.0

        # Trend strength
        if len(prices) >= 20:
            x_vals = np.arange(20)
            y_vals = prices[-20:]
            if len(y_vals) == 20:  # Ensure exact length match
                trend_slope = np.polyfit(x_vals, y_vals, 1)[0]
                indicators['trend_strength'] = float(trend_slope / prices[-1] * 100)
            else:
                indicators['trend_strength'] = 0.0
        else:
            indicators['trend_strength'] = 0.0

        return indicators

    def _determine_setup(self, prices: np.ndarray, signal, indicators: Dict) -> Optional[Dict]:
        """Determine the type of setup and market condition"""
        rsi = indicators.get('rsi', 50)
        momentum_10d = indicators.get('momentum_10d', 0)
        trend_strength = indicators.get('trend_strength', 0)
        vol_ratio = indicators.get('volume_ratio', 1.0)

        setup = {}

        # Determine setup type
        if signal.direction == "BULLISH":
            if rsi < 35 and momentum_10d < -5:
                setup['type'] = "Oversold Reversal"
                setup['hold_time'] = "2-5 days"
            elif trend_strength > 0.5 and vol_ratio > 1.2:
                setup['type'] = "Momentum Breakout"
                setup['hold_time'] = "1-3 days"
            elif momentum_10d > 3 and trend_strength > 0.3:
                setup['type'] = "Trend Continuation"
                setup['hold_time'] = "3-7 days"
            else:
                setup['type'] = "Bullish Setup"
                setup['hold_time'] = "2-5 days"
        else:
            if rsi > 65 and momentum_10d > 5:
                setup['type'] = "Overbought Reversal"
                setup['hold_time'] = "2-5 days"
            elif trend_strength < -0.5 and vol_ratio > 1.2:
                setup['type'] = "Breakdown"
                setup['hold_time'] = "1-3 days"
            elif momentum_10d < -3 and trend_strength < -0.3:
                setup['type'] = "Downtrend Continuation"
                setup['hold_time'] = "3-7 days"
            else:
                setup['type'] = "Bearish Setup"
                setup['hold_time'] = "2-5 days"

        # Market condition
        if abs(trend_strength) > 0.5:
            setup['market_condition'] = "Strong Trend"
        elif abs(trend_strength) < 0.1:
            setup['market_condition'] = "Range-Bound"
        else:
            setup['market_condition'] = "Trending"

        return setup

    def _calculate_price_levels(self, current_price: float, direction: str,
                                 indicators: Dict, setup: Dict) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        atr = indicators.get('atr', current_price * 0.02)

        levels = {}
        levels['key_levels'] = {}

        if direction == "BULLISH":
            # Entry: slight pullback or current
            entry = current_price * 0.998  # 0.2% below current

            # Stop loss: 1.5x ATR or 2-3% below entry
            stop_distance = max(atr * 1.5, entry * 0.02)
            stop_loss = entry - stop_distance

            # Take profits
            risk = entry - stop_loss
            tp1 = entry + (risk * 2.0)  # 2R
            tp2 = entry + (risk * 3.5)  # 3.5R
            tp3 = entry + (risk * 5.0)  # 5R

            invalidation = stop_loss * 0.995

            levels['key_levels']['resistance_1'] = entry + (risk * 2.0)
            levels['key_levels']['resistance_2'] = entry + (risk * 4.0)
            levels['key_levels']['support'] = stop_loss
        else:
            # Short setup
            entry = current_price * 1.002  # 0.2% above current

            stop_distance = max(atr * 1.5, entry * 0.02)
            stop_loss = entry + stop_distance

            risk = stop_loss - entry
            tp1 = entry - (risk * 2.0)
            tp2 = entry - (risk * 3.5)
            tp3 = entry - (risk * 5.0)

            invalidation = stop_loss * 1.005

            levels['key_levels']['support_1'] = entry - (risk * 2.0)
            levels['key_levels']['support_2'] = entry - (risk * 4.0)
            levels['key_levels']['resistance'] = stop_loss

        levels['entry'] = round(entry, 2)
        levels['stop_loss'] = round(stop_loss, 2)
        levels['tp1'] = round(tp1, 2)
        levels['tp2'] = round(tp2, 2)
        levels['tp3'] = round(tp3, 2)
        levels['invalidation'] = round(invalidation, 2)
        levels['risk_reward'] = round((tp1 - entry) / abs(entry - stop_loss), 2) if abs(entry - stop_loss) > 0 else 2.0

        return levels

    def _calculate_scores(self, signal, indicators: Dict, setup: Dict, prices: np.ndarray) -> Dict:
        """Calculate individual component scores"""
        scores = {}

        # Technical Score (0-100)
        tech_score = 50.0
        rsi = indicators.get('rsi', 50)
        if signal.direction == "BULLISH":
            if 30 < rsi < 50:
                tech_score += 20
            elif rsi < 30:
                tech_score += 30
        else:
            if 50 < rsi < 70:
                tech_score += 20
            elif rsi > 70:
                tech_score += 30

        scores['technical'] = min(tech_score, 100.0)

        # Momentum Score
        momentum_10d = indicators.get('momentum_10d', 0)
        momentum_score = 50.0 + min(abs(momentum_10d) * 3, 50)
        scores['momentum'] = min(momentum_score, 100.0)

        # Volatility Score (prefer moderate volatility)
        vol = indicators.get('volatility', 0.20)
        if 0.15 < vol < 0.35:
            vol_score = 80.0
        elif 0.10 < vol < 0.50:
            vol_score = 60.0
        else:
            vol_score = 40.0
        scores['volatility'] = vol_score

        # Volume Score
        vol_ratio = indicators.get('volume_ratio', 1.0)
        if vol_ratio > 1.3:
            volume_score = 90.0
        elif vol_ratio > 1.1:
            volume_score = 70.0
        else:
            volume_score = 50.0
        scores['volume'] = volume_score

        return scores

    def _calculate_confidence(self, scores: Dict, signal) -> float:
        """Calculate overall confidence score"""
        # Weighted average
        confidence = (
            scores['technical'] * 0.30 +
            scores['momentum'] * 0.25 +
            scores['volatility'] * 0.15 +
            scores['volume'] * 0.20 +
            signal.confidence * 100 * 0.10
        )

        return round(confidence, 1)

    def _calculate_position_size(self, confidence: float, risk_reward: float) -> float:
        """Calculate suggested position size as % of portfolio"""
        # Base size on confidence and risk/reward
        base_size = (confidence / 100) * 10  # Max 10% if 100% confidence

        # Adjust for risk/reward
        if risk_reward >= 3.0:
            multiplier = 1.2
        elif risk_reward >= 2.0:
            multiplier = 1.0
        else:
            multiplier = 0.8

        suggested_size = base_size * multiplier
        return round(min(suggested_size, 15.0), 1)  # Cap at 15%

    def _generate_reasoning(self, symbol: str, signal, indicators: Dict,
                           setup_analysis: Dict, scores: Dict) -> Dict:
        """Generate comprehensive reasoning for the trade"""
        reasoning = {}

        # Primary reason
        if setup_analysis['type'] == "Momentum Breakout":
            reasoning['primary'] = f"{symbol} showing strong momentum breakout with high volume confirmation and technical alignment."
        elif setup_analysis['type'] == "Oversold Reversal":
            reasoning['primary'] = f"{symbol} is oversold (RSI: {indicators['rsi']:.1f}) with signs of reversal and positive divergence."
        elif setup_analysis['type'] == "Trend Continuation":
            reasoning['primary'] = f"{symbol} continues strong trend with momentum intact and favorable risk/reward."
        elif setup_analysis['type'] == "Overbought Reversal":
            reasoning['primary'] = f"{symbol} showing overbought conditions (RSI: {indicators['rsi']:.1f}) with reversal signals."
        else:
            reasoning['primary'] = f"{symbol} presents high-probability {signal.direction.lower()} setup with strong technical confirmation."

        # Supporting factors
        supporting = []

        if indicators['momentum_10d'] > 5:
            supporting.append(f"Strong upward momentum (+{indicators['momentum_10d']:.1f}% over 10 days)")
        elif indicators['momentum_10d'] < -5:
            supporting.append(f"Strong downward momentum ({indicators['momentum_10d']:.1f}% over 10 days)")

        if indicators['volume_ratio'] > 1.3:
            supporting.append(f"Above-average volume (1.3x recent average) confirming move")

        if indicators.get('sma_20') and indicators.get('sma_50'):
            if indicators['sma_20'] > indicators['sma_50']:
                supporting.append("Bullish MA crossover (20 > 50)")
            else:
                supporting.append("Bearish MA alignment (20 < 50)")

        if scores['technical'] > 75:
            supporting.append("Strong technical setup with multiple confirmations")

        if signal.confidence > 0.55:
            supporting.append(f"High model confidence ({signal.confidence*100:.0f}%)")

        reasoning['supporting'] = supporting

        # Risk factors
        risks = []

        if indicators['volatility'] > 0.40:
            risks.append("High volatility - expect larger price swings")

        if indicators['volume_ratio'] < 0.8:
            risks.append("Below-average volume - may lack follow-through")

        if setup_analysis['market_condition'] == "Range-Bound":
            risks.append("Range-bound market - breakout may fail")

        if not risks:
            risks.append("Standard market risk applies")

        reasoning['risks'] = risks

        # Technical summary
        tech_summary = f"RSI: {indicators['rsi']:.1f} | "
        tech_summary += f"10-Day Momentum: {indicators['momentum_10d']:+.1f}% | "
        tech_summary += f"Volatility: {indicators['volatility']*100:.1f}% | "
        tech_summary += f"Volume: {indicators['volume_ratio']:.2f}x avg | "
        tech_summary += f"Trend: {setup_analysis['market_condition']}"

        reasoning['technical_summary'] = tech_summary

        # Alerts
        alerts = []
        # Note: 'entry' is not in setup_analysis, it's in levels dict calculated earlier
        alerts.append(f"Monitor price for entry signal")
        alerts.append(f"Expected hold time: {setup_analysis['hold_time']}")
        alerts.append(f"Scale out at multiple profit targets for optimal risk management")

        reasoning['alerts'] = alerts

        return reasoning


# Singleton
_breaking_trades_gen = None

def get_breaking_trades_generator() -> BreakingTradesGenerator:
    global _breaking_trades_gen
    if _breaking_trades_gen is None:
        _breaking_trades_gen = BreakingTradesGenerator()
    return _breaking_trades_gen
