"""
Octavian Market Simulation Engine
Ultra-realistic daily market simulation system for model training and improvement

This module runs comprehensive market simulations 3 times daily, generating
realistic market scenarios with news, events, and circumstances for the AI
model to navigate and learn from.

Author: APB - Octavian Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import random
import json
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
import schedule
import time
from concurrent.futures import ThreadPoolExecutor
import os

# Make heavy imports lazy to prevent app-wide crashes
try:
    from source_credibility_engine import SourceCredibilityEngine
except ImportError:
    SourceCredibilityEngine = None

try:
    from timeframe_analysis_engine import TimeframeAnalysisEngine, TimeframeScope
except ImportError:
    TimeframeAnalysisEngine = None
    class TimeframeScope:
        INTRADAY = "intraday"
        SWING = "swing"

# Quant Ensemble integration (graceful)
try:
    from quant_ensemble_model import get_quant_ensemble
    HAS_QUANT_ENSEMBLE = True
except ImportError:
    HAS_QUANT_ENSEMBLE = False

LEARNED_PARAMS_FILE = "octavian_learned_params.json"

def _load_learned_params() -> Dict[str, Any]:
    """Load previously learned parameters from disk."""
    try:
        if os.path.exists(LEARNED_PARAMS_FILE):
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                params = json.load(f)
            print(f"[OK] Loaded learned params from {LEARNED_PARAMS_FILE}: {params}")
            return params
    except Exception as e:
        print(f"[WARN] Could not load learned params: {e}")
    return {}

def _save_learned_params(params: Dict[str, Any]):
    """Persist learned parameters to disk."""
    try:
        with open(LEARNED_PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        print(f"[OK] Saved learned params to {LEARNED_PARAMS_FILE}")
    except Exception as e:
        print(f"[WARN] Could not save learned params: {e}")


class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    SECTOR_ROTATION = "sector_rotation"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"

class NewsType(Enum):
    EARNINGS = "earnings"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    CENTRAL_BANK = "central_bank"
    CORPORATE_ACTION = "corporate_action"
    SECTOR_NEWS = "sector_news"
    REGULATORY = "regulatory"
    MARKET_STRUCTURE = "market_structure"
    ANALYST_RATING = "analyst_rating"
    INSIDER_TRADE = "insider_trade"
    MACRO_INDICATOR = "macro_indicator"
    COMMODITY_SHOCK = "commodity_shock"
    CURRENCY_EVENT = "currency_event"
    TECHNICAL_SIGNAL = "technical_signal"
    SOCIAL_SENTIMENT = "social_sentiment"

@dataclass
class SimulatedNewsEvent:
    """Simulated news event with market impact."""
    event_id: str
    timestamp: datetime
    news_type: NewsType
    headline: str
    content: str
    affected_symbols: List[str]
    affected_sectors: List[str]
    sentiment_score: float
    market_impact_score: float
    credibility_tier: str
    source: str
    expected_price_impact: Dict[str, float]
    volatility_impact: float
    volume_impact: float

@dataclass
class SimulatedMarketData:
    """Simulated market data point."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    volatility: float
    bid_ask_spread: float
    market_cap: float
    news_events: List[SimulatedNewsEvent]

@dataclass
class TradingDecision:
    """AI model trading decision."""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: float
    price: float
    reasoning: List[str]
    confidence: float
    expected_return: float
    risk_assessment: float
    timeframe: TimeframeScope
    stop_loss: Optional[float]
    take_profit: Optional[float]

@dataclass
class SimulationResult:
    """Complete simulation result."""
    simulation_id: str
    start_time: datetime
    end_time: datetime
    market_regime: MarketRegime
    total_decisions: int
    successful_decisions: int
    failed_decisions: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    key_insights: List[str]
    model_improvements: List[str]
    missed_opportunities: List[str]
    successful_predictions: List[str]
    market_reactions: Dict[str, Any]
    performance_metrics: Dict[str, float]
    news_impact_analysis: Dict[str, Any]
    sector_performance: Dict[str, float]
    volatility_analysis: Dict[str, float]
    correlation_analysis: Dict[str, float]

@dataclass
class _FallbackAnalysis:
    profit_probability: float
    model_reasoning: List[str]

class _FallbackUnbiasedAnalyzer:
    def __init__(self):
        self.asset_universe = {
            # Mega/Large cap — diverse sectors
            "stocks_mega": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS",
            ],
            "stocks_tech": [
                "CRM", "ADBE", "AMD", "INTC", "AVGO", "QCOM", "NOW", "ORCL",
                "NET", "CRWD", "DDOG", "ZS", "PLTR", "SNOW", "COIN", "SHOP",
                "MU", "LRCX", "KLAC", "AMAT", "ARM", "SMCI", "DELL", "HPE",
            ],
            "stocks_finance": [
                "GS", "BAC", "MS", "WFC", "SCHW", "BLK", "AXP", "C",
                "KKR", "BX", "PNC", "USB", "CME", "ICE", "PYPL", "SQ",
                "SOFI", "HOOD", "COIN", "AFRM", "UPST",
            ],
            "stocks_healthcare": [
                "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "AMGN", "GILD",
                "REGN", "VRTX", "MRNA", "BIIB", "ISRG", "SYK", "CVS", "CI",
                "CRSP", "BEAM", "EDIT", "NTLA", "ALNY", "SRPT", "HIMS",
            ],
            "stocks_energy": [
                "XOM", "CVX", "COP", "SLB", "OXY", "HAL", "DVN", "MPC",
                "VLO", "PSX", "EOG", "FANG", "MRO", "BKR", "KMI", "WMB",
                "AR", "RRC", "EQT", "CTRA",
            ],
            "stocks_consumer": [
                "WMT", "COST", "MCD", "SBUX", "NKE", "LULU", "CMG", "TGT",
                "LOW", "TJX", "KO", "PEP", "EL", "MNST", "DPZ", "YUM",
                "ABNB", "BKNG", "UBER", "DASH",
            ],
            "stocks_industrial": [
                "CAT", "DE", "HON", "GE", "RTX", "LMT", "BA", "UPS",
                "UNP", "NOC", "GD", "ETN", "ITW", "FDX", "WM", "HEI",
                "AXON", "KTOS", "AVAV",
            ],
            "stocks_materials": [
                "LIN", "SHW", "FCX", "NEM", "NUE", "STLD", "CLF", "X",
                "AA", "APD", "DOW", "DD", "VMC", "MLM", "ALB", "CENX",
            ],
            "stocks_mining": [
                "NEM", "GOLD", "AEM", "FNV", "WPM", "RGLD", "KGC", "AGI",
                "PAAS", "HL", "EGO", "AU", "BTG", "IAG", "CDE", "NGD",
                "AG", "MAG", "FSM", "USAS", "EXK", "SVM", "SILV",
                "FCX", "SCCO", "TECK", "HBM", "ERO",
                "BHP", "RIO", "VALE",
                "HMY", "DRD", "GFI", "SBSW", "SA",
            ],
            "stocks_minerals_ree": [
                "MP", "UUUU", "ALB", "SQM", "LTHM", "PLL", "SGML", "LAC",
                "CCJ", "UEC", "DNN", "NXE", "LEU", "SMR", "OKLO", "URG",
                "NTR", "MOS", "CF", "IPI", "ICL",
            ],
            "stocks_nuclear_uranium": [
                "CCJ", "UEC", "UUUU", "DNN", "NXE", "LEU", "SMR", "OKLO",
                "BWXT", "GEV", "CEG", "VST",
            ],
            "stocks_realestate": [
                "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "EXR", "AVB",
                "EQR", "SPG", "WELL", "DLR", "MPW", "STAG", "REXR", "IIPR",
            ],
            "stocks_utilities": [
                "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE",
                "AWK", "WEC", "ES", "ED",
            ],
            "stocks_renewables": [
                "ENPH", "FSLR", "SEDG", "RUN", "PLUG", "BE", "CHPT", "EVGO",
                "BLNK", "STEM", "BEEM", "ARRY", "NOVA",
            ],
            "stocks_defense": [
                "LMT", "RTX", "NOC", "GD", "LHX", "HII", "LDOS", "SAIC",
                "BAH", "KTOS", "AVAV", "AXON", "SWBI", "RGR",
            ],
            "stocks_agriculture": [
                "ADM", "BG", "INGR", "DAR", "DE", "AGCO", "CNHI",
                "NTR", "MOS", "CF", "TSN", "HRL", "ZTS",
            ],
            "stocks_shipping": [
                "ZIM", "MATX", "GOGL", "EGLE", "GNK", "SBLK", "STNG",
                "DAC", "NMM", "INSW", "NAT", "DHT", "FRO",
            ],
            "stocks_cannabis": [
                "TLRY", "CGC", "ACB", "CRON", "OGI", "SNDL", "IIPR", "GRWG",
            ],
            # Mid/Small cap — high potential, often overlooked
            "stocks_smallmid": [
                "SOFI", "HOOD", "RIVN", "LCID", "JOBY", "IONQ", "AFRM", "UPST",
                "ROKU", "RBLX", "DKNG", "ENPH", "FSLR", "PLUG", "SMCI", "AI",
                "ABNB", "DASH", "LYFT", "PINS", "SNAP", "ETSY", "W", "CHWY",
                "DNA", "BEAM", "EDIT", "CRSP", "STEM", "BLNK", "EVGO", "PATH",
                "SOUN", "LUNR", "ASTS", "RKLB", "BKSY",
            ],
            # International ADRs
            "stocks_international": [
                "BABA", "TSM", "ASML", "NVO", "TM", "SAP", "SE", "MELI",
                "NU", "PDD", "JD", "BIDU", "NIO", "LI", "XPEV", "INFY",
                "VALE", "PBR", "ITUB", "SHOP", "BHP", "RIO",
            ],
            # ETFs — broad market, sector, thematic
            "etfs": [
                "SPY", "QQQ", "IWM", "VTI", "DIA", "ARKK", "XLF", "XLK",
                "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "VEA",
                "VWO", "EEM", "FXI", "EWJ", "EWZ", "INDA", "ICLN", "HACK",
                "GDX", "GDXJ", "SIL", "COPX", "REMX", "URA", "LIT", "MSOS",
            ],
            # Crypto
            "crypto": [
                "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD",
                "XRP-USD", "AVAX-USD", "MATIC-USD", "LINK-USD", "DOT-USD",
            ],
            # Forex
            "forex": [
                "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
                "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF",
            ],
            # Futures
            "futures": [
                "ES=F", "NQ=F", "YM=F", "CL=F", "GC=F", "SI=F",
                "NG=F", "ZN=F", "ZB=F", "HG=F",
            ],
            # Commodities / Bonds ETFs
            "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC"],
            "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "EMB", "TIP"],
        }
        self._price_history: Dict[str, List[float]] = {}
        # Track per-symbol persistent trend state for smarter analysis
        self._symbol_trend: Dict[str, float] = {}

    def update_price(self, symbol: str, price: float):
        """Feed price data so the analyzer can compute real signals."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)
        if len(self._price_history[symbol]) > 80:
            self._price_history[symbol] = self._price_history[symbol][-80:]

    async def analyze_unbiased(self, symbol: str) -> _FallbackAnalysis:
        """Smart analysis using actual simulated price history.
        
        Key insight: the price simulator now generates persistent trends,
        so momentum signals ARE exploitable. This analyzer is calibrated
        to detect and act on those trends while respecting mean-reversion
        at extremes.
        """
        prices = self._price_history.get(symbol, [])

        if len(prices) < 8:
            return _FallbackAnalysis(
                profit_probability=0.50,
                model_reasoning=[f"Waiting for data ({len(prices)}/8 bars)."]
            )

        prices_arr = np.array(prices, dtype=float)
        n = len(prices_arr)

        #  1. Multi-timeframe momentum (primary signal) 
        ret_3 = (prices_arr[-1] / prices_arr[-3] - 1) if n >= 3 else 0.0
        ret_5 = (prices_arr[-1] / prices_arr[-5] - 1) if n >= 5 else 0.0
        ret_10 = (prices_arr[-1] / prices_arr[-10] - 1) if n >= 10 else 0.0
        ret_20 = (prices_arr[-1] / prices_arr[-20] - 1) if n >= 20 else 0.0

        # Weighted momentum — emphasize recent moves (where trend is strongest)
        momentum = ret_3 * 0.40 + ret_5 * 0.30 + ret_10 * 0.20 + ret_20 * 0.10

        #  2. Trend strength via linear regression slope 
        lookback = min(n, 30)
        recent = prices_arr[-lookback:]
        x = np.arange(lookback)
        if lookback >= 5:
            slope = np.polyfit(x, recent, 1)[0]
            # Normalize slope relative to price level
            norm_slope = slope / (recent.mean() + 1e-8)
        else:
            norm_slope = 0.0

        #  3. Mean reversion (only at extremes, z-score > 2) 
        if n >= 20:
            ma20 = prices_arr[-20:].mean()
            std20 = prices_arr[-20:].std()
            z_score = (prices_arr[-1] - ma20) / (std20 + 1e-8)
            # Only apply mean reversion at extremes (|z| > 1.8)
            if abs(z_score) > 1.8:
                mean_rev = -np.tanh((z_score - np.sign(z_score) * 1.8) * 0.8) * 0.25
            else:
                mean_rev = 0.0
        else:
            mean_rev = 0.0
            z_score = 0.0

        #  4. Volatility-adjusted confidence 
        if n >= 10:
            returns = np.diff(prices_arr[-10:]) / prices_arr[-10:-1]
            vol = returns.std()
            # Signal-to-noise: strong trend with low vol = high confidence
            signal_to_noise = abs(momentum) / (vol + 1e-8)
            vol_adj = min(vol * 1.5, 0.12)
        else:
            vol_adj = 0.05
            signal_to_noise = 0.0

        #  5. Acceleration (is momentum increasing or decreasing?) 
        if n >= 10:
            mom_recent = ret_3
            mom_older = (prices_arr[-6] / prices_arr[-8] - 1) if n >= 8 else 0.0
            acceleration = mom_recent - mom_older
        else:
            acceleration = 0.0

        #  6. Combine into probability 
        raw_signal = (
            momentum * 20.0 +              # Primary: trend following
            norm_slope * 8.0 +              # Slope confirmation
            mean_rev +                       # Mean reversion at extremes only
            acceleration * 5.0 +             # Momentum acceleration bonus
            min(signal_to_noise, 3) * 0.03   # SNR bonus
        )

        prob = 0.50 + np.tanh(raw_signal) * 0.38
        prob = max(0.12, min(0.88, prob - vol_adj * 0.5))

        reasoning = []
        reasoning.append(f"Momentum 3/5/10: {ret_3*100:+.3f}%/{ret_5*100:+.3f}%/{ret_10*100:+.3f}%")
        reasoning.append(f"Trend slope: {norm_slope*100:.4f}%/bar | Accel: {acceleration*100:+.3f}%")
        if abs(z_score) > 1.5:
            reasoning.append(f"Z-score {z_score:.2f} — {'overbought, mean-rev active' if z_score > 0 else 'oversold, mean-rev active'}")
        reasoning.append(f"SNR: {signal_to_noise:.2f} | Vol penalty: -{vol_adj*0.5:.3f}")
        reasoning.append(f"Final prob: {prob:.3f}")

        return _FallbackAnalysis(
            profit_probability=prob,
            model_reasoning=reasoning
        )

class SimulationLearningEngine:
    """
    Analyzes simulation outcomes to identify systematic model errors,
    generate actionable improvement insights, and auto-apply parameter changes.
    """

    def __init__(self, engine: 'MarketSimulationEngine'):
        self.engine = engine

    def analyze_and_improve(self, state: Dict[str, Any], regime: MarketRegime) -> Dict[str, Any]:
        """Run full post-simulation analysis, return insights and auto-apply fixes."""
        trades = state.get('trades', [])
        if not trades:
            return {'insights': ["No trades executed — consider lowering buy threshold."],
                    'parameter_changes': [], 'error_patterns': [],
                    'simulation_grade': self._grade_simulation_empty(),
                    'trade_grades': []}

        sells = [t for t in trades if t.get('action') == 'SELL' and t.get('actual_outcome') is not None]
        buys = [t for t in trades if t.get('action') == 'BUY']

        results = {
            'insights': [],
            'parameter_changes': [],
            'error_patterns': [],
            'regime': regime.value,
        }

        # --- 1. Per-sector win-rate analysis ---
        self._analyze_sector_performance(sells, results)

        # --- 2. Regime-specific accuracy ---
        self._analyze_regime_fit(sells, regime, results)

        # --- 3. Confidence calibration ---
        self._analyze_confidence_calibration(sells, results)

        # --- 4. Hold-time analysis ---
        self._analyze_hold_time(state, sells, results)

        # --- 5. Position sizing effectiveness ---
        self._analyze_position_sizing(trades, results)

        # --- 6. Threshold tuning ---
        self._analyze_threshold_effectiveness(sells, results)

        # --- 7. Volatility-based errors ---
        self._analyze_volatility_errors(trades, state, results)

        # --- 8. Loss pattern detection ---
        self._analyze_loss_patterns(sells, results)

        #  GRADING 
        trade_outcomes = [t.get('actual_outcome', 0) for t in sells if t.get('actual_outcome') is not None]
        results['trade_grades'] = self._grade_trades(sells)
        results['simulation_grade'] = self._grade_full_simulation(sells, trade_outcomes, state)

        # Add grade insights
        sg = results['simulation_grade']
        results['insights'].insert(0,
            f" SIMULATION GRADE: {sg.get('letter_grade', '?')} ({sg.get('composite', 0):.0f}/100)"
        )
        for c in sg.get('commentary', []):
            results['insights'].append(f"[LIST] {c}")

        # Grade distribution insight
        grade_dist = {}
        for tg in results['trade_grades']:
            lg = tg.get('letter_grade', '?')
            grade_dist[lg] = grade_dist.get(lg, 0) + 1
        if grade_dist:
            dist_str = " | ".join(f"{g}: {c}" for g, c in sorted(grade_dist.items()))
            results['insights'].append(f" Trade Grade Distribution: {dist_str}")
        results['grade_distribution'] = grade_dist

        # --- 9. Auto-apply improvements ---
        self._auto_apply_improvements(results)

        return results

    def _analyze_sector_performance(self, sells, results):
        """Find sectors where the model consistently loses money."""
        sector_stats: Dict[str, Dict] = {}
        for t in sells:
            sec = self.engine._get_symbol_sector(t['symbol'])
            if sec not in sector_stats:
                sector_stats[sec] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
            if t['actual_outcome'] > 0:
                sector_stats[sec]['wins'] += 1
            else:
                sector_stats[sec]['losses'] += 1
            sector_stats[sec]['total_pnl'] += t.get('actual_outcome_dollar', 0)

        for sec, stats in sector_stats.items():
            total = stats['wins'] + stats['losses']
            if total < 3:
                continue
            wr = stats['wins'] / total
            pnl = stats['total_pnl']

            if wr < 0.35 and total >= 3:
                results['error_patterns'].append({
                    'type': 'sector_underperformance',
                    'sector': sec,
                    'win_rate': wr,
                    'pnl_dollar': pnl,
                    'trades': total,
                })
                results['insights'].append(
                    f"[ALERT] SECTOR WEAKNESS: {sec} sector has {wr:.0%} win rate across {total} trades "
                    f"(${pnl:+,.2f}). Model should RAISE buy threshold for {sec} by +0.05 "
                    f"or REDUCE max sector exposure from 5 to 3."
                )
                results['parameter_changes'].append({
                    'param': f'sector_threshold_adj_{sec}',
                    'before': self.engine.decision_thresholds['buy_prob'],
                    'after': min(0.80, self.engine.decision_thresholds['buy_prob'] + 0.05),
                    'reason': f"{sec} win rate {wr:.0%} is below 35%"
                })
            elif wr > 0.65 and total >= 3:
                results['insights'].append(
                    f"[GOOD] SECTOR STRENGTH: {sec} sector has {wr:.0%} win rate across {total} trades "
                    f"(${pnl:+,.2f}). Model should INCREASE allocation to {sec}."
                )

    def _analyze_regime_fit(self, sells, regime, results):
        """Check if model performs poorly in the current regime."""
        if not sells:
            return
        total = len(sells)
        wins = sum(1 for t in sells if t['actual_outcome'] > 0)
        wr = wins / total if total > 0 else 0

        if wr < 0.40 and total >= 5:
            results['error_patterns'].append({
                'type': 'regime_mismatch',
                'regime': regime.value,
                'win_rate': wr,
            })
            if regime in (MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS):
                results['insights'].append(
                    f"[ALERT] REGIME MISMATCH: Model achieved only {wr:.0%} accuracy in {regime.value} regime. "
                    f"In high-volatility regimes, the model should: "
                    f"(1) INCREASE buy threshold to >{0.65:.2f}, "
                    f"(2) WIDEN stop losses by 1.5x, "
                    f"(3) REDUCE position sizes by 40%. Auto-applying adjustments."
                )
                results['parameter_changes'].append({
                    'param': 'buy_prob_regime_adj',
                    'before': self.engine.decision_thresholds['buy_prob'],
                    'after': min(0.75, self.engine.decision_thresholds['buy_prob'] + 0.07),
                    'reason': f"Poor {wr:.0%} accuracy in {regime.value}"
                })
            elif regime in (MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY):
                results['insights'].append(
                    f"[ALERT] REGIME MISMATCH: Model achieved only {wr:.0%} accuracy in {regime.value} regime. "
                    f"In low-vol/sideways markets, momentum signals are unreliable. "
                    f"Model should WEIGHT mean-reversion signals higher and REDUCE momentum weight."
                )
        elif wr > 0.60 and total >= 5:
            results['insights'].append(
                f"[GOOD] REGIME FIT: Model performed well in {regime.value} ({wr:.0%} accuracy, {total} trades). "
                f"Current parameters are well-calibrated for this regime."
            )

    def _analyze_confidence_calibration(self, sells, results):
        """Check if confidence scores actually predict success."""
        high_conf = [t for t in sells if t.get('confidence', 0) > 0.7]
        low_conf = [t for t in sells if t.get('confidence', 0) < 0.5]

        if len(high_conf) >= 3:
            hc_wr = sum(1 for t in high_conf if t['actual_outcome'] > 0) / len(high_conf)
            if hc_wr < 0.50:
                results['error_patterns'].append({
                    'type': 'confidence_miscalibration',
                    'high_conf_win_rate': hc_wr,
                })
                results['insights'].append(
                    f"[ALERT] CONFIDENCE MISCALIBRATION: High-confidence trades (>70%) only won {hc_wr:.0%} of the time. "
                    f"The confidence formula overestimates edge. Fix: "
                    f"(1) REDUCE edge multiplier from 2.0 to 1.5 in confidence calc, "
                    f"(2) ADD volatility penalty: subtract vol*0.15 instead of vol*0.1, "
                    f"(3) REQUIRE minimum 3 confirming signals for >70% confidence."
                )
            elif hc_wr > 0.65:
                results['insights'].append(
                    f"[GOOD] CONFIDENCE CALIBRATED: High-confidence trades won {hc_wr:.0%}. "
                    f"Consider INCREASING allocation for high-confidence setups."
                )

        if len(low_conf) >= 3:
            lc_wr = sum(1 for t in low_conf if t['actual_outcome'] > 0) / len(low_conf)
            if lc_wr > 0.55:
                results['insights'].append(
                    f"[WARN] LOW-CONFIDENCE OPPORTUNITY: Low-confidence trades (<50%) actually won {lc_wr:.0%}. "
                    f"Model is under-confident. Consider LOWERING buy threshold for these setups."
                )

    def _analyze_hold_time(self, state, sells, results):
        """Analyze whether hold duration affects outcomes."""
        short_holds = []  # < 5 min
        long_holds = []   # > 15 min

        for t in sells:
            sym = t['symbol']
            buy_time = state.get('last_buy_time', {}).get(sym)
            sell_time = t.get('timestamp')
            if buy_time and sell_time:
                try:
                    hold_secs = (sell_time - buy_time).total_seconds()
                except Exception:
                    continue
                if hold_secs < 300:
                    short_holds.append(t)
                elif hold_secs > 900:
                    long_holds.append(t)

        if len(short_holds) >= 3:
            sh_wr = sum(1 for t in short_holds if t['actual_outcome'] > 0) / len(short_holds)
            if sh_wr < 0.40:
                results['insights'].append(
                    f"[ALERT] PREMATURE EXITS: Short-hold trades (<5 min) have {sh_wr:.0%} win rate. "
                    f"Model exits too early. Fix: INCREASE min_hold_minutes from "
                    f"{self.engine.min_hold_minutes} to {self.engine.min_hold_minutes + 3}."
                )
                results['parameter_changes'].append({
                    'param': 'min_hold_minutes',
                    'before': self.engine.min_hold_minutes,
                    'after': self.engine.min_hold_minutes + 3,
                    'reason': f"Short holds win only {sh_wr:.0%}"
                })

        if len(long_holds) >= 3:
            lh_wr = sum(1 for t in long_holds if t['actual_outcome'] > 0) / len(long_holds)
            if lh_wr > 0.60:
                results['insights'].append(
                    f"[GOOD] PATIENCE PAYS: Long-hold trades (>15 min) have {lh_wr:.0%} win rate. "
                    f"Model benefits from longer holds. Consider INCREASING min_hold_minutes."
                )

    def _analyze_position_sizing(self, trades, results):
        """Check if larger positions perform worse (over-concentration risk)."""
        buys = [t for t in trades if t.get('action') == 'BUY' and t.get('cost')]
        if len(buys) < 5:
            return

        costs = [t['cost'] for t in buys]
        median_cost = float(np.median(costs))

        large_trades = [t for t in trades if t.get('action') == 'SELL'
                       and t.get('actual_outcome') is not None
                       and t.get('proceeds', 0) > median_cost * 1.5]
        small_trades = [t for t in trades if t.get('action') == 'SELL'
                       and t.get('actual_outcome') is not None
                       and t.get('proceeds', 0) < median_cost * 0.5]

        if len(large_trades) >= 3:
            lt_wr = sum(1 for t in large_trades if t['actual_outcome'] > 0) / len(large_trades)
            if lt_wr < 0.40:
                results['insights'].append(
                    f"[ALERT] OVER-SIZING RISK: Large positions win only {lt_wr:.0%}. "
                    f"Model over-allocates to weak setups. Fix: REDUCE max_position_pct from "
                    f"{self.engine.max_position_pct:.0%} to {max(0.05, self.engine.max_position_pct - 0.03):.0%}."
                )
                results['parameter_changes'].append({
                    'param': 'max_position_pct',
                    'before': self.engine.max_position_pct,
                    'after': max(0.05, self.engine.max_position_pct - 0.03),
                    'reason': f"Large positions win only {lt_wr:.0%}"
                })

    def _analyze_threshold_effectiveness(self, sells, results):
        """Evaluate if current buy/sell thresholds are optimal."""
        if len(sells) < 5:
            return

        total = len(sells)
        wins = sum(1 for t in sells if t['actual_outcome'] > 0)
        wr = wins / total

        buy_thr = self.engine.decision_thresholds['buy_prob']
        sell_thr = self.engine.decision_thresholds['sell_prob']

        # Analyze the distribution of outcomes
        outcomes = [t['actual_outcome'] for t in sells]
        avg_outcome = float(np.mean(outcomes))
        median_outcome = float(np.median(outcomes))

        if wr < 0.45:
            new_buy = min(0.78, buy_thr + 0.03)
            new_sell = max(0.22, sell_thr - 0.03)
            results['insights'].append(
                f"[ALERT] THRESHOLD RECALIBRATION NEEDED: Overall win rate {wr:.0%} is below 45%. "
                f"Avg outcome: {avg_outcome:+.4f}, Median: {median_outcome:+.4f}. "
                f"AUTO-FIX: Tightening buy threshold {buy_thr:.2f} → {new_buy:.2f}, "
                f"sell threshold {sell_thr:.2f} → {new_sell:.2f}. "
                f"This forces the model to only take higher-probability setups."
            )
            results['parameter_changes'].append({
                'param': 'buy_prob',
                'before': buy_thr,
                'after': new_buy,
                'reason': f"Win rate {wr:.0%} below 45%"
            })
            results['parameter_changes'].append({
                'param': 'sell_prob',
                'before': sell_thr,
                'after': new_sell,
                'reason': f"Win rate {wr:.0%} below 45%"
            })
        elif wr > 0.62:
            new_buy = max(0.52, buy_thr - 0.02)
            new_sell = min(0.48, sell_thr + 0.02)
            results['insights'].append(
                f"[GOOD] THRESHOLDS TOO CONSERVATIVE: Win rate {wr:.0%} is very high but may be "
                f"missing profitable trades. Avg outcome: {avg_outcome:+.4f}. "
                f"AUTO-FIX: Relaxing buy threshold {buy_thr:.2f} → {new_buy:.2f} "
                f"to capture more opportunities while maintaining edge."
            )
            results['parameter_changes'].append({
                'param': 'buy_prob',
                'before': buy_thr,
                'after': new_buy,
                'reason': f"Win rate {wr:.0%} suggests room to capture more trades"
            })
        else:
            results['insights'].append(
                f"[OK] THRESHOLDS ADEQUATE: Win rate {wr:.0%} with avg outcome {avg_outcome:+.4f}. "
                f"Current thresholds (buy>{buy_thr:.2f}, sell<{sell_thr:.2f}) are performing within tolerance."
            )

    def _analyze_volatility_errors(self, trades, state, results):
        """Find if the model makes worse decisions during high-vol periods."""
        # Use reasoning text to infer volatility context
        high_vol_sells = []
        low_vol_sells = []

        for t in trades:
            if t.get('action') != 'SELL' or t.get('actual_outcome') is None:
                continue
            reasoning = t.get('reasoning', '')
            if 'High volatility' in reasoning or 'vol' in reasoning.lower() and '0.3' in reasoning:
                high_vol_sells.append(t)
            elif 'Low volatility' in reasoning:
                low_vol_sells.append(t)

        if len(high_vol_sells) >= 3:
            hv_wr = sum(1 for t in high_vol_sells if t['actual_outcome'] > 0) / len(high_vol_sells)
            hv_pnl = sum(t.get('actual_outcome_dollar', 0) for t in high_vol_sells)
            if hv_wr < 0.40:
                results['insights'].append(
                    f"[ALERT] HIGH-VOL WEAKNESS: In high-volatility conditions, model wins only {hv_wr:.0%} "
                    f"(${hv_pnl:+,.2f} across {len(high_vol_sells)} trades). "
                    f"Algorithm fix: When vol > 0.30, MULTIPLY buy_prob threshold by 1.15 "
                    f"and HALVE position allocation. Adding vol-adjusted dynamic thresholds."
                )

    def _analyze_loss_patterns(self, sells, results):
        """Detect consecutive loss streaks and common losing characteristics."""
        if len(sells) < 5:
            return

        # Find max consecutive losses
        max_streak = 0
        current_streak = 0
        for t in sells:
            if t['actual_outcome'] <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        if max_streak >= 4:
            results['error_patterns'].append({
                'type': 'loss_streak',
                'max_consecutive_losses': max_streak,
            })
            results['insights'].append(
                f"[ALERT] LOSS STREAK DETECTED: {max_streak} consecutive losing trades observed. "
                f"This suggests the model doesn't adapt mid-session when conditions shift. "
                f"Algorithm fix: Implement CIRCUIT BREAKER — after 3 consecutive losses, "
                f"PAUSE new buys for 10 steps and TIGHTEN thresholds by +0.05 temporarily. "
                f"Also ADD a momentum-shift detector that flags regime changes intra-simulation."
            )

        # Analyze which symbols appear most in losses
        loss_symbols: Dict[str, int] = {}
        for t in sells:
            if t['actual_outcome'] <= 0:
                sym = t['symbol']
                loss_symbols[sym] = loss_symbols.get(sym, 0) + 1

        if loss_symbols:
            worst_sym = max(loss_symbols, key=loss_symbols.get)
            worst_count = loss_symbols[worst_sym]
            if worst_count >= 3:
                results['insights'].append(
                    f"[ALERT] REPEAT LOSER: {worst_sym} lost {worst_count} times. "
                    f"Model keeps re-entering a losing position. "
                    f"Fix: ADD per-symbol loss limit — after 2 losses on same symbol, "
                    f"BLACKLIST it for remainder of simulation."
                )

    def _auto_apply_improvements(self, results):
        """Auto-apply the parameter changes to the engine."""
        applied = []
        for change in results['parameter_changes']:
            param = change['param']
            new_val = change['after']
            old_val = change['before']

            try:
                if param == 'buy_prob':
                    self.engine.decision_thresholds['buy_prob'] = new_val
                    applied.append(f"buy_prob: {old_val:.3f} → {new_val:.3f}")
                elif param == 'sell_prob':
                    self.engine.decision_thresholds['sell_prob'] = new_val
                    applied.append(f"sell_prob: {old_val:.3f} → {new_val:.3f}")
                elif param == 'min_hold_minutes':
                    self.engine.min_hold_minutes = int(new_val)
                    applied.append(f"min_hold_minutes: {old_val} → {int(new_val)}")
                elif param == 'max_position_pct':
                    self.engine.max_position_pct = new_val
                    applied.append(f"max_position_pct: {old_val:.2%} → {new_val:.2%}")
                # Sector-specific adjustments stored for reference but don't change global threshold
                elif param.startswith('sector_threshold_adj_'):
                    applied.append(f"{param}: noted (sector-specific, applied contextually)")
                elif param == 'buy_prob_regime_adj':
                    self.engine.decision_thresholds['buy_prob'] = new_val
                    applied.append(f"buy_prob (regime adj): {old_val:.3f} → {new_val:.3f}")
            except Exception as e:
                results['insights'].append(f"[WARN] Failed to auto-apply {param}: {e}")

        if applied:
            results['insights'].append(
                f"[FIX] AUTO-APPLIED {len(applied)} PARAMETER CHANGES: " + " | ".join(applied)
            )
        else:
            results['insights'].append(
                "[OK] No parameter changes needed — model is performing within acceptable bounds."
            )

    def _grade_trades(self, sells: List[Dict]) -> List[Dict]:
        """Grade each completed trade."""
        grades = []
        for t in sells:
            outcome = t.get('actual_outcome', 0)
            confidence = t.get('confidence', 0.5)
            symbol = t.get('symbol', '?')
            action = t.get('action', '?')

            # P&L Quality
            if outcome > 0:
                pnl_score = min(100, 60 + outcome * 100 * 8)
            else:
                pnl_score = max(0, 50 + outcome * 100 * 10)

            # Confidence Calibration
            if outcome > 0:
                cal_score = 50 + confidence * 50
            else:
                cal_score = max(0, 70 - confidence * 80)

            # Risk/Reward
            rr_score = min(100, 50 + abs(outcome) * 100 * 5) if outcome > 0 else max(0, 50 - abs(outcome) * 100 * 5)

            # Sizing
            sizing_score = min(100, 60 + confidence * 40) if confidence >= 0.3 else 40 + confidence * 60

            composite = pnl_score * 0.40 + cal_score * 0.25 + rr_score * 0.20 + sizing_score * 0.15

            # Letter grade
            if composite >= 97: lg = "A+"
            elif composite >= 93: lg = "A"
            elif composite >= 90: lg = "A-"
            elif composite >= 87: lg = "B+"
            elif composite >= 83: lg = "B"
            elif composite >= 80: lg = "B-"
            elif composite >= 77: lg = "C+"
            elif composite >= 73: lg = "C"
            elif composite >= 70: lg = "C-"
            elif composite >= 67: lg = "D+"
            elif composite >= 63: lg = "D"
            elif composite >= 60: lg = "D-"
            else: lg = "F"

            grades.append({
                "symbol": symbol, "action": action, "letter_grade": lg,
                "composite": round(composite, 1),
                "pnl_quality": round(pnl_score, 1),
                "confidence_calibration": round(cal_score, 1),
                "risk_reward": round(rr_score, 1),
                "sizing_discipline": round(sizing_score, 1),
            })
        return grades

    def _grade_full_simulation(self, sells: List[Dict], trade_outcomes: List[float],
                                state: Dict) -> Dict:
        """Grade the overall simulation."""
        total_trades = len(trade_outcomes)
        if total_trades == 0:
            return self._grade_simulation_empty()

        wins = [r for r in trade_outcomes if r > 0]
        losses = [r for r in trade_outcomes if r <= 0]
        total_return = sum(trade_outcomes)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0
        sharpe = float((np.mean(trade_outcomes) / (np.std(trade_outcomes) + 1e-10)) * np.sqrt(252)) if total_trades > 1 else 0

        cum = np.cumsum(trade_outcomes)
        running_max = np.maximum.accumulate(cum)
        max_dd = float(np.min(cum - running_max)) if len(cum) > 0 else 0

        symbols_analyzed = len(set(t.get('symbol', '') for t in sells))

        scores = {}

        # Returns Quality
        ret_pct = total_return * 100
        scores["returns_quality"] = min(100, max(0, 40 + ret_pct * 8))

        # Decision Accuracy
        scores["decision_accuracy_score"] = min(100, max(0, win_rate * 100 * 1.3 - 15))

        # Risk Management
        dd_pct = abs(max_dd) * 100
        risk_score = max(0, 100 - dd_pct * 5)
        if losses and wins:
            loss_control = min(1.5, avg_win / (float(np.mean([abs(l) for l in losses])) + 1e-10))
            risk_score = min(100, risk_score + loss_control * 10)
        scores["risk_management"] = risk_score

        # Sharpe quality
        scores["risk_adjusted"] = min(100, max(0, 40 + sharpe * 20))

        # Coverage
        if symbols_analyzed > 0:
            selectivity = total_trades / max(symbols_analyzed, 1)
            coverage_score = 85 if 0.15 <= selectivity <= 0.50 else 60 if selectivity < 0.15 else 65
        else:
            coverage_score = 50
        if total_trades >= 15: coverage_score = min(100, coverage_score + 10)
        elif total_trades < 5: coverage_score = max(30, coverage_score - 20)
        scores["coverage_discipline"] = coverage_score

        # Learning value
        learning_score = 60
        if wins and losses: learning_score += 20
        if len(trade_outcomes) >= 5 and np.std(trade_outcomes) > 0.01: learning_score += 10
        if total_trades >= 20: learning_score += 10
        scores["learning_value"] = min(100, learning_score)

        composite = (scores["returns_quality"] * 0.25 + scores["decision_accuracy_score"] * 0.20 +
                     scores["risk_management"] * 0.20 + scores["risk_adjusted"] * 0.15 +
                     scores["coverage_discipline"] * 0.10 + scores["learning_value"] * 0.10)

        if composite >= 97: lg = "A+"
        elif composite >= 93: lg = "A"
        elif composite >= 90: lg = "A-"
        elif composite >= 87: lg = "B+"
        elif composite >= 83: lg = "B"
        elif composite >= 80: lg = "B-"
        elif composite >= 77: lg = "C+"
        elif composite >= 73: lg = "C"
        elif composite >= 70: lg = "C-"
        elif composite >= 67: lg = "D+"
        elif composite >= 63: lg = "D"
        elif composite >= 60: lg = "D-"
        else: lg = "F"

        commentary = []
        if lg.startswith("A"): commentary.append("Outstanding simulation — model demonstrated strong edge.")
        elif lg.startswith("B"): commentary.append("Good simulation — competent with room for improvement.")
        elif lg.startswith("C"): commentary.append("Average simulation — calibration improvements needed.")
        elif lg.startswith("D"): commentary.append("Below average — significant weaknesses detected.")
        else: commentary.append("Poor simulation — major recalibration required.")

        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)
        commentary.append(f"Strongest: {strongest.replace('_', ' ').title()} ({scores[strongest]:.0f}/100)")
        commentary.append(f"Weakest: {weakest.replace('_', ' ').title()} ({scores[weakest]:.0f}/100)")

        scores["composite"] = round(composite, 1)
        scores["letter_grade"] = lg
        scores["commentary"] = commentary
        return scores

    def _grade_simulation_empty(self) -> Dict:
        return {
            "letter_grade": "F", "composite": 0,
            "returns_quality": 0, "decision_accuracy_score": 0,
            "risk_management": 0, "risk_adjusted": 0,
            "coverage_discipline": 0, "learning_value": 0,
            "commentary": ["No trades executed — cannot grade."],
        }

class MarketSimulationEngine:
    """Ultra-realistic market simulation engine with configurable duration."""
    
    def __init__(self, date_str: str = None):
        self._fallback_analyzer = _FallbackUnbiasedAnalyzer()
        self._use_fallback = True

        if os.getenv("SIMULATION_USE_FULL_ANALYZER", "0") == "1":
            try:
                from unbiased_market_analyzer import UnbiasedMarketAnalyzer
                self.unbiased_analyzer = UnbiasedMarketAnalyzer()
                self._use_fallback = False
            except Exception as e:
                self.unbiased_analyzer = self._fallback_analyzer
        else:
            self.unbiased_analyzer = self._fallback_analyzer

        # Quant ensemble model integration
        self._quant_ensemble = None
        if HAS_QUANT_ENSEMBLE:
            try:
                self._quant_ensemble = get_quant_ensemble()
            except Exception:
                pass

        self.credibility_engine = None
        try:
            if SourceCredibilityEngine is not None:
                self.credibility_engine = SourceCredibilityEngine()
        except Exception:
            pass
        
        self.timeframe_engine = None
        try:
            if TimeframeAnalysisEngine is not None:
                self.timeframe_engine = TimeframeAnalysisEngine()
        except Exception:
            pass
        
        self.db_path = self.get_current_db_path()
        self.setup_simulation_database()
        
        #  CONFIGURABLE SIMULATION PARAMETERS 
        self.simulation_duration = timedelta(hours=2)  # User can change
        self.time_step = timedelta(minutes=1)
        self.universe_size = None
        
        # News generation: much higher frequency for thousands of events
        self.news_frequency = {
            NewsType.EARNINGS: 0.08,
            NewsType.ECONOMIC_DATA: 0.06,
            NewsType.GEOPOLITICAL: 0.04,
            NewsType.CENTRAL_BANK: 0.03,
            NewsType.CORPORATE_ACTION: 0.05,
            NewsType.SECTOR_NEWS: 0.07,
            NewsType.REGULATORY: 0.03,
            NewsType.MARKET_STRUCTURE: 0.02,
            NewsType.ANALYST_RATING: 0.08,
            NewsType.INSIDER_TRADE: 0.04,
            NewsType.MACRO_INDICATOR: 0.05,
            NewsType.COMMODITY_SHOCK: 0.03,
            NewsType.CURRENCY_EVENT: 0.04,
            NewsType.TECHNICAL_SIGNAL: 0.06,
            NewsType.SOCIAL_SENTIMENT: 0.07,
        }

        # ...existing code for decision_thresholds, model_tuning, volatility_regimes...
        self.decision_thresholds = {
            "buy_prob": 0.63,
            "sell_prob": 0.37
        }
        self.model_tuning = {"last_accuracy": 0.0, "tuning_steps": 0}

        self.volatility_regimes = {
            MarketRegime.LOW_VOLATILITY: {'base_vol': 0.15, 'vol_of_vol': 0.3},
            MarketRegime.HIGH_VOLATILITY: {'base_vol': 0.35, 'vol_of_vol': 0.8},
            MarketRegime.CRISIS: {'base_vol': 0.60, 'vol_of_vol': 1.2},
            MarketRegime.BULL_MARKET: {'base_vol': 0.18, 'vol_of_vol': 0.4},
            MarketRegime.BEAR_MARKET: {'base_vol': 0.25, 'vol_of_vol': 0.6},
            MarketRegime.SIDEWAYS: {'base_vol': 0.12, 'vol_of_vol': 0.2},
            MarketRegime.RECOVERY: {'base_vol': 0.20, 'vol_of_vol': 0.5},
            MarketRegime.SECTOR_ROTATION: {'base_vol': 0.22, 'vol_of_vol': 0.5},
            MarketRegime.RISK_ON: {'base_vol': 0.20, 'vol_of_vol': 0.4},
            MarketRegime.RISK_OFF: {'base_vol': 0.30, 'vol_of_vol': 0.7},
        }

        self.min_hold_minutes = 8
        self.trade_cooldown_minutes = 5
        self.step_trade_cap = 12
        self.max_trades_per_symbol = 3
        self.max_position_pct = 0.08
        self.initial_capital = 100_000.0
        self.max_total_invested = 100_000.0
        self.profit_take_pct = 0.018
        self.stop_loss_pct = 0.010
        self.trailing_stop_pct = 0.007

        self._symbol_drift: Dict[str, float] = {}
        self._symbol_trend_regime: Dict[str, str] = {}
        self._last_trades: List[Dict] = []
        self._last_news: List[SimulatedNewsEvent] = []

        #  MACRO EVENT SYSTEM for comprehensive simulation 
        self._macro_events_schedule: List[Dict] = []

        self._apply_persisted_params()

    def _apply_persisted_params(self):
        """Apply any previously learned parameter adjustments."""
        learned = _load_learned_params()
        if not learned:
            return
        if 'buy_prob' in learned:
            self.decision_thresholds['buy_prob'] = float(learned['buy_prob'])
        if 'sell_prob' in learned:
            self.decision_thresholds['sell_prob'] = float(learned['sell_prob'])
        if 'min_hold_minutes' in learned:
            self.min_hold_minutes = int(learned['min_hold_minutes'])
        if 'max_position_pct' in learned:
            self.max_position_pct = float(learned['max_position_pct'])
        if 'tuning_steps' in learned:
            self.model_tuning['tuning_steps'] = int(learned['tuning_steps'])

    def _persist_current_params(self):
        """Save current parameters to disk after learning."""
        params = {
            'buy_prob': self.decision_thresholds['buy_prob'],
            'sell_prob': self.decision_thresholds['sell_prob'],
            'min_hold_minutes': self.min_hold_minutes,
            'max_position_pct': self.max_position_pct,
            'tuning_steps': self.model_tuning.get('tuning_steps', 0),
            'last_updated': datetime.now().isoformat(),
        }
        _save_learned_params(params)

    def get_current_db_path(self) -> str:
        """Get the path to the current day's simulation database."""
        today = datetime.now().strftime('%Y_%m_%d')
        return f"octavian_simulations_{today}.db"

    def setup_simulation_database(self):
        """Initialize the SQLite database for storing simulation results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS simulations (
                    simulation_id TEXT PRIMARY KEY,
                    start_time DATETIME,
                    end_time DATETIME,
                    market_regime TEXT,
                    total_decisions INTEGER,
                    successful_decisions INTEGER,
                    failed_decisions INTEGER,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    performance_data TEXT,
                    insights_data TEXT
                );

                CREATE TABLE IF NOT EXISTS trading_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT,
                    timestamp DATETIME,
                    symbol TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    confidence REAL,
                    actual_outcome REAL,
                    reasoning TEXT
                );

                CREATE TABLE IF NOT EXISTS simulated_news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT,
                    timestamp DATETIME,
                    headline TEXT,
                    news_type TEXT,
                    market_impact_score REAL,
                    content TEXT,
                    affected_symbols TEXT
                );

                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT,
                    insight_type TEXT,
                    insight_text TEXT,
                    timestamp DATETIME
                );
            ''')
            conn.commit()

    async def _get_analysis_safe(self, symbol: str):
        """Safely obtain analysis with fallback on failure."""
        if self._use_fallback:
            return await self._fallback_analyzer.analyze_unbiased(symbol)
        try:
            analysis = await self.unbiased_analyzer.analyze_unbiased(symbol)
            if not hasattr(analysis, "profit_probability"):
                raise ValueError("invalid_analysis")
            return analysis
        except Exception as e:
            print(f"Analysis fallback for {symbol}: {e}")
            return await self._fallback_analyzer.analyze_unbiased(symbol)

    def set_simulation_config(self, duration_minutes: int = 120, universe_size: int = 80):
        """Allow user to configure simulation duration and universe size."""
        self.simulation_duration = timedelta(minutes=duration_minutes)
        self.universe_size = universe_size

    # 
    # PRICE SIMULATION — Persistent trends, mean reversion, regime effects
    # 

    def _initialize_symbol_trends(self, universe: List[str], regime: MarketRegime):
        """Assign each symbol a persistent drift/trend for this simulation."""
        regime_bias = {
            MarketRegime.BULL_MARKET: 0.0003,
            MarketRegime.BEAR_MARKET: -0.0003,
            MarketRegime.CRISIS: -0.0006,
            MarketRegime.RECOVERY: 0.0002,
            MarketRegime.RISK_ON: 0.0002,
            MarketRegime.RISK_OFF: -0.0002,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.LOW_VOLATILITY: 0.0001,
            MarketRegime.SECTOR_ROTATION: 0.0,
        }.get(regime, 0.0)

        for sym in universe:
            # Each symbol gets its own persistent drift centered around regime bias
            self._symbol_drift[sym] = regime_bias + random.gauss(0, 0.0004)
            # Assign trend regime: trending, mean_reverting, or choppy
            r = random.random()
            if r < 0.45:
                self._symbol_trend_regime[sym] = "trending"
            elif r < 0.75:
                self._symbol_trend_regime[sym] = "mean_reverting"
            else:
                self._symbol_trend_regime[sym] = "choppy"

    def _rotate_some_trends(self, universe: List[str], regime: MarketRegime):
        """Periodically rotate some symbols' trend direction (regime shifts)."""
        num_to_rotate = max(1, len(universe) // 10)
        to_rotate = random.sample(universe, min(num_to_rotate, len(universe)))
        for sym in to_rotate:
            old_drift = self._symbol_drift.get(sym, 0.0)
            # Flip or dampen the drift
            if random.random() < 0.4:
                self._symbol_drift[sym] = -old_drift * random.uniform(0.5, 1.5)
            else:
                self._symbol_drift[sym] = old_drift * random.uniform(0.3, 0.8) + random.gauss(0, 0.0002)
            # Small chance to change trend regime
            if random.random() < 0.25:
                self._symbol_trend_regime[sym] = random.choice(["trending", "mean_reverting", "choppy"])

    def _get_previous_price(self, symbol: str, timestamp: datetime,
                            state: Dict[str, Any]) -> Optional[SimulatedMarketData]:
        """Get the most recent price for a symbol from state history."""
        prices = self._fallback_analyzer._price_history.get(symbol, [])
        if prices:
            # Create a minimal SimulatedMarketData-like object
            p = prices[-1]
            return SimulatedMarketData(
                symbol=symbol, timestamp=timestamp,
                open_price=p, high_price=p, low_price=p, close_price=p,
                volume=0, volatility=0, bid_ask_spread=0, market_cap=0, news_events=[]
            )
        return None

    def _generate_realistic_price_movement(self, prev_price: float,
                                            regime_params: Dict[str, float],
                                            timestamp: datetime, symbol: str,
                                            state: Dict[str, Any]) -> Dict[str, float]:
        """Generate realistic OHLC price movement with persistent trends."""
        base_vol = regime_params.get('base_vol', 0.20)
        vol_of_vol = regime_params.get('vol_of_vol', 0.5)

        # Per-minute volatility
        annual_vol = base_vol * (1 + random.gauss(0, vol_of_vol * 0.1))
        minute_vol = annual_vol / np.sqrt(252 * 390)  # ~390 trading minutes/day

        # Persistent drift for this symbol
        drift = self._symbol_drift.get(symbol, 0.0)
        trend_regime = self._symbol_trend_regime.get(symbol, "trending")

        # Generate return
        noise = random.gauss(0, minute_vol)

        if trend_regime == "trending":
            # Strong drift, moderate noise
            ret = drift + noise * 0.8
        elif trend_regime == "mean_reverting":
            # Weaker drift, mean-reversion toward recent average
            prices = self._fallback_analyzer._price_history.get(symbol, [])
            if len(prices) >= 10:
                ma = np.mean(prices[-10:])
                reversion = (ma - prev_price) / prev_price * 0.02
                ret = drift * 0.3 + reversion + noise * 0.7
            else:
                ret = drift * 0.5 + noise
        else:  # choppy
            ret = random.gauss(0, minute_vol * 1.3)

        # Time-of-day effect (more volatile at open/close)
        tod_effect = self._calculate_time_of_day_effect(timestamp)
        ret *= (1 + tod_effect * 0.3)

        # News impact
        news_impact = self._calculate_news_impact(symbol, state)
        ret += news_impact

        # Clamp extreme moves
        ret = max(-0.05, min(0.05, ret))

        close = prev_price * (1 + ret)
        close = max(0.01, close)

        # Generate OHLC from close
        intrabar_vol = abs(ret) + minute_vol * 0.5
        high = close * (1 + random.uniform(0, intrabar_vol))
        low = close * (1 - random.uniform(0, intrabar_vol))
        open_price = prev_price * (1 + random.gauss(0, minute_vol * 0.3))

        high = max(high, open_price, close)
        low = min(low, open_price, close)
        low = max(0.01, low)

        return {
            'open': round(open_price, 4),
            'high': round(high, 4),
            'low': round(low, 4),
            'close': round(close, 4),
            'volatility': annual_vol,
        }

    def _calculate_time_of_day_effect(self, timestamp: datetime) -> float:
        """Simulate U-shaped intraday volatility."""
        minute = timestamp.hour * 60 + timestamp.minute
        # Market hours roughly 9:30-16:00 = minutes 570-960
        market_open = 570
        market_close = 960
        mid = (market_open + market_close) / 2

        if minute < market_open or minute > market_close:
            return 0.0
        # U-shape: high at open/close, low at midday
        dist_from_mid = abs(minute - mid) / (market_close - market_open) * 2
        return dist_from_mid * 0.5

    def _calculate_news_impact(self, symbol: str, state: Dict[str, Any]) -> float:
        """Calculate cumulative news impact on a symbol."""
        impact = 0.0
        recent_news = state.get('news_events', [])[-20:]  # Last 20 news events
        for event in recent_news:
            if symbol in event.affected_symbols:
                # Decay impact over time
                impact += event.sentiment_score * event.market_impact_score * 0.001
        return impact

    def _calculate_momentum_effect(self, symbol: str) -> float:
        """Calculate momentum effect from recent price history."""
        prices = self._fallback_analyzer._price_history.get(symbol, [])
        if len(prices) < 5:
            return 0.0
        ret_5 = (prices[-1] / prices[-5] - 1) if prices[-5] > 0 else 0
        return ret_5 * 0.01

    def _calculate_mean_reversion_effect(self, symbol: str) -> float:
        """Calculate mean reversion pull."""
        prices = self._fallback_analyzer._price_history.get(symbol, [])
        if len(prices) < 20:
            return 0.0
        ma20 = np.mean(prices[-20:])
        current = prices[-1]
        z = (current - ma20) / (np.std(prices[-20:]) + 1e-8)
        if abs(z) > 2.0:
            return -z * 0.0005
        return 0.0

    def _generate_realistic_volume(self, symbol: str, price_data: Dict[str, float],
                                    regime: MarketRegime, state: Dict[str, Any]) -> int:
        """Generate realistic volume."""
        # Base volume depends on symbol type
        if '-USD' in symbol:
            base_vol = random.randint(50_000, 500_000)
        elif '=F' in symbol:
            base_vol = random.randint(10_000, 200_000)
        elif '/' in symbol:
            base_vol = random.randint(100_000, 1_000_000)
        else:
            base_vol = random.randint(500_000, 20_000_000)

        # Regime multiplier
        regime_mult = {
            MarketRegime.CRISIS: 2.5, MarketRegime.HIGH_VOLATILITY: 1.8,
            MarketRegime.LOW_VOLATILITY: 0.6, MarketRegime.BULL_MARKET: 1.2,
        }.get(regime, 1.0)

        # Price change multiplier (bigger moves = more volume)
        price_change = abs(price_data['close'] - price_data['open']) / (price_data['open'] + 1e-8)
        change_mult = 1 + price_change * 10

        volume = int(base_vol * regime_mult * change_mult * random.uniform(0.7, 1.3))
        return max(1000, volume)

    def _generate_bid_ask_spread(self, price: float, regime: MarketRegime) -> float:
        """Generate realistic bid-ask spread."""
        base_spread_pct = 0.0005  # 5 bps
        if regime in (MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY):
            base_spread_pct *= 3
        elif regime == MarketRegime.LOW_VOLATILITY:
            base_spread_pct *= 0.5
        spread = price * base_spread_pct * random.uniform(0.5, 2.0)
        return round(spread, 4)

    # 
    # NEWS GENERATION — High-frequency realistic events
    # 

    def _generate_news_events(self, timestamp: datetime, regime: MarketRegime) -> List[SimulatedNewsEvent]:
        """Generate realistic news events with higher frequency."""
        events = []
        for news_type, base_frequency in self.news_frequency.items():
            regime_multiplier = self._get_news_frequency_multiplier(regime, news_type)
            adjusted_frequency = base_frequency * regime_multiplier
            if random.random() < adjusted_frequency:
                event = self._create_news_event(timestamp, news_type, regime)
                events.append(event)
            # Cluster effect: small chance of second event
            if random.random() < adjusted_frequency * 0.25:
                event = self._create_news_event(timestamp, news_type, regime)
                events.append(event)
        return events

    def _get_news_frequency_multiplier(self, regime: MarketRegime, news_type: NewsType) -> float:
        """Regime-dependent news frequency scaling."""
        multiplier = 1.0
        if regime == MarketRegime.CRISIS:
            multiplier = 2.5
            if news_type in (NewsType.CENTRAL_BANK, NewsType.GEOPOLITICAL, NewsType.MACRO_INDICATOR):
                multiplier *= 1.8
        elif regime == MarketRegime.HIGH_VOLATILITY:
            multiplier = 1.6
        elif regime == MarketRegime.LOW_VOLATILITY:
            multiplier = 0.6
        elif regime == MarketRegime.BULL_MARKET:
            if news_type in (NewsType.EARNINGS, NewsType.ANALYST_RATING):
                multiplier = 1.4
        elif regime == MarketRegime.BEAR_MARKET:
            if news_type in (NewsType.ECONOMIC_DATA, NewsType.INSIDER_TRADE):
                multiplier = 1.6
        elif regime == MarketRegime.SECTOR_ROTATION:
            if news_type == NewsType.SECTOR_NEWS:
                multiplier = 2.2
        elif regime == MarketRegime.RISK_OFF:
            if news_type in (NewsType.COMMODITY_SHOCK, NewsType.CURRENCY_EVENT):
                multiplier = 1.8
        return multiplier

    # 
    # SIMULATION EXECUTION & SCHEDULING
    # 

    def start_simulation_engine(self):
        """Start the simulation engine in a background thread."""
        if hasattr(self, "_simulation_thread") and self._simulation_thread.is_alive():
            return self._simulation_thread
            
        self._running = True
        self._simulation_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._simulation_thread.start()
        print("[OK] Market Simulation Engine started.")
        return self._simulation_thread

    def stop_simulation_engine(self):
        """Stop the simulation engine."""
        self._running = False
        if hasattr(self, "_simulation_thread"):
            self._simulation_thread.join(timeout=1.0)
            print("[STOP] Market Simulation Engine stopped.")

    def _run_scheduler(self):
        """Internal scheduler loop."""
        # Schedule simulations 3 times a day
        schedule.every().day.at("09:30").do(self.run_daily_simulation)
        schedule.every().day.at("12:00").do(self.run_daily_simulation)
        schedule.every().day.at("15:30").do(self.run_daily_simulation)
        
        # Also run one immediately on startup if none ran today
        # (Simplified logic: just run one now for demo/testing if needed, 
        # or relying on user to manually trigger if they want)
        
        while self._running:
            schedule.run_pending()
            time.sleep(1)

    def run_daily_simulation(self):
        """Run a complete daily market simulation."""
        print(f"[START] Starting daily simulation at {datetime.now()}")
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        # 1. Determine daily regime
        regime = random.choice(list(MarketRegime))
        
        # 2. Setup universe
        universe = self._fallback_analyzer.asset_universe["stocks_mega"][:20] # Default subset
        if self.universe_size:
            # Expand universe logic here if needed
            pass
            
        self._initialize_symbol_trends(universe, regime)
        
        # 3. Simulate market step-by-step
        current_time = start_time
        end_time = start_time + self.simulation_duration
        
        state = {
            'trades': [],
            'news_events': [],
            'cash': self.initial_capital,
            'portfolio': {},
            'history': []
        }
        
        while current_time < end_time:
            # Generate price movements for all symbols
            step_prices = {}
            for sym in universe:
                prev_price_data = self._get_previous_price(sym, current_time, state)
                prev_price = prev_price_data.close_price if prev_price_data else 100.0 # Default start
                
                # Get regime params
                regime_params = self.volatility_regimes.get(regime, {})
                
                # Generate OHLC
                ohlc = self._generate_realistic_price_movement(
                    prev_price, regime_params, current_time, sym, state
                )
                
                # Update fallback analyzer history
                self._fallback_analyzer.update_price(sym, ohlc['close'])
                step_prices[sym] = ohlc
            
            # Generate news
            news = self._generate_news_events(current_time, regime)
            state['news_events'].extend(news)
            
            # Execute trading logic (simplified for this engine, usually AI loop goes here)
            # For now we just record the market data evolution
            
            current_time += self.time_step
            
        # 4. Finalize
        self._finalize_simulation(simulation_id, start_time, datetime.now(), regime, state)

    def _finalize_simulation(self, sim_id, start, end, regime, state):
        """Save results and generate insights."""
        # Calculate stats
        total_decisions = len(state['trades'])
        
        # Store in DB
        self._store_simulation_results(sim_id, start, end, regime, state)
        
        # Trigger learning
        learning_engine = SimulationLearningEngine(self)
        results = learning_engine.analyze_and_improve(state, regime)
        
        # Persist learned params
        self._persist_current_params()
        
        print(f"[END] Simulation {sim_id} complete. Grade: {results.get('simulation_grade', {}).get('letter_grade')}")

    def _store_simulation_results(self, sim_id, start, end, regime, state):
        """Store simulation data to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO simulations (simulation_id, start_time, end_time, market_regime, total_decisions)
                    VALUES (?, ?, ?, ?, ?)
                ''', (sim_id, start, end, regime.value, len(state['trades'])))
                conn.commit()
        except Exception as e:
            print(f"[WARN] Failed to store simulation results: {e}")

    def _schedule_macro_events(self, regime):
        """Schedule major events for the simulation."""
        pass # Implement if needed

    def _process_macro_events(self, current_time, state):
        """Check and process scheduled macro events."""
        pass # Implement if needed

    def _create_empty_result(self, sim_id, start, end, regime):
        """Create a placeholder result."""
        return SimulationResult(
            simulation_id=sim_id, start_time=start, end_time=end, market_regime=regime,
            total_decisions=0, successful_decisions=0, failed_decisions=0, total_return=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            key_insights=[], model_improvements=[], missed_opportunities=[], successful_predictions=[],
            market_reactions={}, performance_metrics={}, news_impact_analysis={}, sector_performance={},
            volatility_analysis={}, correlation_analysis={}
        )

    def get_recent_simulations(self, limit: int = 20) -> list[dict]:
        """
        Retrieve recent simulation results from the database.
        Returns a list of dictionaries containing simulation metadata and key metrics.
        """
        import sqlite3
        import os
        
        recent_simulations = []
        
        try:
            # Get database path
            db_path = self.get_current_db_path()
            
            if not os.path.exists(db_path):
                # Try to find any simulation databases
                import glob
                db_files = glob.glob("octavian_simulations_*.db")
                if db_files:
                    db_path = max(db_files)  # Use most recent
                else:
                    return []
            
            # Connect and query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get recent simulations
            cursor.execute("""
                SELECT simulation_id, start_time, end_time, market_regime, 
                       total_return, sharpe_ratio, max_drawdown, win_rate, 
                       total_decisions, successful_decisions, failed_decisions
                FROM simulations 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            for row in rows:
                recent_simulations.append({
                    "simulation_id": row[0],
                    "start_time": row[1],
                    "end_time": row[2],
                    "market_regime": row[3],
                    "total_return": row[4] if row[4] is not None else 0.0,
                    "sharpe_ratio": row[5] if row[5] is not None else 0.0,
                    "max_drawdown": row[6] if row[6] is not None else 0.0,
                    "win_rate": row[7] if row[7] is not None else 0.0,
                    "total_decisions": row[8] if row[8] is not None else 0,
                    "successful_decisions": row[9] if row[9] is not None else 0,
                    "failed_decisions": row[10] if row[10] is not None else 0,
                })
            
            conn.close()
            
        except Exception as e:
            # If database retrieval fails, return empty list
            pass
        
        return recent_simulations

