"""
Octavian Quant Ensemble Model
Momentum + Mean-Reversion + Volatility Regime + Stat-Arb sub-models
with Bayesian dynamic weighting.

Author: APB - Octavian Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings("ignore")


@dataclass
class QuantSignal:
    direction: str
    probability: float
    expected_return: float
    confidence: float
    sharpe_estimate: float
    sub_model_signals: Dict[str, Dict[str, float]]
    reasoning: List[str]
    risk_metrics: Dict[str, float]
    optimal_position_size: float
    stop_loss_pct: float
    take_profit_pct: float


class QuantSubModel:
    def __init__(self, name: str, initial_weight: float):
        self.name = name
        self.weight = initial_weight
        self.initial_weight = initial_weight
        self.track_record: deque = deque(maxlen=200)

    def record_outcome(self, prediction: float, actual_return: float):
        self.track_record.append((prediction, actual_return))
        self._update_weight()

    def _update_weight(self):
        if len(self.track_record) < 10:
            return
        recent = list(self.track_record)[-50:]
        correct = sum(1 for pred, actual in recent
                      if (pred > 0.5 and actual > 0) or (pred < 0.5 and actual < 0))
        accuracy = correct / len(recent)
        ratio = accuracy / 0.5
        self.weight = np.clip(self.initial_weight * (0.5 + 0.5 * ratio), 0.05, 0.45)

    def get_accuracy(self, lookback: int = 50) -> float:
        recent = list(self.track_record)[-lookback:] if self.track_record else []
        if not recent:
            return 0.5
        correct = sum(1 for pred, actual in recent
                      if (pred > 0.5 and actual > 0) or (pred < 0.5 and actual < 0))
        return correct / len(recent)


class MomentumModel(QuantSubModel):
    def __init__(self):
        super().__init__("momentum", initial_weight=0.28)

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[float, List[str]]:
        n = len(prices)
        if n < 10:
            return 0.5, ["Insufficient data for momentum"]
        reasoning = []
        scores = []
        for period, w in [(3, 0.35), (5, 0.25), (10, 0.20), (20, 0.12), (40, 0.08)]:
            if n >= period + 1:
                ret = prices[-1] / prices[-period] - 1
                scores.append(ret * w * 100)
        lookback = min(n, 40)
        x = np.arange(lookback)
        slope = np.polyfit(x, prices[-lookback:], 1)[0]
        norm_slope = slope / (prices[-lookback:].mean() + 1e-8)
        scores.append(norm_slope * 500)
        if n >= 10:
            mom_recent = prices[-1] / prices[-3] - 1 if n >= 3 else 0
            mom_older = prices[-5] / prices[-8] - 1 if n >= 8 else 0
            scores.append((mom_recent - mom_older) * 200)
        if volumes is not None and len(volumes) >= 20:
            vol_ratio = np.mean(volumes[-5:]) / (np.mean(volumes[-20:]) + 1e-8)
            ret_5 = prices[-1] / prices[-5] - 1 if n >= 5 else 0
            if vol_ratio > 1.3 and ret_5 > 0:
                scores.append(0.15)
                reasoning.append(f"Volume confirms uptrend ({vol_ratio:.1f}x)")
            elif vol_ratio > 1.3 and ret_5 < 0:
                scores.append(-0.15)
                reasoning.append(f"Volume confirms downtrend ({vol_ratio:.1f}x)")
        raw = np.sum(scores) if scores else 0
        prob = float(np.clip(0.5 + np.tanh(raw) * 0.38, 0.08, 0.92))
        ret_3 = (prices[-1] / prices[-3] - 1) * 100 if n >= 3 else 0
        ret_10 = (prices[-1] / prices[-10] - 1) * 100 if n >= 10 else 0
        reasoning.insert(0, f"MOM 3d={ret_3:+.2f}% 10d={ret_10:+.2f}% slope={norm_slope*100:.3f}%/bar")
        return prob, reasoning


class MeanReversionModel(QuantSubModel):
    def __init__(self):
        super().__init__("mean_reversion", initial_weight=0.22)

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[float, List[str]]:
        n = len(prices)
        if n < 20:
            return 0.5, ["Insufficient data for mean reversion"]
        reasoning = []
        ma20 = np.mean(prices[-20:])
        std20 = np.std(prices[-20:])
        z_score = (prices[-1] - ma20) / (std20 + 1e-8)
        bb_width = (2 * std20) / (ma20 + 1e-8)
        changes = np.diff(prices[-15:])
        gains = changes[changes > 0]
        losses = -changes[changes < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        mr_signal = 0.0
        if abs(z_score) > 2.5:
            mr_signal = -np.tanh((z_score - np.sign(z_score) * 2.0) * 0.8) * 0.35
            reasoning.append(f"EXTREME z={z_score:.2f}")
        elif abs(z_score) > 1.8:
            mr_signal = -np.tanh((z_score - np.sign(z_score) * 1.5) * 0.5) * 0.20
        rsi_signal = 0.0
        if rsi < 25:
            rsi_signal = 0.18
            reasoning.append(f"RSI={rsi:.0f} deeply oversold")
        elif rsi > 75:
            rsi_signal = -0.18
            reasoning.append(f"RSI={rsi:.0f} deeply overbought")
        elif rsi < 35:
            rsi_signal = 0.08
        elif rsi > 65:
            rsi_signal = -0.08
        prob = float(np.clip(0.5 + mr_signal + rsi_signal, 0.08, 0.92))
        reasoning.insert(0, f"MR z={z_score:.2f} RSI={rsi:.0f} BB={bb_width:.4f}")
        return prob, reasoning


class VolatilityRegimeModel(QuantSubModel):
    def __init__(self):
        super().__init__("vol_regime", initial_weight=0.20)

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[float, List[str]]:
        n = len(prices)
        if n < 30:
            return 0.5, ["Insufficient data for vol regime"]
        reasoning = []
        returns = np.diff(np.log(prices))
        vol_5 = np.std(returns[-5:]) * np.sqrt(252)
        vol_20 = np.std(returns[-20:]) * np.sqrt(252)
        vol_60 = np.std(returns[-min(60, len(returns)):]) * np.sqrt(252)
        vol_ratio = vol_5 / (vol_20 + 1e-8)
        vol_trend = vol_20 / (vol_60 + 1e-8)
        if vol_ratio > 1.8:
            regime = "HIGH_VOL_EXPANSION"
            signal = -0.15
            reasoning.append(f"Vol expanding ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            regime = "VOL_COMPRESSION"
            signal = 0.05
            reasoning.append(f"Vol compressed ({vol_ratio:.1f}x)")
        elif vol_trend > 1.3:
            regime = "RISING_VOL"
            signal = -0.08
        elif vol_trend < 0.7:
            regime = "FALLING_VOL"
            signal = 0.08
        else:
            regime = "STABLE"
            signal = 0.0
        prob = float(np.clip(0.5 + signal, 0.15, 0.85))
        reasoning.insert(0, f"VOL {regime} 5d={vol_5:.1%} 20d={vol_20:.1%} ratio={vol_ratio:.2f}")
        return prob, reasoning


class StatArbModel(QuantSubModel):
    def __init__(self):
        super().__init__("stat_arb", initial_weight=0.18)

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[float, List[str]]:
        n = len(prices)
        if n < 40:
            return 0.5, ["Insufficient data for stat arb"]
        reasoning = []
        returns = np.diff(np.log(prices))
        ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
        ac_signal = ac1 * np.sign(returns[-1]) * 0.15
        skew = float(pd.Series(returns[-40:]).skew())
        skew_signal = 0.05 if (skew < -0.5 and returns[-1] > 0) else (-0.05 if (skew > 0.5 and returns[-1] < 0) else 0)
        kurt = float(pd.Series(returns[-40:]).kurtosis())
        kurt_penalty = -0.02 * min(kurt / 5, 1) if kurt > 3 else 0
        close_bias = np.mean(np.sign(returns[-10:])) * 0.08 if n >= 10 else 0
        # Hurst approximation
        half = len(returns) // 2
        if half >= 5:
            r1 = np.max(np.cumsum(returns[:half])) - np.min(np.cumsum(returns[:half]))
            s1 = np.std(returns[:half]) + 1e-8
            r2 = np.max(np.cumsum(returns[half:])) - np.min(np.cumsum(returns[half:]))
            s2 = np.std(returns[half:]) + 1e-8
            rs1, rs2 = r1 / s1, r2 / s2
            hurst = np.clip(np.log(rs2 / (rs1 + 1e-8)) / np.log(2) + 0.5, 0.1, 0.9) if rs1 > 0 and rs2 > 0 else 0.5
        else:
            hurst = 0.5
        hurst_signal = 0.0
        if hurst > 0.6:
            hurst_signal = np.sign(np.mean(returns[-5:])) * (hurst - 0.5) * 0.3
            reasoning.append(f"Hurst={hurst:.2f} trending")
        elif hurst < 0.4:
            hurst_signal = -np.sign(np.mean(returns[-5:])) * (0.5 - hurst) * 0.3
            reasoning.append(f"Hurst={hurst:.2f} mean-reverting")
        total = ac_signal + skew_signal + kurt_penalty + close_bias + hurst_signal
        prob = float(np.clip(0.5 + np.tanh(total) * 0.30, 0.10, 0.90))
        reasoning.insert(0, f"STAT ac1={ac1:.3f} skew={skew:.2f} kurt={kurt:.1f} hurst={hurst:.2f}")
        return prob, reasoning


class QuantEnsembleModel:
    def __init__(self):
        self.momentum = MomentumModel()
        self.mean_reversion = MeanReversionModel()
        self.vol_regime = VolatilityRegimeModel()
        self.stat_arb = StatArbModel()
        self.sub_models = [self.momentum, self.mean_reversion, self.vol_regime, self.stat_arb]
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(m.weight for m in self.sub_models)
        if total > 0:
            for m in self.sub_models:
                m.weight /= total

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> QuantSignal:
        self._normalize_weights()
        sub_signals: Dict[str, Dict[str, float]] = {}
        all_reasoning: List[str] = []
        weighted_prob = 0.0
        total_weight = 0.0
        for model in self.sub_models:
            prob, reasoning = model.predict(prices, volumes)
            sub_signals[model.name] = {
                "probability": prob, "weight": model.weight,
                "weighted_contribution": prob * model.weight,
                "accuracy": model.get_accuracy(),
            }
            weighted_prob += prob * model.weight
            total_weight += model.weight
            all_reasoning.extend([f"[{model.name.upper()}] {r}" for r in reasoning])
        ensemble_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        probs = [sub_signals[m.name]["probability"] for m in self.sub_models]
        prob_std = np.std(probs)
        disagreement_penalty = min(prob_std * 0.5, 0.15)
        ensemble_prob = 0.5 + (ensemble_prob - 0.5) * (1 - disagreement_penalty)
        ensemble_prob = float(np.clip(ensemble_prob, 0.08, 0.92))
        direction = "BULLISH" if ensemble_prob > 0.58 else "BEARISH" if ensemble_prob < 0.42 else "NEUTRAL"
        n = len(prices)
        if n >= 20:
            hist_returns = np.diff(np.log(prices[-20:]))
            avg_daily = np.mean(hist_returns)
            vol_daily = np.std(hist_returns) + 1e-8
            expected_return = avg_daily * 5 + (ensemble_prob - 0.5) * vol_daily * 3
        else:
            expected_return = (ensemble_prob - 0.5) * 0.02
            vol_daily = 0.02
        signal_strength = abs(ensemble_prob - 0.5) * 2
        agreement = 1 - prob_std * 2
        confidence = float(np.clip(signal_strength * 0.6 + max(0, agreement) * 0.4, 0.05, 0.95))
        sharpe_est = expected_return / (vol_daily * np.sqrt(5) + 1e-8) if n >= 20 else 0.0
        risk_metrics = self._compute_risk_metrics(prices)
        if ensemble_prob > 0.5:
            edge = ensemble_prob - 0.5
            odds = abs(expected_return) / (risk_metrics.get("var_95", 0.02) + 1e-8)
            kelly = float(np.clip(edge - (1 - ensemble_prob) / (odds + 1e-8), 0.0, 0.25))
        else:
            kelly = 0.0
        atr = risk_metrics.get("atr_pct", 0.015)
        all_reasoning.append(f"[ENSEMBLE] prob={ensemble_prob:.3f} conf={confidence:.2f} disagree={prob_std:.3f}")
        return QuantSignal(
            direction=direction, probability=ensemble_prob,
            expected_return=float(expected_return), confidence=confidence,
            sharpe_estimate=float(sharpe_est), sub_model_signals=sub_signals,
            reasoning=all_reasoning, risk_metrics=risk_metrics,
            optimal_position_size=kelly,
            stop_loss_pct=float(max(0.005, atr * 1.5)),
            take_profit_pct=float(max(0.008, atr * 2.5)),
        )

    def record_outcome(self, predictions: Dict[str, float], actual_return: float):
        for model in self.sub_models:
            if model.name in predictions:
                model.record_outcome(predictions[model.name], actual_return)
        self._normalize_weights()

    def _compute_risk_metrics(self, prices: np.ndarray) -> Dict[str, float]:
        n = len(prices)
        if n < 10:
            return {"volatility": 0.2, "var_95": 0.03, "atr_pct": 0.015, "max_drawdown": 0.05, "tail_ratio": 1.0}
        returns = np.diff(np.log(prices))
        true_ranges = np.abs(np.diff(prices)) / prices[:-1]
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        neg = returns[returns < 0]
        p95 = np.percentile(returns, 95) if len(returns) > 5 else 0.01
        p5 = np.abs(np.percentile(returns, 5)) if len(returns) > 5 else 0.01
        return {
            "volatility": float(np.std(returns) * np.sqrt(252)),
            "var_95": float(np.abs(np.percentile(returns, 5))) if len(returns) > 5 else 0.03,
            "var_99": float(np.abs(np.percentile(returns, 1))) if len(returns) > 5 else 0.05,
            "atr_pct": float(np.mean(true_ranges[-14:])) if len(true_ranges) >= 14 else float(np.mean(true_ranges)),
            "max_drawdown": float(np.abs(np.min(dd))) if len(dd) > 0 else 0.05,
            "downside_dev": float(np.std(neg) * np.sqrt(252)) if len(neg) > 0 else 0.0,
            "tail_ratio": float(p95 / p5) if p5 > 0 else 1.0,
        }

    def get_model_weights(self) -> Dict[str, float]:
        self._normalize_weights()
        return {m.name: round(m.weight, 4) for m in self.sub_models}

    def get_model_accuracies(self) -> Dict[str, float]:
        return {m.name: round(m.get_accuracy(), 4) for m in self.sub_models}


_quant_ensemble = None

def get_quant_ensemble() -> QuantEnsembleModel:
    global _quant_ensemble
    if _quant_ensemble is None:
        _quant_ensemble = QuantEnsembleModel()
    return _quant_ensemble
