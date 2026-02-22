"""
Patch for ml_analysis.py — fixes always-bullish Random Forest and always-neutral XGBoost.

Import this ONCE at startup, or apply the fixes directly to ml_analysis.py.
The core problem: the models use hardcoded or poorly calibrated thresholds
that collapse all predictions to one direction.

Usage:
    import ml_analysis_fix  # auto-patches
"""

try:
    from ml_analysis import get_analyzer

    _original_predict = None

    def _patched_predict(self, df):
        """Wrapper that fixes bias in RF/XGB predictions."""
        import numpy as np

        # Call original predict
        result = _original_predict(self, df)

        if not isinstance(result, dict):
            return result

        # Fix 1: If bullish_prob and bearish_prob exist, re-derive signal from them
        bull_p = result.get('bullish_prob', 0.5)
        bear_p = result.get('bearish_prob', 0.5)
        neutral_p = result.get('neutral_prob', 0.0)

        # Fix 2: If probabilities are present, use balanced thresholds
        # Instead of winner-take-all, use proper thresholds
        if bull_p is not None and bear_p is not None:
            # Normalize
            total = bull_p + bear_p + neutral_p
            if total > 0:
                bull_p /= total
                bear_p /= total
                neutral_p /= total

            # Balanced signal derivation
            if bull_p > 0.45 and bull_p > bear_p * 1.3:
                signal = 'BULLISH'
            elif bear_p > 0.45 and bear_p > bull_p * 1.3:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'

            result['signal'] = signal
            result['bullish_prob'] = round(bull_p, 4)
            result['bearish_prob'] = round(bear_p, 4)
            result['neutral_prob'] = round(neutral_p, 4)

        # Fix 3: If only a single probability and signal exist
        elif 'probability' in result or 'confidence' in result:
            prob = result.get('probability', result.get('confidence', 0.5))
            if prob > 0.58:
                result['signal'] = 'BULLISH'
            elif prob < 0.42:
                result['signal'] = 'BEARISH'
            else:
                result['signal'] = 'NEUTRAL'

        return result

    # Apply patch
    analyzer = get_analyzer()
    if hasattr(analyzer, 'predict'):
        _original_predict = type(analyzer).predict
        type(analyzer).predict = _patched_predict
        print("[OK] [ml_analysis_fix] Patched predict() to fix bullish/neutral bias")
    else:
        print("[WARN] [ml_analysis_fix] Analyzer has no predict method, skipping patch")

except Exception as e:
    print(f"[WARN] [ml_analysis_fix] Could not patch: {e}")
