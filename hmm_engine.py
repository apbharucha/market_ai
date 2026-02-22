"""
Market Regime Detection Engine
Uses Gaussian Mixture Models (GMM) to identify market states (Bull/Bear/Sideways/Volatile).
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OctavianHMM")

class MarketRegimeDetector:
    """
    Detects market regimes using unsupervised learning (GMM).
    Classifies periods into:
    0: Low Volatility Bull (Steady uptrend)
    1: High Volatility Bull (Correction/Rally)
    2: Low Volatility Bear (Slow bleed)
    3: High Volatility Bear (Crash/Panic)
    (Note: Cluster IDs are arbitrary and need mapping after fitting)
    """
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_map = {} # Maps cluster ID to readable name
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection."""
        if len(df) < 50:
            return None
            
        data = df.copy()
        
        # Calculate Returns
        data['returns'] = data['Close'].pct_change()
        
        # Calculate Volatility (20-day rolling std dev)
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Calculate Volume Change
        if 'Volume' in data.columns:
            data['vol_change'] = data['Volume'].pct_change()
        else:
            data['vol_change'] = 0
            
        # Calculate Trend Strength (ADX proxy: abs(EMA20 - EMA50) / Close)
        data['ema20'] = data['Close'].ewm(span=20).mean()
        data['ema50'] = data['Close'].ewm(span=50).mean()
        data['trend_strength'] = abs(data['ema20'] - data['ema50']) / data['Close']
        
        # Drop NaNs
        features = data[['returns', 'volatility', 'trend_strength']].dropna()
        return features

    def fit(self, df: pd.DataFrame):
        """Fit the GMM model to historical data."""
        features = self._prepare_features(df)
        if features is None or len(features) < 100:
            logger.warning("Insufficient data to fit HMM/GMM.")
            return
            
        X = self.scaler.fit_transform(features)
        self.model.fit(X)
        self.is_fitted = True
        
        # Heuristic to map clusters to regimes
        means = self.model.means_
        # means shape: [n_components, n_features]
        # features: 0=returns, 1=volatility, 2=trend_strength
        
        # Sort clusters by returns and volatility to label them
        for i in range(self.n_components):
            avg_ret = means[i][0]
            avg_vol = means[i][1]
            
            label = "Unknown"
            if avg_ret > 0 and avg_vol < 0: # High returns, low vol scaling
                label = "Steady Bull"
            elif avg_ret > 0 and avg_vol > 0: # High returns, high vol
                label = "Volatile Rally"
            elif avg_ret < 0 and avg_vol < 0:
                label = "Sideways/Correction"
            elif avg_ret < 0 and avg_vol > 0:
                label = "Bear Crash"
            else:
                label = "Transition"
                
            self.regime_map[i] = label
            
        logger.info(f"Regime map: {self.regime_map}")

    def predict_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict the current market regime."""
        if not self.is_fitted:
            self.fit(df) # Auto-fit if needed
            
        features = self._prepare_features(df)
        if features is None or len(features) == 0:
            return {"regime": "Unknown", "confidence": 0.0}
            
        last_feature = features.iloc[[-1]] # Keep as DataFrame
        X_last = self.scaler.transform(last_feature)
        
        regime_id = self.model.predict(X_last)[0]
        probs = self.model.predict_proba(X_last)[0]
        confidence = probs[regime_id]
        
        regime_label = self.regime_map.get(regime_id, f"Regime {regime_id}")
        
        return {
            "regime_id": int(regime_id),
            "regime": regime_label,
            "confidence": float(confidence),
            "probs": probs.tolist(),
            "desc": self._get_regime_desc(regime_label)
        }
        
    def _get_regime_desc(self, label: str) -> str:
        descriptions = {
            "Steady Bull": "Low volatility uptrend. Buy dips.",
            "Volatile Rally": "High volatility uptrend. Tight stops recommended.",
            "Sideways/Correction": "Market is chopping or slowly bleeding. Range trade.",
            "Bear Crash": "High volatility downtrend. Cash or Short positions preferred.",
            "Transition": "Market is changing state. Caution advised."
        }
        return descriptions.get(label, "Market state is ambiguous.")

# Singleton
_detector = None

def get_regime_detector():
    global _detector
    if _detector is None:
        _detector = MarketRegimeDetector()
    return _detector
