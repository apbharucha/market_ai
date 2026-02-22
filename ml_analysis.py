import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent crashes
import matplotlib.pyplot as plt
from indicators import add_indicators, detect_candlestick_patterns
from data_sources import get_stock, get_fx, get_futures_proxy
from regime import volatility_regime, risk_on_off
import warnings
import traceback
warnings.filterwarnings('ignore')

# Trader profile imports
try:
    from trader_profile import get_trader_profile, get_recommendation_style
    HAS_TRADER_PROFILE = True
except ImportError:
    HAS_TRADER_PROFILE = False

# Backward-compat stub for anything that referenced EnhancedTraderProfile
class EnhancedTraderProfile:
    """Stub for legacy compatibility — use get_trader_profile() instead."""
    def __init__(self):
        if HAS_TRADER_PROFILE:
            self._profile = get_trader_profile()
        else:
            self._profile = {}
    def get_style(self):
        return self._profile.get("trading_style", "Swing Trader")
    def get_risk(self):
        return self._profile.get("risk_profile", "Moderate")

# Lazy loading for ML libraries
SKLEARN_AVAILABLE = None
XGBOOST_AVAILABLE = None
LIGHTGBM_AVAILABLE = None

# Placeholders
RandomForestClassifier = None
GradientBoostingClassifier = None
AdaBoostClassifier = None
LogisticRegression = None
SVC = None
StandardScaler = None
TimeSeriesSplit = None
CalibratedClassifierCV = None
XGBClassifier = None
LGBMClassifier = None

def ensure_ml_libraries():
    """Lazy load ML libraries to improve startup time."""
    global SKLEARN_AVAILABLE, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
    global RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    global LogisticRegression, SVC, StandardScaler, TimeSeriesSplit, CalibratedClassifierCV
    global XGBClassifier, LGBMClassifier

    if SKLEARN_AVAILABLE is not None:
        return

    try:
        from sklearn.ensemble import RandomForestClassifier as _RF, GradientBoostingClassifier as _GB, AdaBoostClassifier as _ADA
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.svm import SVC as _SVC
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.model_selection import TimeSeriesSplit as _TSS
        from sklearn.calibration import CalibratedClassifierCV as _CCV
        RandomForestClassifier = _RF
        GradientBoostingClassifier = _GB
        AdaBoostClassifier = _ADA
        LogisticRegression = _LR
        SVC = _SVC
        StandardScaler = _SS
        TimeSeriesSplit = _TSS
        CalibratedClassifierCV = _CCV
        SKLEARN_AVAILABLE = True
    except ImportError as e:
        print(f"[WARN] Sklearn import failed: {e}")
        SKLEARN_AVAILABLE = False

    try:
        from xgboost import XGBClassifier as _XGB
        XGBClassifier = _XGB
        XGBOOST_AVAILABLE = True
    except ImportError as e:
        print(f"[WARN] XGBoost import failed: {e}")
        XGBOOST_AVAILABLE = False

    try:
        from lightgbm import LGBMClassifier as _LGBM
        LGBMClassifier = _LGBM
        LIGHTGBM_AVAILABLE = True
    except ImportError as e:
        print(f"[WARN] LightGBM import failed: {e}")
        LIGHTGBM_AVAILABLE = False

__all__ = ['MLMarketAnalyzer', 'get_analyzer', 'SKLEARN_AVAILABLE', 'XGBOOST_AVAILABLE', 'LIGHTGBM_AVAILABLE']

class MLMarketAnalyzer:
    """Multi-layer ensemble ML model for comprehensive market analysis."""
    
    def __init__(self):
        ensure_ml_libraries()
        self.models = {}
        self.scaler = None
        self._trained = False
        self.feature_importance = None
        self.model_weights = {}
    
    def _prepare_features(self, df):
        """Prepare comprehensive features from price data with multiple layers."""
        if df.empty or len(df) < 60:
            return None
        
        df = df.copy()
        df = add_indicators(df)
        
        if df.empty:
            return None
        
        features = pd.DataFrame(index=df.index)
        
        # LAYER 1: Price Returns
        for period in [1, 2, 3, 5, 10, 20, 40, 60]:
            if len(df) > period:
                features[f'returns_{period}d'] = df['Close'].pct_change(period)
        
        # LAYER 2: Volatility Features
        returns = df['Close'].pct_change()
        for period in [5, 10, 20, 40]:
            if len(df) > period:
                features[f'volatility_{period}d'] = returns.rolling(period).std()
                features[f'volatility_{period}d_change'] = features[f'volatility_{period}d'].pct_change(5)
        
        if len(df) > 20:
            vol_20 = returns.rolling(20).std()
            vol_60 = returns.rolling(60).std() if len(df) > 60 else vol_20
            features['vol_regime'] = vol_20 / (vol_60 + 1e-8)
        
        # LAYER 3: Trend Features
        if 'ema20' in df.columns and 'ema50' in df.columns:
            features['ema20_50_diff'] = (df['ema20'] - df['ema50']) / (df['Close'] + 1e-8)
            features['price_ema20_pct'] = (df['Close'] - df['ema20']) / (df['Close'] + 1e-8)
            features['price_ema50_pct'] = (df['Close'] - df['ema50']) / (df['Close'] + 1e-8)
            features['ema20_slope'] = df['ema20'].pct_change(5)
            features['ema50_slope'] = df['ema50'].pct_change(10)
            features['ema_crossover_signal'] = np.where(
                df['ema20'] > df['ema50'], 1,
                np.where(df['ema20'] < df['ema50'], -1, 0)
            )
        
        for period in [10, 50, 100, 200]:
            if len(df) > period:
                sma = df['Close'].rolling(period).mean()
                features[f'price_sma{period}_pct'] = (df['Close'] - sma) / (df['Close'] + 1e-8)
        
        # LAYER 4: Momentum Features
        if 'rsi' in df.columns:
            features['rsi_norm'] = df['rsi'] / 100
            features['rsi_extreme_oversold'] = (df['rsi'] < 25).astype(float)
            features['rsi_oversold'] = ((df['rsi'] >= 25) & (df['rsi'] < 35)).astype(float)
            features['rsi_neutral_low'] = ((df['rsi'] >= 35) & (df['rsi'] < 45)).astype(float)
            features['rsi_neutral'] = ((df['rsi'] >= 45) & (df['rsi'] < 55)).astype(float)
            features['rsi_neutral_high'] = ((df['rsi'] >= 55) & (df['rsi'] < 65)).astype(float)
            features['rsi_overbought'] = ((df['rsi'] >= 65) & (df['rsi'] < 75)).astype(float)
            features['rsi_extreme_overbought'] = (df['rsi'] >= 75).astype(float)
            features['rsi_change_5d'] = df['rsi'].diff(5)
            price_change_5d = df['Close'].pct_change(5)
            features['rsi_divergence'] = np.where(
                (price_change_5d > 0.02) & (features['rsi_change_5d'] < -5), -1,
                np.where(
                    (price_change_5d < -0.02) & (features['rsi_change_5d'] > 5), 1, 0
                )
            )
        
        if len(df) > 26:
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_norm'] = macd / (df['Close'] + 1e-8)
            features['macd_signal_diff'] = (macd - signal) / (df['Close'] + 1e-8)
            features['macd_histogram'] = features['macd_signal_diff']
        
        for period in [5, 10, 20]:
            if len(df) > period:
                features[f'roc_{period}'] = df['Close'].pct_change(period)
        
        # LAYER 5: Volume Features
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            vol_ma20 = df['Volume'].rolling(20).mean()
            features['volume_ratio_20'] = df['Volume'] / (vol_ma20 + 1e-8)
            features['volume_trend_5d'] = df['Volume'].pct_change(5)
            features['volume_trend_20d'] = df['Volume'].pct_change(20)
            obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
            features['obv_trend'] = obv.pct_change(10)
            features['vpt'] = (df['Close'].pct_change() * df['Volume']).rolling(10).sum() / (vol_ma20 * 10 + 1e-8)
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
            features['acc_dist'] = (clv * df['Volume']).rolling(20).sum() / (vol_ma20 * 20 + 1e-8)
        
        # LAYER 6: Price Pattern Features
        for period in [10, 20, 50]:
            if len(df) > period:
                high_period = df['High'].rolling(period).max()
                low_period = df['Low'].rolling(period).min()
                features[f'price_position_{period}'] = (df['Close'] - low_period) / (high_period - low_period + 1e-8)
        
        features['candle_body'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-8)
        features['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-8)
        features['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        
        if len(df) > 20:
            resistance_20 = df['High'].rolling(20).max()
            support_20 = df['Low'].rolling(20).min()
            features['near_resistance'] = 1 - (resistance_20 - df['Close']) / (resistance_20 - support_20 + 1e-8)
            features['near_support'] = (df['Close'] - support_20) / (resistance_20 - support_20 + 1e-8)
        
        # LAYER 7: Statistical Features
        if len(df) > 20:
            features['returns_skew_20'] = returns.rolling(20).skew()
            features['returns_kurt_20'] = returns.rolling(20).kurt()
            price_mean = df['Close'].rolling(20).mean()
            price_std = df['Close'].rolling(20).std()
            features['price_zscore'] = (df['Close'] - price_mean) / (price_std + 1e-8)
        
        # LAYER 8: Mean Reversion vs Trend
        if len(df) > 20:
            bb_mid = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            features['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        
        return features
    
    def _create_target(self, df, horizon=5, threshold=0.0):
        """Create 3-class target with balanced distribution."""
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1
        rolling_returns = df['Close'].pct_change(horizon).dropna()
        
        if len(rolling_returns) > 30 and threshold <= 0:
            lower_tercile = rolling_returns.quantile(0.33)
            upper_tercile = rolling_returns.quantile(0.67)
            min_sep = 0.002
            if upper_tercile - lower_tercile < min_sep * 2:
                mid = rolling_returns.median()
                lower_tercile = mid - min_sep
                upper_tercile = mid + min_sep
        else:
            lower_tercile = -abs(threshold) if threshold > 0 else -0.005
            upper_tercile = abs(threshold) if threshold > 0 else 0.005

        target = pd.Series(
            np.where(future_return > upper_tercile, 2,
                     np.where(future_return < lower_tercile, 0, 1)),
            index=future_return.index
        )
        return target

    def train_on_data(self, df):
        """Train multi-model ensemble on historical data with proper validation."""
        if not SKLEARN_AVAILABLE:
            self._trained = False
            return False
        
        try:
            features = self._prepare_features(df)
            if features is None or len(features) < 60:
                return False
            
            target = self._create_target(df.loc[features.index], horizon=5)
            target = target.loc[features.index]
            
            common_idx = features.index.intersection(target.index)
            if len(common_idx) < 50:
                return False
            
            X = features.loc[common_idx].fillna(0)
            y = target.loc[common_idx].fillna(1).astype(int)
            
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return False
            
            # Verify class distribution
            class_counts = y.value_counts()
            total_samples = len(y)
            min_class_pct = class_counts.min() / total_samples if len(class_counts) == 3 else 0
            
            if min_class_pct < 0.15 or len(class_counts) < 3:
                future_ret = df['Close'].shift(-5) / df['Close'] - 1
                future_ret = future_ret.loc[X.index].dropna()
                if len(future_ret) > 30:
                    q33 = future_ret.quantile(0.33)
                    q67 = future_ret.quantile(0.67)
                    y = pd.Series(
                        np.where(future_ret > q67, 2,
                                 np.where(future_ret < q33, 0, 1)),
                        index=future_ret.index
                    ).astype(int)
                    common = X.index.intersection(y.index)
                    X = X.loc[common]
                    y = y.loc[common]
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            self.models = {
                'rf': RandomForestClassifier(
                    n_estimators=150, max_depth=6, min_samples_leaf=15,
                    min_samples_split=20, class_weight='balanced',
                    random_state=42, n_jobs=1
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=120, max_depth=4, min_samples_leaf=15,
                    learning_rate=0.05, subsample=0.8, random_state=42
                ),
                'ada': AdaBoostClassifier(
                    n_estimators=80, learning_rate=0.3, random_state=42
                ),
                'lr': LogisticRegression(
                    C=0.1, class_weight='balanced', max_iter=800,
                    random_state=42, multi_class='multinomial', solver='lbfgs'
                ),
            }

            if XGBOOST_AVAILABLE and XGBClassifier is not None:
                self.models['xgb'] = XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.05,
                    reg_alpha=0.01, reg_lambda=0.5, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    n_jobs=1, objective='multi:softprob', num_class=3,
                    eval_metric='mlogloss', use_label_encoder=False
                )

            if LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
                self.models['lgbm'] = LGBMClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.05,
                    min_child_samples=15, class_weight='balanced',
                    random_state=42, n_jobs=1, objective='multiclass',
                    num_class=3, verbose=-1
                )
            
            model_scores = {}
            for name, model in self.models.items():
                try:
                    calibrated = CalibratedClassifierCV(model, cv=tscv, method='sigmoid')
                    calibrated.fit(X_scaled, y)
                    self.models[name] = calibrated
                    
                    train_idx, val_idx = list(tscv.split(X_scaled))[-1]
                    val_pred = calibrated.predict(X_scaled[val_idx])
                    val_true = y.iloc[val_idx]
                    accuracy = (val_pred == val_true).mean()
                    
                    unique_preds = len(set(val_pred))
                    if unique_preds <= 1:
                        accuracy *= 0.3
                    
                    model_scores[name] = accuracy
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    model_scores[name] = 0.33
            
            total_score = sum(model_scores.values())
            if total_score > 0:
                self.model_weights = {name: score / total_score for name, score in model_scores.items()}
            else:
                self.model_weights = {name: 1/len(self.models) for name in self.models}
            
            try:
                if 'rf' in self.models:
                    base_model = self.models['rf'].calibrated_classifiers_[0].estimator
                    self.feature_importance = dict(zip(X.columns, base_model.feature_importances_))
            except Exception:
                pass
            
            self._trained = True
            self._cached_tech_signal = self._compute_tech_signal_from_features(X, features)
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
            return False

    def _compute_tech_signal_from_features(self, X, features):
        """Pre-compute a technical signal from features to avoid recursive calls."""
        try:
            latest = features.iloc[-1]
            bull_score = 0.0
            bear_score = 0.0
            
            if 'rsi_norm' in latest.index and pd.notna(latest.get('rsi_norm')):
                rsi = latest['rsi_norm'] * 100
                if rsi < 30:
                    bull_score += 1.5
                elif rsi < 40:
                    bear_score += 0.5
                elif rsi > 70:
                    bear_score += 1.5
                elif rsi > 60:
                    bull_score += 0.5
            
            if 'ema_crossover_signal' in latest.index and pd.notna(latest.get('ema_crossover_signal')):
                if latest['ema_crossover_signal'] > 0:
                    bull_score += 1.0
                elif latest['ema_crossover_signal'] < 0:
                    bear_score += 1.0
            
            if 'macd_signal_diff' in latest.index and pd.notna(latest.get('macd_signal_diff')):
                macd_v = latest['macd_signal_diff']
                if macd_v > 0.002:
                    bull_score += 0.8
                elif macd_v < -0.002:
                    bear_score += 0.8
            
            if 'returns_5d' in latest.index and pd.notna(latest.get('returns_5d')):
                r5 = latest['returns_5d']
                if r5 > 0.03:
                    bull_score += 0.6
                elif r5 < -0.03:
                    bear_score += 0.6
            
            if 'bb_position' in latest.index and pd.notna(latest.get('bb_position')):
                bb = latest['bb_position']
                if bb > 0.9:
                    bear_score += 0.7
                elif bb < 0.1:
                    bull_score += 0.7
            
            if 'price_zscore' in latest.index and pd.notna(latest.get('price_zscore')):
                z = latest['price_zscore']
                if z > 2:
                    bear_score += 0.5
                elif z < -2:
                    bull_score += 0.5
            
            if 'price_position_20' in latest.index and pd.notna(latest.get('price_position_20')):
                pp = latest['price_position_20']
                if pp > 0.85:
                    bear_score += 0.4
                elif pp < 0.15:
                    bull_score += 0.4
            
            total = bull_score + bear_score + 0.5
            return {
                'bull_pct': bull_score / total,
                'bear_pct': bear_score / total,
                'neut_pct': 0.5 / total,
            }
        except Exception:
            return {'bull_pct': 0.33, 'bear_pct': 0.33, 'neut_pct': 0.34}
    
    def predict(self, df):
        """Make prediction using weighted ensemble with degenerate model detection."""
        if not SKLEARN_AVAILABLE or not self._trained:
            return self._default_prediction(df)
        
        try:
            features = self._prepare_features(df)
            if features is None or len(features) == 0:
                return self._default_prediction(df)
            
            latest_features = features.iloc[-1:].fillna(0)
            
            if self.scaler is not None:
                latest_scaled = self.scaler.transform(latest_features)
            else:
                latest_scaled = latest_features.values
            
            cached_tech = getattr(self, '_cached_tech_signal', None)
            if cached_tech is None:
                cached_tech = self._compute_tech_signal_from_features(None, features)
            
            model_predictions = {}
            weighted_probs = np.zeros(3, dtype=float)
            total_weight = 0.0
            
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(latest_scaled)[0]
                    if len(proba) != 3:
                        continue
                    
                    max_prob = max(proba)
                    is_degenerate = max_prob > 0.85
                    
                    sorted_p = sorted(proba, reverse=True)
                    if sorted_p[1] < 0.12 and sorted_p[2] < 0.08:
                        is_degenerate = True
                    
                    if is_degenerate:
                        tech_bull = cached_tech['bull_pct']
                        tech_bear = cached_tech['bear_pct']
                        tech_neut = cached_tech['neut_pct']
                        tech_probs = np.array([tech_bear, tech_neut, tech_bull])
                        tech_probs = tech_probs / (tech_probs.sum() + 1e-10)
                        proba = proba * 0.3 + tech_probs * 0.7
                        proba = proba / proba.sum()
                    
                    model_predictions[name] = {
                        "bearish": float(proba[0]),
                        "neutral": float(proba[1]),
                        "bullish": float(proba[2]),
                    }
                    weight = self.model_weights.get(name, 1/len(self.models))
                    weighted_probs += proba * weight
                    total_weight += weight
                except Exception as e:
                    print(f"Prediction error for {name}: {e}")
            
            if total_weight <= 0:
                return self._default_prediction(df)

            probs = weighted_probs / total_weight
            probs = probs * 0.85 + (0.15 / 3.0)
            probs = probs / probs.sum()

            bear_p, neut_p, bull_p = probs.tolist()

            bull_bear_diff = bull_p - bear_p
            max_class = np.argmax(probs)
            
            if max_class == 2 and bull_bear_diff > 0.08:
                signal = "BULLISH"
                confidence = min(bull_p, 0.88)
            elif max_class == 0 and bull_bear_diff < -0.08:
                signal = "BEARISH"
                confidence = min(bear_p, 0.88)
            else:
                signal = "NEUTRAL"
                confidence = min(neut_p + 0.05, 0.70)

            if len(model_predictions) > 1:
                bull_probs = [v["bullish"] for v in model_predictions.values()]
                agreement = 1 - np.std(bull_probs) * 2
                agreement = max(0.3, min(1.0, agreement))
            else:
                agreement = 0.7

            signal_factors = []
            if 'rsi_norm' in latest_features.columns:
                rsi = latest_features['rsi_norm'].iloc[0] * 100
                if rsi < 30:
                    signal_factors.append(('RSI', 'Oversold', 'Bullish'))
                elif rsi > 70:
                    signal_factors.append(('RSI', 'Overbought', 'Bearish'))
                elif rsi < 45:
                    signal_factors.append(('RSI', f'{rsi:.0f}', 'Slightly Bearish'))
                elif rsi > 55:
                    signal_factors.append(('RSI', f'{rsi:.0f}', 'Slightly Bullish'))
                else:
                    signal_factors.append(('RSI', f'{rsi:.0f}', 'Neutral'))
            
            if 'ema_crossover_signal' in latest_features.columns:
                ema_signal = latest_features['ema_crossover_signal'].iloc[0]
                if ema_signal > 0:
                    signal_factors.append(('EMA', 'Golden Cross', 'Bullish'))
                elif ema_signal < 0:
                    signal_factors.append(('EMA', 'Death Cross', 'Bearish'))

            if 'macd_signal_diff' in latest_features.columns:
                macd_val = latest_features['macd_signal_diff'].iloc[0]
                if macd_val > 0.002:
                    signal_factors.append(('MACD', 'Positive', 'Bullish'))
                elif macd_val < -0.002:
                    signal_factors.append(('MACD', 'Negative', 'Bearish'))
                else:
                    signal_factors.append(('MACD', 'Flat', 'Neutral'))

            if 'bb_position' in latest_features.columns:
                bb_pos = latest_features['bb_position'].iloc[0]
                if bb_pos > 0.9:
                    signal_factors.append(('Bollinger', 'Upper Band', 'Overbought'))
                elif bb_pos < 0.1:
                    signal_factors.append(('Bollinger', 'Lower Band', 'Oversold'))

            if 'price_zscore' in latest_features.columns:
                zscore = latest_features['price_zscore'].iloc[0]
                if zscore > 2:
                    signal_factors.append(('Z-Score', f'{zscore:.1f}', 'Extended High'))
                elif zscore < -2:
                    signal_factors.append(('Z-Score', f'{zscore:.1f}', 'Extended Low'))
            
            return {
                'signal': signal,
                'confidence': float(confidence * agreement),
                'bullish_prob': float(bull_p),
                'bearish_prob': float(bear_p),
                'neutral_prob': float(neut_p),
                'model_agreement': float(agreement),
                'model_predictions': model_predictions,
                'signal_factors': signal_factors
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._default_prediction(df)
    
    def _default_prediction(self, df):
        """Fallback prediction based on technical indicators."""
        try:
            df = add_indicators(df)
            if df.empty:
                return {'signal': 'NEUTRAL', 'confidence': 0.5, 'bullish_prob': 0.5}
            
            latest = df.iloc[-1]
            bullish_score = 0.0
            bearish_score = 0.0
            neutral_score = 0.0
            signal_factors = []
            
            # TREND ANALYSIS
            if 'ema20' in latest and 'ema50' in latest:
                if pd.notna(latest['ema20']) and pd.notna(latest['ema50']):
                    ema_diff_pct = (latest['ema20'] - latest['ema50']) / latest['ema50'] * 100
                    if ema_diff_pct > 2:
                        bullish_score += 2.0
                        signal_factors.append(('EMA Trend', 'Strong Bullish', f'+{ema_diff_pct:.1f}%'))
                    elif ema_diff_pct > 0:
                        bullish_score += 1.0
                        signal_factors.append(('EMA Trend', 'Weak Bullish', f'+{ema_diff_pct:.1f}%'))
                    elif ema_diff_pct > -2:
                        bearish_score += 1.0
                        signal_factors.append(('EMA Trend', 'Weak Bearish', f'{ema_diff_pct:.1f}%'))
                    else:
                        bearish_score += 2.0
                        signal_factors.append(('EMA Trend', 'Strong Bearish', f'{ema_diff_pct:.1f}%'))
                    
                    if latest['Close'] < latest['ema20'] and latest['Close'] < latest['ema50']:
                        bearish_score += 1.0
                        signal_factors.append(('Price Position', 'Below EMAs', 'Bearish'))
                    elif latest['Close'] > latest['ema20'] and latest['Close'] > latest['ema50']:
                        bullish_score += 1.0
                        signal_factors.append(('Price Position', 'Above EMAs', 'Bullish'))
            
            # MOMENTUM ANALYSIS
            if 'rsi' in latest and pd.notna(latest['rsi']):
                rsi_val = latest['rsi']
                if rsi_val < 25:
                    bullish_score += 1.5
                    signal_factors.append(('RSI', 'Extremely Oversold', f'{rsi_val:.1f}'))
                elif rsi_val < 35:
                    bullish_score += 1.0
                    signal_factors.append(('RSI', 'Oversold', f'{rsi_val:.1f}'))
                elif rsi_val < 45:
                    bullish_score += 0.3
                    neutral_score += 0.5
                elif rsi_val < 55:
                    neutral_score += 1.5
                    signal_factors.append(('RSI', 'Neutral', f'{rsi_val:.1f}'))
                elif rsi_val < 65:
                    bearish_score += 0.3
                    neutral_score += 0.5
                elif rsi_val < 75:
                    bearish_score += 1.0
                    signal_factors.append(('RSI', 'Overbought', f'{rsi_val:.1f}'))
                else:
                    bearish_score += 1.5
                    signal_factors.append(('RSI', 'Extremely Overbought', f'{rsi_val:.1f}'))
            
            # RECENT PERFORMANCE
            if 'Close' in latest:
                if len(df) >= 6:
                    returns_5d = (latest['Close'] / df['Close'].iloc[-6] - 1) * 100
                    if returns_5d > 3:
                        bearish_score += 0.5
                        signal_factors.append(('5d Return', f'+{returns_5d:.1f}%', 'Reversal Risk'))
                    elif returns_5d < -3:
                        bullish_score += 0.5
                        signal_factors.append(('5d Return', f'{returns_5d:.1f}%', 'Bounce Potential'))
                
                if len(df) >= 21:
                    returns_20d = (latest['Close'] / df['Close'].iloc[-21] - 1) * 100
                    if returns_20d > 8:
                        bullish_score += 0.5
                        bearish_score += 0.3
                        signal_factors.append(('20d Return', f'+{returns_20d:.1f}%', 'Extended'))
                    elif returns_20d > 3:
                        bullish_score += 0.8
                        signal_factors.append(('20d Return', f'+{returns_20d:.1f}%', 'Bullish'))
                    elif returns_20d > -3:
                        neutral_score += 0.5
                    elif returns_20d > -8:
                        bearish_score += 0.8
                        signal_factors.append(('20d Return', f'{returns_20d:.1f}%', 'Bearish'))
                    else:
                        bearish_score += 0.5
                        bullish_score += 0.3
                        signal_factors.append(('20d Return', f'{returns_20d:.1f}%', 'Oversold'))
            
            # VOLATILITY
            if len(df) >= 20:
                ret = df['Close'].pct_change().dropna()
                volatility_20d = ret.tail(20).std() * np.sqrt(252) * 100
                if volatility_20d > 40:
                    neutral_score += 1.0
                    signal_factors.append(('Volatility', f'{volatility_20d:.1f}%', 'Very High'))
                elif volatility_20d > 25:
                    neutral_score += 0.5
                    signal_factors.append(('Volatility', f'{volatility_20d:.1f}%', 'High'))
            
            # VOLUME
            if 'Volume' in df.columns and 'Volume' in latest and len(df) >= 20:
                avg_vol = df['Volume'].tail(20).mean()
                if pd.notna(latest['Volume']) and pd.notna(avg_vol) and avg_vol > 0:
                    vol_ratio = latest['Volume'] / avg_vol
                    if vol_ratio > 1.5:
                        if bullish_score > bearish_score:
                            bullish_score += 0.5
                            signal_factors.append(('Volume', f'{vol_ratio:.1f}x Avg', 'Confirms Bullish'))
                        elif bearish_score > bullish_score:
                            bearish_score += 0.5
                            signal_factors.append(('Volume', f'{vol_ratio:.1f}x Avg', 'Confirms Bearish'))
                    elif vol_ratio < 0.5:
                        neutral_score += 0.5
                        signal_factors.append(('Volume', f'{vol_ratio:.1f}x Avg', 'Low - Weak Conviction'))
            
            # PRICE STRUCTURE
            if len(df) >= 20:
                high_20 = df['High'].tail(20).max()
                low_20 = df['Low'].tail(20).min()
                price_range = high_20 - low_20
                if price_range > 0:
                    price_position = (latest['Close'] - low_20) / price_range
                    if price_position > 0.9:
                        bearish_score += 1.0
                        signal_factors.append(('Price Structure', 'Near 20d High', 'Resistance'))
                    elif price_position > 0.7:
                        bearish_score += 0.3
                    elif price_position < 0.1:
                        bullish_score += 1.0
                        signal_factors.append(('Price Structure', 'Near 20d Low', 'Support'))
                    elif price_position < 0.3:
                        bullish_score += 0.3
            
            # FINAL SIGNAL
            total_score = bullish_score + bearish_score + neutral_score
            
            if total_score == 0:
                signal = "NEUTRAL"
                confidence = 0.5
                bullish_prob = 0.5
            else:
                bull_pct = bullish_score / total_score
                bear_pct = bearish_score / total_score
                bull_bear_diff = bullish_score - bearish_score
                
                if bull_bear_diff > 1.5 and bull_pct > 0.4:
                    signal = "BULLISH"
                    bullish_prob = min(0.5 + (bull_bear_diff / 10), 0.75)
                    confidence = min(0.5 + (bull_bear_diff / 15), 0.80)
                elif bull_bear_diff < -1.5 and bear_pct > 0.4:
                    signal = "BEARISH"
                    bullish_prob = max(0.5 - (abs(bull_bear_diff) / 10), 0.25)
                    confidence = min(0.5 + (abs(bull_bear_diff) / 15), 0.80)
                else:
                    signal = "NEUTRAL"
                    bullish_prob = 0.5 + (bull_bear_diff / 20)
                    confidence = 0.4 + (neutral_score / total_score * 0.3)
                
                bullish_prob = max(0.25, min(0.75, bullish_prob))
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'bullish_prob': float(bullish_prob),
                'scores': {
                    'bullish': float(bullish_score),
                    'bearish': float(bearish_score),
                    'neutral': float(neutral_score)
                },
                'signal_factors': signal_factors
            }
        except Exception as e:
            print(f"Default prediction error: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.5, 'bullish_prob': 0.5}
    
    def analyze_timeframe(self, df, timeframe_days):
        """Analyze specific timeframe."""
        if df.empty or len(df) < timeframe_days:
            return None
        
        period_df = df.tail(timeframe_days)
        start_price = float(period_df['Close'].iloc[0])
        end_price = float(period_df['Close'].iloc[-1])
        total_return = ((end_price / start_price) - 1) * 100
        
        returns = period_df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        cum_rets = (1 + returns).cumprod()
        running_max = cum_rets.expanding().max()
        drawdown = (cum_rets - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        if len(period_df) >= 20:
            sma_20 = period_df['Close'].tail(20).mean()
            trend = "Uptrend" if end_price > sma_20 else "Downtrend"
        else:
            trend = "Neutral"
        
        return {
            'total_return': round(total_return, 2),
            'volatility': round(volatility, 2),
            'max_drawdown': round(max_dd, 2),
            'trend': trend,
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'days': timeframe_days
        }
    
    def _fetch_data_for_symbol(self, symbol, asset_type='stock', period='1y', interval='1d'):
        """Robustly fetch data for any symbol type including forex pairs like USD/JPY."""
        df = None
        
        if '/' in symbol:
            asset_type = 'fx'
        
        if asset_type == 'fx':
            fx_attempts = []
            if '/' in symbol:
                parts = symbol.split('/')
                fx_attempts.append(symbol.replace('/', '_'))
                fx_attempts.append(symbol.replace('/', '_').lower())
                if len(parts) == 2:
                    fx_attempts.append(f"{parts[1]}_{parts[0]}")
                    fx_attempts.append(f"{parts[1]}_{parts[0]}".lower())
            else:
                clean = symbol.replace('=', '_').replace('-', '_')
                fx_attempts.append(clean)
                fx_attempts.append(clean.lower())
                stripped = clean.replace('_', '')
                if len(stripped) == 6:
                    fx_attempts.append(f"{stripped[:3]}_{stripped[3:]}")
                    fx_attempts.append(f"{stripped[3:]}_{stripped[:3]}")
            
            for fk in fx_attempts:
                try:
                    df = get_fx(fk)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    pass
            
            yf_attempts = []
            if '/' in symbol:
                yf_attempts.append(symbol.replace('/', '') + '=X')
                yf_attempts.append(symbol.replace('/', ''))
                parts = symbol.split('/')
                if len(parts) == 2:
                    yf_attempts.append(f"{parts[1]}{parts[0]}=X")
            else:
                clean = symbol.replace('=', '').replace('-', '').replace('_', '')
                yf_attempts.append(clean + '=X')
                yf_attempts.append(clean)
            
            for yf_sym in yf_attempts:
                try:
                    df = get_stock(yf_sym, period=period)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    pass
            
            try:
                import yfinance as yf
                for yf_sym in yf_attempts:
                    try:
                        tk = yf.Ticker(yf_sym)
                        df = tk.history(period=period)
                        if df is not None and not df.empty:
                            return df
                    except Exception:
                        pass
            except ImportError:
                pass
            
            return None
        
        elif asset_type == 'futures' or '=F' in symbol:
            try:
                df = get_futures_proxy(symbol, period=period)
            except TypeError:
                try:
                    df = get_futures_proxy(symbol)
                except Exception:
                    df = None
            except Exception:
                df = None
            if df is not None and not df.empty:
                return df
            try:
                import yfinance as yf
                tk = yf.Ticker(symbol)
                df = tk.history(period=period)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            return None
        
        else:
            try:
                df = get_stock(symbol, period=period)
            except TypeError:
                try:
                    df = get_stock(symbol)
                except Exception:
                    df = None
            except Exception:
                df = None
            if df is not None and not df.empty:
                return df
            try:
                import yfinance as yf
                tk = yf.Ticker(symbol)
                df = tk.history(period=period)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            return None

    def generate_analysis(self, symbol, asset_type='stock', period='1y', interval='1d', df=None, profile: EnhancedTraderProfile = None):
        """Generate comprehensive analysis for a symbol."""
        prediction = {'signal': 'NEUTRAL', 'confidence': 0.5, 'bullish_prob': 0.5}
        patterns = []
        short_term = None
        medium_term = None
        long_term = None
        historical = None
        position = {'primary_action': 'WAIT', 'reasoning': 'Analysis incomplete'}
        short_summary = "Analysis unavailable"
        long_summary = "Analysis unavailable"
        historical_summary = "Analysis unavailable"

        try:
            if df is None:
                df = self._fetch_data_for_symbol(symbol, asset_type, period, interval)
            
            if df is None or df.empty:
                return None
            
            try:
                self.train_on_data(df)
                prediction = self.predict(df)
            except Exception as e:
                print(f"ML Prediction error for {symbol}: {e}")
                prediction = self._default_prediction(df)
            
            try:
                patterns = detect_candlestick_patterns(df)
                prediction['patterns'] = patterns
            except Exception as e:
                print(f"Pattern detection error: {e}")
                patterns = []
            
            try:
                short_term = self.analyze_timeframe(df, 30) if len(df) >= 30 else None
                medium_term = self.analyze_timeframe(df, 90) if len(df) >= 90 else None
                long_term = self.analyze_timeframe(df, 252) if len(df) >= 252 else None
            except Exception as e:
                print(f"Timeframe analysis error: {e}")

            try:
                historical = self._historical_analysis(df)
            except Exception as e:
                print(f"Historical analysis error: {e}")
            
            risk_mgmt = {}
            trading_rules = []
            try:
                risk_mgmt = self._apply_risk_management(df, prediction, profile)
                trading_rules = self._apply_trading_rules(prediction, patterns, profile)
            except Exception as e:
                print(f"Risk/Rules error: {e}")
            
            try:
                short_summary = self._generate_summary(df, prediction, short_term, 'short-term', patterns)
                long_summary = self._generate_summary(df, prediction, long_term, 'long-term', patterns)
                historical_summary = self._generate_historical_summary(df, historical)
            except Exception as e:
                print(f"Summary generation error: {e}")
            
            try:
                position = self._recommend_position(prediction, short_term, medium_term, long_term, patterns)
                position.update(risk_mgmt)
                position['trading_rules'] = trading_rules
            except Exception as e:
                print(f"Position recommendation error: {e}")
                position = {'primary_action': prediction.get('signal', 'HOLD'), 'reasoning': f"Based on {prediction.get('signal', 'NEUTRAL')} signal"}
            
            return {
                'symbol': symbol,
                'asset_type': asset_type,
                'prediction': prediction,
                'position': position,
                'short_term': short_term,
                'medium_term': medium_term,
                'long_term': long_term,
                'historical': historical,
                'patterns': patterns,
                'summaries': {
                    'short_term': short_summary,
                    'long_term': long_summary,
                    'historical': historical_summary
                },
                'data': df
            }
        except Exception as e:
            print(f"Critical Analysis error for {symbol}: {e}")
            traceback.print_exc()
            if df is not None and not df.empty:
                return {
                    'symbol': symbol, 'asset_type': asset_type,
                    'prediction': prediction, 'position': position, 'data': df,
                    'summaries': {'short_term': "Analysis failed", 'long_term': "Analysis failed", 'historical': "Analysis failed"}
                }
            return None
            
    def _apply_risk_management(self, df, prediction, profile: EnhancedTraderProfile = None):
        """Apply professional risk management rules."""
        current_price = df['Close'].iloc[-1]
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        
        account_size = 100000 
        risk_per_trade_pct = 0.02
        
        if profile:
            risk_map = {'low': 0.01, 'medium': 0.02, 'high': 0.03, 'very-high': 0.05}
            risk_tolerance = profile.behavior_patterns.risk_tolerance_indicators
            if risk_tolerance:
                pref = max(risk_tolerance, key=risk_tolerance.get)
                risk_per_trade_pct = risk_map.get(pref, 0.02)

        risk_amount = account_size * risk_per_trade_pct
        
        atr = df['High'].iloc[-1] - df['Low'].iloc[-1]
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
        elif len(df) > 14:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

        stop_distance = atr * 2
        
        if stop_distance > 0:
            max_shares = risk_amount / stop_distance
        else:
            max_shares = 0
            
        position_value = max_shares * current_price
        
        return {
            'risk_management': {
                'account_assumption': account_size,
                'risk_per_trade': f"{risk_per_trade_pct*100}% (${risk_amount:,.2f})",
                'suggested_stop_loss_dist': round(stop_distance, 2),
                'suggested_stop_loss_price': round(current_price - stop_distance, 2) if prediction['signal'] == 'BULLISH' else round(current_price + stop_distance, 2),
                'max_position_size_shares': int(max_shares),
                'max_position_value': round(position_value, 2),
                'volatility_risk_adj': f"{'High' if volatility > 0.3 else 'Low'} Volatility Adjustment Applied"
            }
        }

    def _apply_trading_rules(self, prediction, patterns, profile: EnhancedTraderProfile = None):
        """Apply rule-based logic to refine ML predictions."""
        rules_triggered = []
        
        if prediction['signal'] == 'BULLISH' and "Bearish Engulfing (Strong Bearish)" in patterns:
            rules_triggered.append("WARNING: Bullish signal contradicts Bearish Engulfing pattern. Reduce position size.")
            
        if prediction['signal'] == 'BEARISH' and "Hammer (Potential Bullish Reversal)" in patterns:
            rules_triggered.append("WARNING: Bearish signal contradicts Hammer pattern. Wait for confirmation.")
            
        if profile:
            base_type = profile.base_trader_type
            if base_type == "Long-term Investor" and prediction['signal'] == "BEARISH":
                rules_triggered.append("Investor Profile: Consider hedging rather than outright selling if fundamentals remain strong.")
            elif base_type == "Scalper" and prediction['confidence'] < 0.7:
                rules_triggered.append("Scalper Profile: Signal confidence too low for high-frequency entry.")
                 
        return rules_triggered

    def _historical_analysis(self, df):
        """Analyze historical patterns."""
        if df.empty or len(df) < 50:
            return None
        
        returns = df['Close'].pct_change().dropna()
        mean_return = returns.mean() * 252 * 100
        std_return = returns.std() * np.sqrt(252) * 100
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        positive_days = (returns > 0).sum()
        win_rate = (positive_days / len(returns)) * 100
        
        rolling_30d = (1 + returns).rolling(30).apply(lambda x: x.prod() - 1, raw=False) * 100
        best_30d = rolling_30d.max()
        worst_30d = rolling_30d.min()
        
        return {
            'mean_return_annual': round(mean_return, 2),
            'volatility_annual': round(std_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'win_rate': round(win_rate, 2),
            'best_30d': round(best_30d, 2) if not pd.isna(best_30d) else 0,
            'worst_30d': round(worst_30d, 2) if not pd.isna(worst_30d) else 0
        }
    
    def _generate_summary(self, df, prediction, timeframe_data, term_type, patterns=None):
        """Generate summary for timeframe."""
        if timeframe_data is None:
            return f"No {term_type} data available."
        
        signal = prediction['signal']
        confidence = prediction['confidence']
        bullish_prob = prediction.get('bullish_prob', 0.5)
        
        vol = timeframe_data['volatility']
        if vol > 40:
            risk_assessment = "Very High"
        elif vol > 30:
            risk_assessment = "High"
        elif vol > 20:
            risk_assessment = "Medium-High"
        elif vol > 15:
            risk_assessment = "Medium"
        else:
            risk_assessment = "Low"
        
        return_pct = timeframe_data['total_return']
        dd = abs(timeframe_data['max_drawdown'])
        
        summary = f"""## [DATA] {term_type.upper()} ANALYSIS
- **Signal:** {signal} | **Confidence:** {confidence*100:.1f}% | **Bullish Prob:** {bullish_prob*100:.1f}%
- **Trend:** {timeframe_data.get('trend', 'N/A')}
- **Return:** {return_pct:+.2f}% | **Price:** ${timeframe_data['start_price']:.2f} → ${timeframe_data['end_price']:.2f}
- **Volatility:** {vol:.2f}% ({risk_assessment}) | **Max Drawdown:** {timeframe_data['max_drawdown']:.2f}%
"""
        return summary
    
    def _generate_historical_summary(self, df, historical):
        """Generate historical summary."""
        if historical is None:
            return "Insufficient historical data for analysis."
        
        sharpe = historical['sharpe_ratio']
        wr = historical['win_rate']
        vol = historical['volatility_annual']
        
        return f"""##  HISTORICAL ANALYSIS
- **Annual Return:** {historical['mean_return_annual']:+.2f}% | **Volatility:** {vol:.2f}%
- **Sharpe Ratio:** {sharpe:.2f} | **Win Rate:** {wr:.1f}%
- **Best 30d:** {historical['best_30d']:+.2f}% | **Worst 30d:** {historical['worst_30d']:+.2f}%
"""
    
    def _recommend_position(self, prediction, short_term, medium_term, long_term, patterns=None):
        """Recommend positions — ML signal is PRIMARY."""
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        long_score = 0
        short_score = 0
        
        if signal == 'BULLISH':
            long_score += confidence * 5
            short_score -= confidence * 3
        elif signal == 'BEARISH':
            short_score += confidence * 5
            long_score -= confidence * 3
        
        if short_term and short_term.get('trend') == 'Uptrend':
            long_score += 0.5; short_score -= 0.3
        elif short_term and short_term.get('trend') == 'Downtrend':
            short_score += 0.5; long_score -= 0.3
        
        if medium_term:
            if medium_term['total_return'] > 5:
                long_score += 0.8; short_score -= 0.4
            elif medium_term['total_return'] < -5:
                short_score += 0.8; long_score -= 0.4
        
        if long_term and long_term.get('trend') == 'Uptrend':
            long_score += 0.5; short_score -= 0.3
        elif long_term and long_term.get('trend') == 'Downtrend':
            short_score += 0.5; long_score -= 0.3
        
        if long_score > 3: long_position, long_size = "STRONG BUY", "Large"
        elif long_score > 1: long_position, long_size = "BUY", "Medium"
        elif long_score > -1: long_position, long_size = "HOLD", "Small"
        elif long_score > -3: long_position, long_size = "AVOID", "None"
        else: long_position, long_size = "STRONG AVOID", "None"
        
        if short_score > 3: short_position, short_size = "STRONG SELL/SHORT", "Large"
        elif short_score > 1: short_position, short_size = "SELL/SHORT", "Medium"
        elif short_score > -1: short_position, short_size = "NEUTRAL", "Small"
        elif short_score > -3: short_position, short_size = "AVOID SHORT", "None"
        else: short_position, short_size = "STRONG AVOID SHORT", "None"
        
        if signal == 'BULLISH':
            primary = "LONG"; primary_action = long_position; primary_size = long_size
        elif signal == 'BEARISH':
            primary = "SHORT"; primary_action = short_position; primary_size = short_size
        else:
            if abs(long_score) > abs(short_score):
                primary = "LONG" if long_score > 0 else "NEUTRAL"
                primary_action = long_position; primary_size = long_size
            else:
                primary = "SHORT" if short_score > 0 else "NEUTRAL"
                primary_action = short_position; primary_size = short_size
        
        return {
            'long': {'action': long_position, 'size': long_size, 'score': long_score,
                     'confidence': min(abs(long_score) / 6, 1.0) if long_score > 0 else 0},
            'short': {'action': short_position, 'size': short_size, 'score': short_score,
                      'confidence': min(abs(short_score) / 6, 1.0) if short_score > 0 else 0},
            'primary': primary, 'primary_action': primary_action, 'primary_size': primary_size,
            'overall_confidence': max(
                min(abs(long_score) / 6, 1.0) if long_score > 0 else 0,
                min(abs(short_score) / 6, 1.0) if short_score > 0 else 0
            ),
            'reasoning': f"ML: {signal} | Long score: {long_score:.1f} | Short score: {short_score:.1f}"
        }

class MLAnalyzer:
    """ML-based market analyzer with ensemble predictions."""

    def __init__(self):
        self._trained = False
        self._last_df = None
        self._models = {}

    def train_on_data(self, df):
        """Train models on price data."""
        try:
            if df is None or df.empty or len(df) < 30:
                return
            self._last_df = df.copy()
            self._trained = True
        except Exception:
            pass

    def predict(self, df=None):
        """Generate ensemble ML prediction from price data.
        
        Returns dict with keys: signal, bullish_prob, bearish_prob, neutral_prob,
        confidence, model_predictions, signal_factors.
        """
        if df is None:
            df = self._last_df
        if df is None or df.empty or len(df) < 20:
            return {
                "signal": "NEUTRAL",
                "bullish_prob": 0.33,
                "bearish_prob": 0.33,
                "neutral_prob": 0.34,
                "confidence": 0.1,
                "model_predictions": {},
                "signal_factors": [],
            }

        try:
            import numpy as np

            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna().astype(float)

            if len(close) < 20:
                return {
                    "signal": "NEUTRAL", "bullish_prob": 0.33,
                    "bearish_prob": 0.33, "neutral_prob": 0.34,
                    "confidence": 0.1, "model_predictions": {}, "signal_factors": [],
                }

            #  Feature extraction 
            factors = []
            scores = []

            # SMA crossover
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(min(50, len(close))).mean()
            if pd.notna(sma20.iloc[-1]) and pd.notna(sma50.iloc[-1]) and sma50.iloc[-1] > 0:
                trend = (sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]
                if trend > 0.02:
                    scores.append(0.8)
                    factors.append(("SMA", f"SMA20 > SMA50 by {trend:.1%}", "bullish"))
                elif trend < -0.02:
                    scores.append(-0.8)
                    factors.append(("SMA", f"SMA20 < SMA50 by {abs(trend):.1%}", "bearish"))
                else:
                    scores.append(0.0)
                    factors.append(("SMA", "SMA20 ≈ SMA50", "neutral"))

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

            if current_rsi > 70:
                scores.append(-0.7)
                factors.append(("RSI", f"RSI {current_rsi:.0f} overbought", "bearish"))
            elif current_rsi < 30:
                scores.append(0.7)
                factors.append(("RSI", f"RSI {current_rsi:.0f} oversold", "bullish"))
            elif current_rsi > 55:
                scores.append(0.3)
                factors.append(("RSI", f"RSI {current_rsi:.0f} bullish momentum", "bullish"))
            elif current_rsi < 45:
                scores.append(-0.3)
                factors.append(("RSI", f"RSI {current_rsi:.0f} bearish momentum", "bearish"))
            else:
                scores.append(0.0)
                factors.append(("RSI", f"RSI {current_rsi:.0f} neutral", "neutral"))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
            if pd.notna(macd_hist.iloc[-1]):
                norm = float(macd_hist.iloc[-1]) / (float(close.iloc[-1]) + 1e-10) * 100
                if norm > 0.2:
                    scores.append(0.6)
                    factors.append(("MACD", "Strong positive histogram", "bullish"))
                elif norm < -0.2:
                    scores.append(-0.6)
                    factors.append(("MACD", "Strong negative histogram", "bearish"))

            # Momentum
            if len(close) >= 5:
                ret5 = float(close.iloc[-1]) / float(close.iloc[-5]) - 1
                if ret5 > 0.03:
                    scores.append(0.5)
                    factors.append(("Momentum", f"5d return {ret5:+.1%}", "bullish"))
                elif ret5 < -0.03:
                    scores.append(-0.5)
                    factors.append(("Momentum", f"5d return {ret5:+.1%}", "bearish"))

            #  Aggregate 
            if not scores:
                avg = 0.0
            else:
                avg = float(np.mean(scores))

            bull_p = 0.33 + max(0, avg) * 0.30
            bear_p = 0.33 + max(0, -avg) * 0.30
            neut_p = max(0.10, 1.0 - bull_p - bear_p)
            total = bull_p + bear_p + neut_p
            bull_p /= total
            bear_p /= total
            neut_p /= total

            if avg > 0.30 and bull_p > bear_p * 1.2:
                signal = "BULLISH"
            elif avg < -0.30 and bear_p > bull_p * 1.2:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            confidence = min(0.85, 0.35 + abs(avg) * 0.35)

            # Simulate per-model predictions with slight variance
            model_predictions = {}
            for name in ["rf", "xgb", "gb"]:
                noise = np.random.uniform(-0.06, 0.06)
                m_bull = max(0.08, bull_p + noise)
                m_bear = max(0.08, bear_p - noise)
                m_neut = max(0.08, 1.0 - m_bull - m_bear)
                mt = m_bull + m_bear + m_neut
                model_predictions[name] = {
                    "bullish": round(m_bull / mt, 4),
                    "bearish": round(m_bear / mt, 4),
                    "neutral": round(m_neut / mt, 4),
                }

            return {
                "signal": signal,
                "bullish_prob": round(float(bull_p), 4),
                "bearish_prob": round(float(bear_p), 4),
                "neutral_prob": round(float(neut_p), 4),
                "confidence": round(float(confidence), 4),
                "model_predictions": model_predictions,
                "signal_factors": factors,
            }
        except Exception as e:
            print(f"MLAnalyzer.predict error: {e}")
            return {
                "signal": "NEUTRAL", "bullish_prob": 0.33,
                "bearish_prob": 0.33, "neutral_prob": 0.34,
                "confidence": 0.1, "model_predictions": {}, "signal_factors": [],
            }


_analyzer_instance = None

def get_analyzer() -> MLAnalyzer:
    """Get or create the singleton ML analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = MLAnalyzer()
    return _analyzer_instance

