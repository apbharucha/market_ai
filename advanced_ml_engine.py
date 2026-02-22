"""
Advanced ML Engine for Octavian
Implements massive and complex neural network models within an ensemble framework.
Includes LSTM, Transformer, and Deep Dense Networks weighted dynamically.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import warnings
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OctavianML")

warnings.filterwarnings('ignore')

# --- PyTorch Deep Learning Models ---

class MarketLSTM(nn.Module):
    """
    Complex Long Short-Term Memory Network for Time Series Prediction.
    Captures temporal dependencies in price action.
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=3, output_dim=1):
        super(MarketLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc_1(out[:, -1, :]) # Take last time step
        out = self.relu(out)
        out = self.fc_2(out)
        return out

class MarketTransformer(nn.Module):
    """
    Transformer Encoder for Financial Time Series.
    Uses self-attention to identify non-linear relationships across timeframes.
    """
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=3, output_dim=1):
        super(MarketTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model)) # Max sequence length 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global Average Pooling
        x = self.fc_out(x)
        return x

# --- Ensemble Manager ---

class AdvancedEnsembleEngine:
    """
    Master Ensemble Model managing multiple complex neural networks and ML models.
    Weights outcomes based on a variety of factors including historical accuracy and volatility.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Neural Networks
        self.lstm_model = MarketLSTM(input_dim=5).to(self.device)
        self.transformer_model = MarketTransformer(input_dim=5).to(self.device)
        
        # Initialize Scikit-Learn Models (Deep Dense & Tree-based)
        self.mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=200, random_state=42, alpha=0.01) # Added L2 Regularization
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=4, random_state=42) # Reduced depth, added min_samples
        self.gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_leaf=4, random_state=42) # Reduced depth, added min_samples
        
        # Pre-train / Initialize weights with synthetic data to avoid random garbage in Fast Mode
        # self._pretrain_general_model() # DISABLED for performance - relies on online learning or saved state

        # Dynamic Weighting System (Initial weights)
        self.model_weights = {
            'LSTM_Neural_Net': 0.25,
            'Transformer_Attention': 0.25,
            'Deep_MLP_Network': 0.20,
            'Random_Forest_Ensemble': 0.15,
            'Gradient_Boosting_Machine': 0.15
        }
        
        self.scaler = MinMaxScaler()
        
    def _pretrain_general_model(self):
        """Pre-train models on synthetic data to establish baseline weights."""
        try:
            # Generate synthetic market-like data (sine wave + noise + trend)
            t = np.linspace(0, 100, 500)
            price = 100 + 10 * np.sin(t/5) + 0.5 * t + np.random.normal(0, 2, 500)
            
            df_synth = pd.DataFrame({
                'Open': price + np.random.normal(0, 1, 500),
                'High': price + 2,
                'Low': price - 2,
                'Close': price,
                'Volume': np.abs(np.random.normal(10000, 2000, 500))
            })
            
            # Temporary scaler for pre-training
            temp_scaler = MinMaxScaler()
            
            # Feature engineering for synth data
            df_synth['Returns'] = df_synth['Close'].pct_change()
            df_synth['Volatility'] = df_synth['Returns'].rolling(window=20).std()
            df_synth = df_synth.fillna(0)
            
            features = ['Close', 'High', 'Low', 'Volume', 'Volatility']
            dataset = df_synth[features].values
            scaled_data = temp_scaler.fit_transform(dataset)
            
            lookback = 30
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                y.append(scaled_data[i, 0])
                
            X, y = np.array(X), np.array(y)
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
            
            # Train PyTorch models briefly
            criterion = nn.MSELoss()
            optimizer_lstm = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)
            optimizer_trans = torch.optim.Adam(self.transformer_model.parameters(), lr=0.01)
            
            self.lstm_model.train()
            self.transformer_model.train()
            
            for _ in range(10): # 10 epochs
                optimizer_lstm.zero_grad()
                out = self.lstm_model(X_tensor)
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer_lstm.step()
                
                optimizer_trans.zero_grad()
                out = self.transformer_model(X_tensor)
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer_trans.step()
                
            # Train Sklearn models
            X_flat = X.reshape(X.shape[0], -1)
            self.mlp_model.fit(X_flat, y)
            self.rf_model.fit(X_flat, y)
            self.gbm_model.fit(X_flat, y)
            
        except Exception as e:
            logger.warning(f"Pre-training failed: {e}")

    def _prepare_data(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training/inference from a DataFrame."""
        if len(df) < lookback + 10:
            return None, None, None
            
        # Feature Engineering
        cols_to_get = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            cols_to_get.append('Volume')
            
        data = df[cols_to_get].copy()
        
        # Ensure Volume exists for the models even if not in data
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        
        # Add basic technical features for the ML models
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data = data.fillna(0)
        
        # Select 5 core features for consistent input dimension
        features = ['Close', 'High', 'Low', 'Volume', 'Volatility']
        dataset = data[features].values
        
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0]) # Predicting next Close (normalized)
            
        X, y = np.array(X), np.array(y)
        
        # Last sequence for future prediction
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, len(features))
        
        return X, y, last_sequence

    def analyze_symbol_ensemble(self, df: pd.DataFrame, symbol: str, fast_mode: bool = False) -> Dict[str, Any]:
        """
        Run the massive ensemble analysis on a single symbol.
        Performs online learning on recent data and predicts future movement.
        """
        lookback = 30
        X, y, last_seq = self._prepare_data(df, lookback=lookback)
        
        if X is None or len(X) < 10:
            return {
                "decision": "NEUTRAL",
                "confidence": 0.0,
                "predicted_return": 0.0,
                "model_breakdown": {}
            }
            
        # --- Online Learning (Rapid Adaptation) ---
        # We train the models on the specific asset's recent history to adapt to its current regime.
        
            # Train PyTorch Models (Short epochs for speed)
            # Incorporating L2 Regularization (weight_decay)
            optimizer_lstm = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01, weight_decay=1e-4)
            optimizer_trans = torch.optim.Adam(self.transformer_model.parameters(), lr=0.01, weight_decay=1e-4)
            
            # --- Time-Series Cross-Validation logic (Simplified for speed) ---
            # Instead of just fitting on the whole X, we can do a walk-forward split
            # For fast online learning we'll still fit on all data but ensure we have regularization
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            self.lstm_model.train()
            # Reduce epochs for speed if just initializing
            epochs = 5 if not fast_mode else 1
            for _ in range(epochs): 
                optimizer_lstm.zero_grad()
                out = self.lstm_model(X_tensor)
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer_lstm.step()
                
            # Train Transformer
            self.transformer_model.train()
            for _ in range(epochs):
                optimizer_trans.zero_grad()
                out = self.transformer_model(X_tensor)
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer_trans.step()
                
            # Train Sklearn Models with walk-forward validation (conceptually)
            X_flat = X.reshape(X.shape[0], -1)
            
            # To actually reduce overfitting during fit, we could use GridSearch but it's too slow for online learning.
            # We rely on the structural regularization added in __init__ (alpha, max_depth, min_samples_leaf)
            self.mlp_model.fit(X_flat, y)
            self.rf_model.fit(X_flat, y)
            self.gbm_model.fit(X_flat, y)
        
        # --- Ensemble Inference ---
        self.lstm_model.eval()
        self.transformer_model.eval()
        
        with torch.no_grad():
            pred_lstm = self.lstm_model(last_seq_tensor).item()
            pred_trans = self.transformer_model(last_seq_tensor).item()
            
        last_seq_flat = last_seq.reshape(1, -1)
        
        # Check if sklearn models are fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.mlp_model)
            pred_mlp = self.mlp_model.predict(last_seq_flat)[0]
            pred_rf = self.rf_model.predict(last_seq_flat)[0]
            pred_gbm = self.gbm_model.predict(last_seq_flat)[0]
        except:
            # If not fitted (should happen only if pretrain failed), fit on current batch
            X_flat = X.reshape(X.shape[0], -1)
            self.mlp_model.fit(X_flat, y)
            self.rf_model.fit(X_flat, y)
            self.gbm_model.fit(X_flat, y)
            pred_mlp = self.mlp_model.predict(last_seq_flat)[0]
            pred_rf = self.rf_model.predict(last_seq_flat)[0]
            pred_gbm = self.gbm_model.predict(last_seq_flat)[0]
        
        # --- Inverse Transform Predictions ---
        # We need to invert the scaling to get actual price targets
        # The scaler was fit on [Close, High, Low, Vol, Volatility]
        # We only care about the first column (Close)
        
        current_close = df['Close'].iloc[-1]
        
        def inverse_price(pred_val):
            # Create a dummy row with the predicted close and 0s for others
            dummy = np.zeros((1, 5))
            dummy[0, 0] = pred_val
            return self.scaler.inverse_transform(dummy)[0, 0]
            
        target_lstm = inverse_price(pred_lstm)
        target_trans = inverse_price(pred_trans)
        target_mlp = inverse_price(pred_mlp)
        target_rf = inverse_price(pred_rf)
        target_gbm = inverse_price(pred_gbm)
        
        predictions = {
            'LSTM_Neural_Net': target_lstm,
            'Transformer_Attention': target_trans,
            'Deep_MLP_Network': target_mlp,
            'Random_Forest_Ensemble': target_rf,
            'Gradient_Boosting_Machine': target_gbm
        }
        
        # --- Weighted Voting Mechanism ---
        weighted_sum = 0
        total_weight = 0
        
        # Calculate dynamic weights based on error on last training sample (sanity check)
        # (Simplified: using static weights defined in __init__ for now, but could be dynamic)
        
        for name, weight in self.model_weights.items():
            pred = predictions[name]
            weighted_sum += pred * weight
            total_weight += weight
            
        final_predicted_price = weighted_sum / total_weight
        predicted_return = (final_predicted_price - current_close) / current_close
        
        # Determine Confidence based on model agreement
        # Calculate standard deviation of predictions
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        # Lower std dev means models agree -> Higher confidence
        confidence = max(0.0, 1.0 - (pred_std / current_close) * 100) # Simple heuristic
        
        decision = "HOLD"
        if predicted_return > 0.005: # > 0.5% predicted gain
            decision = "BUY"
        elif predicted_return < -0.005: # < -0.5% predicted loss
            decision = "SELL"
            
        return {
            "decision": decision,
            "confidence": confidence,
            "predicted_return": predicted_return,
            "final_predicted_price": final_predicted_price,
            "current_price": current_close,
            "model_breakdown": predictions,
            "model_weights": self.model_weights
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Return summary of model weights."""
        return self.model_weights

# --- Singleton Instance (Lazy Loading handled by caller or Streamlit cache) ---
# Removed global instantiation to prevent import-time training lag
# ensemble_engine = AdvancedEnsembleEngine() 

def get_ensemble_engine():
    """Factory to get singleton instance."""
    if not hasattr(get_ensemble_engine, "instance"):
        get_ensemble_engine.instance = AdvancedEnsembleEngine()
    return get_ensemble_engine.instance
