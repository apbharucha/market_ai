from typing import Dict, List, Optional
import pandas as pd
from data_sources import get_stock # Assuming this is available in python path

class InterestRateManager:
    """
    Manages interest rate data, yield curves, and central bank expectations.
    Fetcher uses standard tickers:
    US 10Y: ^TNX (Divide by 10)
    US 2Y: ^IRX (Approximation or use specific ticker if available. Often ^IRX is 13-week. 
                  Yahoo often uses ^UST2Y but it might not be available. 
                  We will use ^TNX (10Y) and ^FVX (5Y) and ^IRX (13W) as proxies if needed, 
                  or hardcoded logic for now if tickers fail)
    """
    def __init__(self):
        self.yield_curves: Dict[str, Dict[str, float]] = {} # Country -> {Tenor -> Yield}
        self.rate_expectations: Dict[str, str] = {} # Country -> Hawk/Dove/Neutral
        self._initialize_default_curves()
        
    def _initialize_default_curves(self):
        # Initialize with rough defaults in case fetch fails
        self.yield_curves["USD"] = {"2Y": 4.60, "10Y": 4.25, "30Y": 4.40}
        self.yield_curves["EUR"] = {"2Y": 2.90, "10Y": 2.40}
        self.yield_curves["JPY"] = {"2Y": 0.10, "10Y": 0.75}
        self.yield_curves["GBP"] = {"2Y": 4.50, "10Y": 4.10}
        
    def fetch_live_rates(self):
        """
        Attempts to fetch live rates from data sources.
        """
        try:
            # US 10Y
            df_10y = get_stock("^TNX", period="5d")
            if not df_10y.empty:
                # Yahoo reports 42.50 for 4.25% usually, or sometimes 4.25 directly.
                # ^TNX is usually CBOE Interest Rate 10 Year T Note which is yield * 10
                latest = df_10y["Close"].iloc[-1]
                self.yield_curves["USD"]["10Y"] = latest / 10.0
            
            # US 5Y
            df_5y = get_stock("^FVX", period="5d")
            if not df_5y.empty:
                latest = df_5y["Close"].iloc[-1]
                self.yield_curves["USD"]["5Y"] = latest / 10.0
                
            # For 2Y, Yahoo is tricky. We'll estimate or leave default if trigger fails.
            
        except Exception as e:
            print(f"Error fetching rates: {e}")
            
    def get_yield_spread(self, country: str, tenor_long: str = "10Y", tenor_short: str = "2Y") -> float:
        if country not in self.yield_curves:
            return 0.0
        
        curve = self.yield_curves[country]
        long_yield = curve.get(tenor_long)
        short_yield = curve.get(tenor_short)
        
        if long_yield is not None and short_yield is not None:
            return long_yield - short_yield
            
        return 0.0 # Default if missing
        
    def analyze_yield_curve_signal(self, country: str) -> Dict:
        """
        Returns signals based on curve shape (Inverted, Steepening, Flat).
        """
        spread = self.get_yield_spread(country, "10Y", "2Y")
        
        signal = "NEUTRAL"
        description = "Normal positive slope"
        score_modifier = 0 # -10 to +10 impact on risk appetite
        
        if spread < -0.10:
            signal = "INVERTED"
            description = "Recession Warning: Curve Inverted"
            score_modifier = -10 # Risk Off
        elif spread < 0.10:
            signal = "FLAT"
            description = "Economic uncertainty or transition"
            score_modifier = -5
        elif spread > 1.2:
            signal = "STEEP"
            description = "Early cycle recovery or inflation expectations"
            score_modifier = 5
        
        return {
            "signal": signal,
            "spread_bps": round(spread * 100, 1),
            "description": description,
            "risk_modifier": score_modifier
        }
