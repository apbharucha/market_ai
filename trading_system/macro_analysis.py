from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class EconomyData:
    country_code: str # USD, EUR, JPY, GBP, AUD, CAD, CHF, NZD
    gdp_growth_yoy: float
    inflation_rate: float
    unemployment_rate: float
    interest_rate: float
    debt_to_gdp: float
    current_account_balance: float # % of GDP
    manufacturing_pmi: float # > 50 expansion
    services_pmi: float # > 50 expansion
    central_bank_policy: str # HAWKISH, DOVISH, NEUTRAL
    fiscal_stability: str # STABLE, EXPANSIONARY, CONTRACTIONARY, UNSTABLE
    political_risk: str # LOW, MEDIUM, HIGH

class MacroAnalyzer:
    """
    Evaluates macroeconomic strength of economies for FX and global macro trading.
    """
    def __init__(self):
        self.economies: Dict[str, EconomyData] = {}
        self._initialize_default_data()
        
    def _initialize_default_data(self):
        # Initialize with rough current global macro snapshot (Mock/Proxy for now)
        # In a real system, this would fetch from an API like TradingEconomics or Bloomberg
        
        self.economies["USD"] = EconomyData(
            country_code="USD",
            gdp_growth_yoy=2.9,
            inflation_rate=3.1,
            unemployment_rate=3.7,
            interest_rate=5.50,
            debt_to_gdp=122.0,
            current_account_balance=-3.0,
            manufacturing_pmi=49.1,
            services_pmi=53.4,
            central_bank_policy="HAWKISH", # Higher for longer
            fiscal_stability="EXPANSIONARY",
            political_risk="MEDIUM"
        )
        
        self.economies["EUR"] = EconomyData(
            country_code="EUR",
            gdp_growth_yoy=0.5,
            inflation_rate=2.8,
            unemployment_rate=6.4,
            interest_rate=4.50,
            debt_to_gdp=90.0,
            current_account_balance=2.0,
            manufacturing_pmi=46.6,
            services_pmi=48.8,
            central_bank_policy="NEUTRAL",
            fiscal_stability="STABLE",
            political_risk="MEDIUM"
        )
        
        self.economies["JPY"] = EconomyData(
            country_code="JPY",
            gdp_growth_yoy=1.0,
            inflation_rate=2.2,
            unemployment_rate=2.5,
            interest_rate=-0.10,
            debt_to_gdp=260.0,
            current_account_balance=3.5,
            manufacturing_pmi=48.0,
            services_pmi=51.5,
            central_bank_policy="DOVISH", # Still loose but looking to exit
            fiscal_stability="STABLE",
            political_risk="LOW"
        )
        
        self.economies["GBP"] = EconomyData(
            country_code="GBP",
            gdp_growth_yoy=0.2,
            inflation_rate=4.0,
            unemployment_rate=4.2,
            interest_rate=5.25,
            debt_to_gdp=100.0,
            current_account_balance=-3.5,
            manufacturing_pmi=47.5,
            services_pmi=52.0,
            central_bank_policy="HAWKISH",
            fiscal_stability="UNSTABLE",
            political_risk="MEDIUM"
        )

    def update_economic_data(self, country_code: str, data: EconomyData):
        self.economies[country_code] = data
        
    def calculate_strength_score(self, country_code: str) -> float:
        """
        Calculates a 0-100 strength score for an economy.
        50 is Neutral. >50 is Strong/Bullish Currency. <50 is Weak/Bearish.
        """
        if country_code not in self.economies:
            return 50.0 
            
        data = self.economies[country_code]
        score = 50.0
        
        # 1. Growth (GDP & PMI) - Weight: 25%
        # Strong growth drives capital inflows
        score += (data.gdp_growth_yoy - 1.5) * 5 # Baseline 1.5%
        score += (data.services_pmi - 50.0) * 0.5
        score += (data.manufacturing_pmi - 50.0) * 0.3
        
        # 2. Interest Rate & Policy - Weight: 30%
        # Higher rates attract carry trade
        score += (data.interest_rate - 2.5) * 4 # Baseline 2.5% rate
        
        if data.central_bank_policy == "HAWKISH":
            score += 5
        elif data.central_bank_policy == "DOVISH":
            score -= 5
            
        # 3. Inflation Breakdown - Weight: 15%
        # Moderate inflation (2%) is good. High (>3%) is bad unless rates match.
        # Low (<1%) is bad (deflation).
        inflation_diff = abs(data.inflation_rate - 2.0)
        if inflation_diff > 1.0:
            score -= inflation_diff * 2 
        
        # 4. External Balance - Weight: 10%
        # Surplus is bullish
        score += data.current_account_balance * 1.5
        
        # 5. Stability (Fiscal/Political) - Weight: 20%
        if data.political_risk == "HIGH": score -= 10
        if data.political_risk == "LOW": score += 2
        
        if data.fiscal_stability == "UNSTABLE": score -= 5
        
        return max(0.0, min(100.0, score))
        
    def analyze_pair(self, base_currency: str, quote_currency: str) -> Dict:
        """
        Analyzes an FX pair and returns relative strength metrics.
        """
        base_score = self.calculate_strength_score(base_currency)
        quote_score = self.calculate_strength_score(quote_currency)
        
        net_strength = base_score - quote_score
        
        # Construct detailed reasoning
        reasoning = []
        if net_strength > 10:
            reasoning.append(f"{base_currency} Macro significantly stronger than {quote_currency}")
        elif net_strength < -10:
             reasoning.append(f"{quote_currency} Macro significantly stronger than {base_currency}")
        else:
            reasoning.append("Macro conditions relatively neutral/balanced")
            
        return {
            "base_currency": base_currency,
            "base_score": round(base_score, 1),
            "quote_currency": quote_currency,
            "quote_score": round(quote_score, 1),
            "net_relative_strength": round(net_strength, 1),
            "favored_currency": base_currency if net_strength > 0 else quote_currency,
            "reasoning": reasoning
        }
