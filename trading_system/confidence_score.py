from typing import Dict, List

class ConfidenceScorer:
    """
    Calculates a unified confidence score (0-100) based on weighted factors.
    """
    def __init__(self):
        # Default Weights
        self.weights = {
            "technical": 0.30,
            "macro": 0.25,
            "rates": 0.15,
            "risk_reward": 0.15,
            "ml_model": 0.15
        }
        
    def calculate_score(self, 
                        technical_score: float, # 0-100
                        macro_score: float, # 0-100
                        rates_conviction: float, # 0-100
                        risk_reward_ratio: float, # e.g. 3.0
                        ml_probability: float # 0-100
                       ) -> Dict:
        """
        Returns detailed score breakdown.
        """
        
        # Normalize R:R to 0-100 score (Capped at 5R for max score)
        rr_score = min(risk_reward_ratio * 20, 100)
        
        weighted_score = (
            technical_score * self.weights["technical"] +
            macro_score * self.weights["macro"] +
            rates_conviction * self.weights["rates"] +
            rr_score * self.weights["risk_reward"] +
            ml_probability * self.weights["ml_model"]
        )
        
        return {
            "total_score": round(weighted_score, 1),
            "breakdown": {
                "technical": technical_score,
                "macro": macro_score,
                "rates": rates_conviction,
                "risk_reward_score": rr_score,
                "ml_probability": ml_probability
            },
            "rating": self._get_rating(weighted_score)
        }
        
    def _get_rating(self, score: float) -> str:
        if score >= 90: return "A+ (Conviction Buy)"
        if score >= 80: return "A (Strong)"
        if score >= 70: return "B (Good)"
        if score >= 60: return "C (Speculative)"
        return "D (Weak/Avoid)"
