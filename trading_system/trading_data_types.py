from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

class TradeType(Enum):
    SPOT = "Spot"
    FX = "FX"
    FUTURES = "Futures"
    OPTIONS = "Options"
    RATES = "Rates"
    YIELD_SPREAD = "Yield Spread"

class Direction(Enum):
    LONG = "Long"
    SHORT = "Short"

class OrderType(Enum):
    MARKET = "Market"
    STOP_ENTRY = "Stop Entry"  # Triggered when price crosses a level
    MARKET_BREAKOUT = "Market Breakout" # Similar to Stop Entry but specific context
    LIMIT_RETRACEMENT = "Limit Retracement" # Wait for pullback
    LIMIT = "Limit"
    STOP_LOSS = "Stop Loss"

@dataclass
class PositionSpec:
    """
    Full specification of a position before execution.
    """
    position_size_units: float
    position_size_dollars: float
    risk_allocation_pct: float  # % of portfolio allowed to risk
    stop_loss_price: float
    stop_loss_dollars: float
    take_profit_price: float # Primary target
    take_profit_targets: List[float] # Multiple targets
    risk_reward_ratio: float
    expected_return_dollars: float
    expected_return_pct: float
    expected_holding_period: str # e.g. "2-3 Days", "Intraday"
    
@dataclass
class TradeAlert:
    """
    Proactive trade alert that activates only if price reaches a trigger.
    """
    # Identification
    symbol: str
    trade_type: TradeType
    direction: Direction
    trigger_price: float
    order_type: OrderType
    
    # Position Specs
    position_spec: PositionSpec
    
    # Confidence & Reasoning
    confidence_score: float # 0-100
    confidence_breakdown: Dict[str, float] # Factors contributing to score
    reasoning: List[str]
    
    # Context
    macro_score: float # Net Relative Strength
    technical_score: float
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "PENDING" # PENDING, ACTIVE, EXECUTED, CANCELLED, EXPIRED
    
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "type": self.trade_type.value,
            "direction": self.direction.value,
            "trigger": self.trigger_price,
            "confidence": self.confidence_score,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }
