"""
Advanced Risk Manager for Octavian Trading System
Implements position sizing, risk limits, correlation checks, and trade viability analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from risk_engine import correlation_matrix, portfolio_var, position_size
except ImportError:

    def correlation_matrix(symbols):
        return pd.DataFrame()

    def portfolio_var(*args, **kwargs):
        return 0.0

    def position_size(*args, **kwargs):
        return 0


try:
    from trading_system.trading_data_types import PositionSpec
except ImportError:
    try:
        from trading_data_types import PositionSpec
    except ImportError:
        PositionSpec = None


@dataclass
class RiskLimits:
    """Configuration for risk limits"""

    max_position_pct: float = 0.15  # Max 15% of capital per position
    max_portfolio_risk_pct: float = 0.02  # Max 2% total portfolio risk per trade
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss before halt
    max_total_exposure_pct: float = 0.80  # Max 80% of capital deployed
    max_sector_concentration: float = 0.30  # Max 30% in one sector
    max_correlated_positions: int = 3  # Max correlated positions
    correlation_threshold: float = 0.70  # Correlation threshold


class AdvancedRiskManager:
    """
    Advanced risk management system that:
    - Validates trades before execution
    - Calculates position sizes based on risk
    - Monitors portfolio exposure
    - Enforces risk limits
    """

    def __init__(self, total_capital: float, limits: Optional[RiskLimits] = None):
        self.total_capital = total_capital
        self.limits = limits or RiskLimits()
        self.daily_pnl = 0.0
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []

    def update_capital(self, new_capital: float):
        """Update total capital (e.g., after PnL changes)"""
        self.total_capital = new_capital

    def update_daily_pnl(self, pnl: float):
        """Update daily PnL"""
        self.daily_pnl += pnl

    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new trading day)"""
        self.daily_pnl = 0.0

    def check_daily_loss_limit(self) -> Dict[str, any]:
        """Check if daily loss limit has been breached"""
        max_loss = self.total_capital * self.limits.max_daily_loss_pct
        if self.daily_pnl < -max_loss:
            return {
                "approved": False,
                "reason": f"Daily loss limit breached: ${self.daily_pnl:,.2f} (limit: -${max_loss:,.2f})",
                "halt_trading": True,
            }
        return {"approved": True, "halt_trading": False}

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk management rules.

        Returns:
            dict with position_size_units, position_size_dollars, risk_dollars
        """
        if risk_pct is None:
            risk_pct = self.limits.max_portfolio_risk_pct

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return {
                "position_size_units": 0,
                "position_size_dollars": 0,
                "risk_dollars": 0,
                "error": "Stop loss equals entry price",
            }

        # Calculate position size based on risk
        risk_dollars = self.total_capital * risk_pct
        units = risk_dollars / risk_per_share

        # Apply max position size constraint
        max_position_dollars = self.total_capital * self.limits.max_position_pct
        max_units = max_position_dollars / entry_price

        if units * entry_price > max_position_dollars:
            units = max_units
            risk_dollars = units * risk_per_share

        return {
            "position_size_units": round(units, 2),
            "position_size_dollars": round(units * entry_price, 2),
            "risk_dollars": round(risk_dollars, 2),
            "risk_pct": round(risk_dollars / self.total_capital, 4),
        }

    def check_trade_viability(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        entry_price: Optional[float] = None,
        trade_type: str = "SPOT",
    ) -> Dict[str, any]:
        """
        Comprehensive pre-trade risk check.

        Returns:
            dict with approved (bool), reason (str), and suggested adjustments
        """
        if entry_price is None:
            entry_price = price

        # 1. Check daily loss limit
        daily_check = self.check_daily_loss_limit()
        if not daily_check["approved"]:
            return daily_check

        # 2. Check stop loss validity
        if stop_loss <= 0:
            return {"approved": False, "reason": "Invalid stop loss price"}

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share / entry_price > 0.15:  # More than 15% stop
            return {
                "approved": False,
                "reason": f"Stop loss too wide: {risk_per_share / entry_price * 100:.1f}% (max 15%)",
            }

        # 3. Calculate position size
        position_calc = self.calculate_position_size(entry_price, stop_loss)
        if "error" in position_calc:
            return {"approved": False, "reason": position_calc["error"]}

        if position_calc["position_size_units"] == 0:
            return {"approved": False, "reason": "Position size calculated as zero"}

        # 4. Check total exposure
        current_exposure = sum(
            p.get("position_dollars", 0) for p in self.positions.values()
        )
        new_exposure = current_exposure + position_calc["position_size_dollars"]
        max_exposure = self.total_capital * self.limits.max_total_exposure_pct

        if new_exposure > max_exposure:
            return {
                "approved": False,
                "reason": f"Total exposure would exceed limit: ${new_exposure:,.0f} > ${max_exposure:,.0f}",
            }

        # 5. Check correlation with existing positions
        if len(self.positions) > 0:
            symbols_list = list(self.positions.keys()) + [symbol]
            try:
                corr_matrix = correlation_matrix(symbols_list)
                if not corr_matrix.empty and symbol in corr_matrix.columns:
                    high_corr_count = 0
                    for existing_symbol in self.positions.keys():
                        if existing_symbol in corr_matrix.columns:
                            corr = corr_matrix.loc[symbol, existing_symbol]
                            if abs(corr) > self.limits.correlation_threshold:
                                high_corr_count += 1

                    if high_corr_count >= self.limits.max_correlated_positions:
                        return {
                            "approved": False,
                            "reason": f"Too many correlated positions: {high_corr_count} positions with >70% correlation",
                        }
            except Exception as e:
                # If correlation check fails, log but don't block trade
                print(f"Warning: Correlation check failed for {symbol}: {e}")

        # All checks passed
        return {
            "approved": True,
            "reason": "Trade approved",
            "position_size_units": position_calc["position_size_units"],
            "position_size_dollars": position_calc["position_size_dollars"],
            "risk_dollars": position_calc["risk_dollars"],
            "risk_pct": position_calc["risk_pct"],
        }

    def register_position(self, symbol: str, position_data: Dict):
        """Register an open position for tracking"""
        self.positions[symbol] = position_data

    def close_position(self, symbol: str, exit_price: float, exit_time: str = None):
        """Close a position and record in history"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            entry_price = pos.get("entry_price", 0)
            units = pos.get("units", 0)

            if entry_price > 0 and units > 0:
                pnl = (exit_price - entry_price) * units
                self.update_daily_pnl(pnl)

                self.trade_history.append(
                    {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "units": units,
                        "pnl": pnl,
                        "exit_time": exit_time,
                    }
                )

            del self.positions[symbol]
            return True
        return False

    def get_portfolio_risk_metrics(self) -> Dict[str, any]:
        """Get current portfolio risk metrics"""
        total_exposure = sum(
            p.get("position_dollars", 0) for p in self.positions.values()
        )
        total_risk = sum(p.get("risk_dollars", 0) for p in self.positions.values())

        return {
            "total_capital": self.total_capital,
            "total_exposure": total_exposure,
            "total_exposure_pct": total_exposure / self.total_capital
            if self.total_capital > 0
            else 0,
            "total_risk_dollars": total_risk,
            "total_risk_pct": total_risk / self.total_capital
            if self.total_capital > 0
            else 0,
            "open_positions": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.total_capital
            if self.total_capital > 0
            else 0,
            "available_capital": self.total_capital - total_exposure,
        }

    def get_risk_status_summary(self) -> str:
        """Get a human-readable risk status summary"""
        metrics = self.get_portfolio_risk_metrics()

        status_lines = [
            f"Capital: ${metrics['total_capital']:,.0f}",
            f"Exposure: ${metrics['total_exposure']:,.0f} ({metrics['total_exposure_pct']:.1%})",
            f"Total Risk: ${metrics['total_risk_dollars']:,.0f} ({metrics['total_risk_pct']:.1%})",
            f"Positions: {metrics['open_positions']}",
            f"Daily P&L: ${metrics['daily_pnl']:,.0f} ({metrics['daily_pnl_pct']:.1%})",
        ]

        return " | ".join(status_lines)

    def calculate_position_sizing(
        self,
        entry_price: float,
        stop_loss: float,
        risk_pct: Optional[float] = None,
        direction: str = "LONG",
    ):
        """
        Bridge method used by TradeGenerator.
        Wraps calculate_position_size() and returns a PositionSpec dataclass
        (or a plain dict if PositionSpec is unavailable).

        Parameters
        ----------
        entry_price : float  — intended entry price
        stop_loss   : float  — stop-loss price
        risk_pct    : float  — fraction of capital to risk (defaults to limits setting)
        direction   : str    — "LONG" or "SHORT"

        Returns
        -------
        PositionSpec (or plain dict with identical fields)
        """
        result = self.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_pct=risk_pct,
        )

        units = result.get("position_size_units", 0.0)
        risk_dollars = result.get("risk_dollars", 0.0)
        position_dollars = result.get("position_size_dollars", 0.0)

        # Risk-reward: assume minimum 2:1 for primary target
        risk_dist = abs(entry_price - stop_loss)
        if direction == "LONG":
            take_profit_primary = entry_price + risk_dist * 2.0
        else:
            take_profit_primary = entry_price - risk_dist * 2.0

        take_profit_targets = [
            entry_price + risk_dist * 1.5
            if direction == "LONG"
            else entry_price - risk_dist * 1.5,
            take_profit_primary,
            entry_price + risk_dist * 3.0
            if direction == "LONG"
            else entry_price - risk_dist * 3.0,
        ]

        risk_reward = 2.0  # default 2:1

        expected_return_dollars = units * risk_dist * risk_reward
        expected_return_pct = (
            expected_return_dollars / self.total_capital
            if self.total_capital > 0
            else 0.0
        )

        risk_alloc_pct = result.get("risk_pct", self.limits.max_portfolio_risk_pct)

        # Build PositionSpec if the class is available, otherwise return a plain dict
        if PositionSpec is not None:
            return PositionSpec(
                position_size_units=units,
                position_size_dollars=position_dollars,
                risk_allocation_pct=risk_alloc_pct,
                stop_loss_price=stop_loss,
                stop_loss_dollars=risk_dollars,
                take_profit_price=take_profit_primary,
                take_profit_targets=take_profit_targets,
                risk_reward_ratio=risk_reward,
                expected_return_dollars=expected_return_dollars,
                expected_return_pct=expected_return_pct,
                expected_holding_period="2-5 Days",
            )

        # Fallback: plain dict that quacks like PositionSpec
        class _FallbackSpec:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return _FallbackSpec(
            position_size_units=units,
            position_size_dollars=position_dollars,
            risk_allocation_pct=risk_alloc_pct,
            stop_loss_price=stop_loss,
            stop_loss_dollars=risk_dollars,
            take_profit_price=take_profit_primary,
            take_profit_targets=take_profit_targets,
            risk_reward_ratio=risk_reward,
            expected_return_dollars=expected_return_dollars,
            expected_return_pct=expected_return_pct,
            expected_holding_period="2-5 Days",
        )
