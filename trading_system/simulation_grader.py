from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


class SimulationGrader:
    """
    Grades the performance of the trading system / simulation.

    Weight hierarchy (PnL prioritised above all else):
      1. Total PnL            (35%) — Absolute return quality — HIGHEST WEIGHT
      2. PnL Per Trade        (25%) — Average dollar P&L per trade — 2nd HIGHEST
      3. Win Rate             (20%) — Decision accuracy / hit rate
      4. Risk Management      (12%) — Drawdown control and loss containment
      5. Sharpe / Volatility   (8%) — Risk-adjusted return quality — LOWEST WEIGHT
    """

    def __init__(self):
        pass

    def calculate_grade(
        self,
        total_pnl_dollars: float,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate: float,
        trade_count: int,
        pnl_per_trade_dollars: float = None,  # Optional explicit override
    ) -> Dict:
        """
        Calculate a composite simulation grade.

        Parameters
        ----------
        total_pnl_dollars     : Absolute P&L in dollars (realized + unrealized)
        total_return_pct      : Total return as a decimal (e.g. 0.12 = 12%)
        sharpe_ratio          : Annualised Sharpe ratio
        max_drawdown_pct      : Maximum drawdown as a decimal (e.g. 0.05 = 5%)
        win_rate              : Fraction of winning trades (e.g. 0.55 = 55%)
        trade_count           : Number of completed trades
        pnl_per_trade_dollars : Average dollar P&L per trade. If not provided,
                                derived from total_pnl_dollars / trade_count.

        Returns
        -------
        dict containing: final_grade, final_score, breakdown, weights, metrics,
                         pnl_per_trade_dollars (always present as a separate key)
        """

        # ── Derive PnL per trade ──────────────────────────────────────────────
        if pnl_per_trade_dollars is not None:
            ppt = pnl_per_trade_dollars
        elif trade_count > 0:
            ppt = total_pnl_dollars / trade_count
        else:
            ppt = 0.0

        # ─────────────────────────────────────────────────────────────────────
        # 1. TOTAL PNL SCORE (0–100)  — Weight 35%  [HIGHEST]
        # ─────────────────────────────────────────────────────────────────────
        # Scoring curve:
        #   $0      → 40 pts (baseline)
        #   $5 000  → ~53 pts
        #   $20 000 → ~90 pts
        #   $40 000 → 100 pts (cap)
        #   Negative PnL → score decays below 40, floors at 0
        if total_pnl_dollars >= 0:
            total_pnl_score = min(100.0, 40.0 + (total_pnl_dollars / 40_000.0) * 60.0)
        else:
            # Penalty: lose 1 pt per $400 of loss, floor at 0
            total_pnl_score = max(0.0, 40.0 + (total_pnl_dollars / 400.0))

        # ─────────────────────────────────────────────────────────────────────
        # 2. PNL PER TRADE SCORE (0–100)  — Weight 25%  [2nd HIGHEST]
        # ─────────────────────────────────────────────────────────────────────
        # Scoring curve:
        #   $0/trade  → 50 pts (neutral)
        #   $200/trade → 60 pts
        #   $1 000/trade → 75 pts
        #   $2 000+/trade → 100 pts (cap)
        #   Negative → decays below 50, floors at 0
        if ppt >= 0:
            ppt_score = min(100.0, 50.0 + (ppt / 2_000.0) * 50.0)
        else:
            ppt_score = max(0.0, 50.0 + (ppt / 200.0) * 10.0)

        # ─────────────────────────────────────────────────────────────────────
        # 3. WIN RATE SCORE (0–100)  — Weight 20%
        # ─────────────────────────────────────────────────────────────────────
        # Scoring curve:
        #   35% win rate → 0 pts  (floor — random-walk territory)
        #   50% win rate → 60 pts (break-even)
        #   60% win rate → 100 pts (cap)
        if win_rate > 0.35:
            win_rate_score = min(100.0, (win_rate - 0.35) * (100.0 / 0.25))
        else:
            win_rate_score = 0.0

        # ─────────────────────────────────────────────────────────────────────
        # 4. RISK MANAGEMENT SCORE (0–100)  — Weight 12%
        # ─────────────────────────────────────────────────────────────────────
        # Based on max drawdown: lower drawdown = higher score
        #   0%   drawdown → 100 pts
        #   5%   drawdown → 75 pts
        #   10%  drawdown → 50 pts
        #   20%+ drawdown → 0 pts
        dd_pct = abs(max_drawdown_pct) * 100.0  # convert to percentage points
        risk_score = max(0.0, 100.0 - dd_pct * 5.0)

        # ─────────────────────────────────────────────────────────────────────
        # 5. SHARPE / VOLATILITY SCORE (0–100)  — Weight 8%  [LOWEST]
        # ─────────────────────────────────────────────────────────────────────
        # Scoring curve:
        #   Sharpe 0  → 40 pts
        #   Sharpe 1  → 60 pts
        #   Sharpe 2  → 80 pts
        #   Sharpe 2.5+ → 100 pts
        if sharpe_ratio >= 0:
            sharpe_score = min(100.0, 40.0 + sharpe_ratio * 24.0)
        else:
            sharpe_score = max(0.0, 40.0 + sharpe_ratio * 24.0)

        # ─────────────────────────────────────────────────────────────────────
        # WEIGHTED COMPOSITE
        # ─────────────────────────────────────────────────────────────────────
        final_score = (
            total_pnl_score * 0.35
            + ppt_score * 0.25
            + win_rate_score * 0.20
            + risk_score * 0.12
            + sharpe_score * 0.08
        )

        # ─────────────────────────────────────────────────────────────────────
        # LETTER GRADE
        # ─────────────────────────────────────────────────────────────────────
        if final_score >= 90:
            grade = "A+"
        elif final_score >= 85:
            grade = "A"
        elif final_score >= 80:
            grade = "A-"
        elif final_score >= 75:
            grade = "B+"
        elif final_score >= 70:
            grade = "B"
        elif final_score >= 65:
            grade = "B-"
        elif final_score >= 60:
            grade = "C+"
        elif final_score >= 55:
            grade = "C"
        elif final_score >= 50:
            grade = "C-"
        elif final_score >= 45:
            grade = "D"
        else:
            grade = "F"

        return {
            # ── Top-level summary ─────────────────────────────────────────────
            "final_grade": grade,
            "final_score": round(final_score, 1),
            "total_pnl_dollars": round(total_pnl_dollars, 2),
            # ── Per-metric scores (all 0–100) ─────────────────────────────────
            "breakdown": {
                "total_pnl_score": round(total_pnl_score, 1),
                "pnl_per_trade_score": round(ppt_score, 1),
                "win_rate_score": round(win_rate_score, 1),
                "risk_score": round(risk_score, 1),
                "sharpe_score": round(sharpe_score, 1),
            },
            # ── Weights (for display / transparency) ──────────────────────────
            "weights": {
                "total_pnl_weight": "35%",
                "pnl_per_trade_weight": "25%",
                "win_rate_weight": "20%",
                "risk_management_weight": "12%",
                "sharpe_weight": "8%",
            },
            # ── Human-readable metric values ──────────────────────────────────
            "metrics": {
                "pnl_dollars": f"${total_pnl_dollars:,.2f}",
                "pnl_per_trade": f"${ppt:,.2f}",
                "return_pct": f"{total_return_pct * 100:.2f}%",
                "sharpe": f"{sharpe_ratio:.2f}",
                "max_drawdown": f"{max_drawdown_pct * 100:.2f}%",
                "win_rate": f"{win_rate * 100:.1f}%",
                "total_trades": trade_count,
            },
            # ── Explicit separate PnL-per-trade metric (required by spec) ─────
            "pnl_per_trade_dollars": round(ppt, 2),
        }
