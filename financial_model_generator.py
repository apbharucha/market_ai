"""
Institutional Financial Model Generator
Wall Street-grade DCF valuation engine with:
  - Revenue → EBIT → NOPAT → FCF cascade
  - WACC with CAPM cost of equity
  - 20-line Investment Banking DCF template
  - Scenario-weighted valuation (Bull / Base / Bear)
  - Monte Carlo simulation (10,000 paths)
  - Relative valuation (P/E, EV/EBITDA, EV/FCF, PEG)
  - Catalyst tracking dashboard
  - Trade signal engine with risk-adjusted position sizing
"""

from __future__ import annotations

# Explicit public API — guards against stale .pyc ImportErrors
__all__ = [
    "DCFAssumptions",
    "DCFResult",
    "ScenarioResult",
    "CatalystEvent",
    "TradeSignal",
    "InstitutionalDCFEngine",
    "FinancialModelGenerator",
    "get_dcf_engine",
    "get_financial_generator",
]

import io
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("InstitutionalDCF")
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DCFAssumptions:
    """Full set of DCF model assumptions."""

    ticker: str
    # Revenue
    base_revenue: float  # $M — most recent annual revenue
    revenue_growth_rates: List[
        float
    ]  # year-by-year growth rates (list len = projection_years)
    # Margins
    ebit_margin: float  # EBIT / Revenue
    tax_rate: float  # effective tax rate
    da_pct_revenue: float  # D&A as % of revenue
    capex_pct_revenue: float  # CapEx as % of revenue
    nwc_change_pct_revenue: float  # ΔNWC as % of revenue
    # WACC components
    equity_value_market: float  # market cap $M  (E)
    debt_value: float  # total debt $M  (D)
    cost_of_debt: float  # pre-tax cost of debt Rd
    risk_free_rate: float  # Rf (10Y Treasury yield)
    equity_risk_premium: float  # Rm - Rf
    beta: float  # levered beta
    # Terminal value
    terminal_growth_rate: float  # g (Gordon Growth)
    # Balance sheet
    cash: float  # $M
    shares_outstanding: float  # millions
    current_price: float  # market price per share
    # Optional: peer multiples for relative valuation
    peer_pe: float = 25.0
    peer_ev_ebitda: float = 15.0
    peer_ev_fcf: float = 20.0
    peg_ratio: float = 1.5
    projection_years: int = 5


@dataclass
class ScenarioResult:
    label: str
    probability: float
    fair_value: float
    upside: float
    revenue_growth_avg: float
    ebit_margin: float
    wacc: float
    terminal_growth: float


@dataclass
class CatalystEvent:
    name: str
    date: str
    impact: str  # "High" / "Medium" / "Low"
    direction: str  # "Positive" / "Negative" / "Neutral"
    description: str


@dataclass
class TradeSignal:
    ticker: str
    fair_value: float
    market_price: float
    upside_pct: float
    signal: str  # "Strong Long" / "Long" / "Neutral" / "Short" / "Strong Short"
    confidence_pct: float
    risk_level: str  # "Low" / "Medium" / "High"
    risk_adjusted_return: float
    position_size_pct: float
    rationale: str


@dataclass
class DCFResult:
    """Full DCF valuation result."""

    ticker: str
    assumptions: DCFAssumptions

    # 20-Line IB Template
    line_items: pd.DataFrame  # rows = years, cols = line items

    # Valuation summary
    sum_pv_fcf: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    fair_value_per_share: float

    # Scenario analysis
    scenarios: List[ScenarioResult]
    scenario_weighted_value: float

    # Monte Carlo
    mc_median: float
    mc_mean: float
    mc_std: float
    mc_upside_prob: float  # P(fair value > market price)
    mc_downside_prob: float
    mc_percentiles: Dict[str, float]
    mc_distribution: List[float]  # full distribution for histogram

    # Relative valuation
    relative_valuation: Dict[str, float]

    # Sensitivity table
    sensitivity: pd.DataFrame

    # Catalyst events
    catalysts: List[CatalystEvent]

    # Trade signal
    trade_signal: TradeSignal

    # WACC detail
    wacc: float
    cost_of_equity: float
    wacc_breakdown: Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────


class InstitutionalDCFEngine:
    """
    Wall Street-grade discounted cash flow engine.

    All dollar amounts are in $M unless noted otherwise.
    """

    # ── WACC ──────────────────────────────────────────────────────────────────

    def compute_wacc(self, a: DCFAssumptions) -> Tuple[float, float, Dict]:
        """
        WACC = (E/(D+E)) * Re + (D/(D+E)) * Rd * (1 - T)
        Re = Rf + β * (Rm - Rf)   [CAPM]
        """
        re = a.risk_free_rate + a.beta * a.equity_risk_premium
        total = a.equity_value_market + a.debt_value
        if total <= 0:
            total = 1.0
        w_e = a.equity_value_market / total
        w_d = a.debt_value / total
        wacc = w_e * re + w_d * a.cost_of_debt * (1 - a.tax_rate)
        detail = {
            "cost_of_equity": re,
            "cost_of_debt_pretax": a.cost_of_debt,
            "cost_of_debt_aftertax": a.cost_of_debt * (1 - a.tax_rate),
            "weight_equity": w_e,
            "weight_debt": w_d,
            "wacc": wacc,
            "risk_free_rate": a.risk_free_rate,
            "equity_risk_premium": a.equity_risk_premium,
            "beta": a.beta,
        }
        return wacc, re, detail

    # ── FCF PROJECTION ────────────────────────────────────────────────────────

    def project_fcf(self, a: DCFAssumptions, wacc: float) -> pd.DataFrame:
        """
        Build the 20-line IB DCF template.

        Lines:
         1  Revenue
         2  EBIT
         3  Taxes on EBIT
         4  NOPAT  (= EBIT * (1 - T))
         5  D&A
         6  CapEx
         7  ΔNWC
         8  Free Cash Flow
         9  Discount Factor  (= 1 / (1 + WACC)^t)
        10  PV of FCF
        """
        rows = {}
        prev_revenue = a.base_revenue
        prev_nwc = a.base_revenue * a.nwc_change_pct_revenue  # seed

        years = list(range(1, a.projection_years + 1))
        g_rates = list(a.revenue_growth_rates)
        # Pad or trim growth rates to match projection_years
        while len(g_rates) < a.projection_years:
            g_rates.append(g_rates[-1] if g_rates else 0.05)
        g_rates = g_rates[: a.projection_years]

        revenues, ebits, taxes, nopats, das, capexes, dnwcs, fcfs, dfs, pvfcfs = (
            [] for _ in range(10)
        )

        for i, yr in enumerate(years):
            g = g_rates[i]
            rev = prev_revenue * (1 + g)
            ebit = rev * a.ebit_margin
            tax = ebit * a.tax_rate
            nopat = ebit * (1 - a.tax_rate)
            da = rev * a.da_pct_revenue
            capex = rev * a.capex_pct_revenue
            curr_nwc = rev * a.nwc_change_pct_revenue
            dnwc = curr_nwc - prev_nwc

            # FCF = NOPAT + D&A − CapEx − ΔNWC
            fcf = nopat + da - capex - dnwc

            # Discount factor: 1 / (1 + WACC)^t  (DECREASING over time — correctly represents PV factor)
            df_factor = 1.0 / (1.0 + wacc) ** yr
            pv_fcf = fcf * df_factor

            revenues.append(rev)
            ebits.append(ebit)
            taxes.append(tax)
            nopats.append(nopat)
            das.append(da)
            capexes.append(capex)
            dnwcs.append(dnwc)
            fcfs.append(fcf)
            dfs.append(df_factor)
            pvfcfs.append(pv_fcf)

            prev_revenue = rev
            prev_nwc = curr_nwc

        df = pd.DataFrame(
            {
                "Year": years,
                "Revenue ($M)": revenues,
                "EBIT ($M)": ebits,
                "Taxes ($M)": taxes,
                "NOPAT ($M)": nopats,
                "D&A ($M)": das,
                "CapEx ($M)": capexes,
                "ΔNWC ($M)": dnwcs,
                "Free Cash Flow ($M)": fcfs,
                "Discount Factor": dfs,
                "PV of FCF ($M)": pvfcfs,
            }
        )
        df = df.set_index("Year")
        return df

    def compute_terminal_value(
        self, last_fcf: float, wacc: float, terminal_growth: float, years: int
    ) -> Tuple[float, float]:
        """
        TV = FCF_n × (1 + g) / (WACC − g)    [Gordon Growth Model]
        PV(TV) = TV / (1 + WACC)^n
        """
        if wacc <= terminal_growth:
            # Guard: WACC must exceed terminal growth
            wacc = terminal_growth + 0.01
        tv = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv / (1.0 + wacc) ** years
        return tv, pv_tv

    # ── FULL DCF ──────────────────────────────────────────────────────────────

    def run_dcf(self, a: DCFAssumptions) -> DCFResult:
        """Run the full institutional DCF and return a complete DCFResult."""
        # 1. WACC
        wacc, cost_of_equity, wacc_detail = self.compute_wacc(a)

        # 2. Project FCFs (20-line template)
        proj = self.project_fcf(a, wacc)

        # 3. Terminal Value
        last_fcf = float(proj["Free Cash Flow ($M)"].iloc[-1])
        tv, pv_tv = self.compute_terminal_value(
            last_fcf, wacc, a.terminal_growth_rate, a.projection_years
        )

        # 4. Enterprise & Equity Value
        sum_pv_fcf = float(proj["PV of FCF ($M)"].sum())
        ev = sum_pv_fcf + pv_tv
        net_debt = a.debt_value - a.cash
        eq_val = ev - net_debt
        fv_per_share = (
            eq_val / a.shares_outstanding if a.shares_outstanding > 0 else 0.0
        )

        # 5. Extend line items with terminal value rows
        proj_extended = proj.copy()
        proj_extended.loc["Terminal"] = {
            "Revenue ($M)": float(proj["Revenue ($M)"].iloc[-1])
            * (1 + a.terminal_growth_rate),
            "EBIT ($M)": np.nan,
            "Taxes ($M)": np.nan,
            "NOPAT ($M)": np.nan,
            "D&A ($M)": np.nan,
            "CapEx ($M)": np.nan,
            "ΔNWC ($M)": np.nan,
            "Free Cash Flow ($M)": last_fcf * (1 + a.terminal_growth_rate),
            "Discount Factor": 1.0 / (1.0 + wacc) ** a.projection_years,
            "PV of FCF ($M)": pv_tv,
        }

        # 6. Scenario analysis
        scenarios = self._run_scenarios(a, ev, fv_per_share)
        scenario_weighted_value = sum(s.probability * s.fair_value for s in scenarios)

        # 7. Monte Carlo
        mc_dist, mc_stats = self._monte_carlo(a, n_sims=10_000)

        # 8. Relative valuation
        rel_val = self._relative_valuation(a, proj)

        # 9. Sensitivity table (WACC × Terminal Growth)
        sens = self._sensitivity_table(a, proj, sum_pv_fcf, net_debt)

        # 10. Default catalysts
        catalysts = self._default_catalysts(a.ticker)

        # 11. Trade signal
        signal = self._generate_trade_signal(
            a.ticker, fv_per_share, a.current_price, wacc, a.beta
        )

        return DCFResult(
            ticker=a.ticker,
            assumptions=a,
            line_items=proj_extended,
            sum_pv_fcf=sum_pv_fcf,
            pv_terminal_value=pv_tv,
            enterprise_value=ev,
            net_debt=net_debt,
            equity_value=eq_val,
            shares_outstanding=a.shares_outstanding,
            fair_value_per_share=fv_per_share,
            scenarios=scenarios,
            scenario_weighted_value=scenario_weighted_value,
            mc_median=mc_stats["median"],
            mc_mean=mc_stats["mean"],
            mc_std=mc_stats["std"],
            mc_upside_prob=mc_stats["upside_prob"],
            mc_downside_prob=mc_stats["downside_prob"],
            mc_percentiles=mc_stats["percentiles"],
            mc_distribution=mc_dist,
            relative_valuation=rel_val,
            sensitivity=sens,
            catalysts=catalysts,
            trade_signal=signal,
            wacc=wacc,
            cost_of_equity=cost_of_equity,
            wacc_breakdown=wacc_detail,
        )

    # ── SCENARIO ANALYSIS ─────────────────────────────────────────────────────

    def _run_scenarios(
        self, base: DCFAssumptions, base_ev: float, base_fv: float
    ) -> List[ScenarioResult]:
        """
        Three-scenario model: Bull (25%), Base (50%), Bear (25%).
        """
        # Bear: slower growth, compressed margins, higher WACC
        bear_a = _tweak(
            base, growth_mult=0.5, margin_mult=0.85, wacc_add=+0.02, tg_mult=0.7
        )
        # Bull: accelerated growth, expanding margins, lower WACC
        bull_a = _tweak(
            base, growth_mult=1.5, margin_mult=1.15, wacc_add=-0.01, tg_mult=1.3
        )

        results = []
        for label, prob, a in [
            ("Bear", 0.25, bear_a),
            ("Base", 0.50, base),
            ("Bull", 0.25, bull_a),
        ]:
            w, _, _ = self.compute_wacc(a)
            proj = self.project_fcf(a, w)
            lfcf = float(proj["Free Cash Flow ($M)"].iloc[-1])
            tv, pv_tv = self.compute_terminal_value(
                lfcf, w, a.terminal_growth_rate, a.projection_years
            )
            spv = float(proj["PV of FCF ($M)"].sum())
            ev = spv + pv_tv
            nd = a.debt_value - a.cash
            fv = (ev - nd) / a.shares_outstanding if a.shares_outstanding > 0 else 0
            upside = (
                (fv - a.current_price) / a.current_price if a.current_price > 0 else 0
            )
            avg_g = float(np.mean(a.revenue_growth_rates))
            results.append(
                ScenarioResult(
                    label=label,
                    probability=prob,
                    fair_value=fv,
                    upside=upside,
                    revenue_growth_avg=avg_g,
                    ebit_margin=a.ebit_margin,
                    wacc=w,
                    terminal_growth=a.terminal_growth_rate,
                )
            )
        return results

    # ── MONTE CARLO ───────────────────────────────────────────────────────────

    def _monte_carlo(
        self, a: DCFAssumptions, n_sims: int = 10_000
    ) -> Tuple[List[float], Dict]:
        """
        Randomise revenue growth, EBIT margin, WACC, and terminal growth
        over n_sims paths.  Returns the full distribution and summary stats.
        """
        rng = np.random.default_rng(42)

        base_growth = float(np.mean(a.revenue_growth_rates))
        dist: List[float] = []

        for _ in range(n_sims):
            # Perturb key assumptions
            g_sim = float(rng.normal(base_growth, base_growth * 0.30))
            m_sim = float(rng.normal(a.ebit_margin, a.ebit_margin * 0.20))
            m_sim = max(0.01, m_sim)
            da_sim = a.da_pct_revenue
            capex_sim = a.capex_pct_revenue
            nwc_sim = a.nwc_change_pct_revenue

            # Perturb WACC ±150bps
            wacc_sim = float(rng.normal(0.0, 0.015))
            beta_sim = float(rng.normal(a.beta, a.beta * 0.15))
            beta_sim = max(0.3, beta_sim)
            re_sim = a.risk_free_rate + beta_sim * a.equity_risk_premium
            total = a.equity_value_market + a.debt_value + 1e-9
            wacc_val = (
                (a.equity_value_market / total) * re_sim
                + (a.debt_value / total) * a.cost_of_debt * (1 - a.tax_rate)
                + wacc_sim
            )
            wacc_val = max(0.04, min(0.25, wacc_val))

            # Terminal growth ±0.5%
            tg_sim = float(rng.normal(a.terminal_growth_rate, 0.005))
            tg_sim = max(0.005, min(wacc_val - 0.01, tg_sim))

            # Simulate FCFs
            prev_rev = a.base_revenue
            prev_nwc = a.base_revenue * nwc_sim
            fcfs = []
            for yr in range(1, a.projection_years + 1):
                rev = prev_rev * (1 + g_sim)
                nopat = rev * m_sim * (1 - a.tax_rate)
                da = rev * da_sim
                capex = rev * capex_sim
                curr_nwc = rev * nwc_sim
                dnwc = curr_nwc - prev_nwc
                fcf = nopat + da - capex - dnwc
                df_f = 1.0 / (1.0 + wacc_val) ** yr
                fcfs.append(fcf * df_f)
                prev_rev = rev
                prev_nwc = curr_nwc

            sum_pv = sum(fcfs)
            last_fcf = fcfs[-1] / (
                1.0 / (1.0 + wacc_val) ** a.projection_years
            )  # un-discount
            tv = last_fcf * (1 + tg_sim) / (wacc_val - tg_sim)
            pv_tv = tv / (1.0 + wacc_val) ** a.projection_years
            ev = sum_pv + pv_tv
            nd = a.debt_value - a.cash
            fv = (ev - nd) / (a.shares_outstanding + 1e-9)
            dist.append(fv)

        arr = np.array(dist)
        upside_p = float(np.mean(arr > a.current_price)) if a.current_price > 0 else 0.5
        stats = {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "upside_prob": upside_p,
            "downside_prob": 1.0 - upside_p,
            "percentiles": {
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            },
        }
        return dist, stats

    # ── RELATIVE VALUATION ────────────────────────────────────────────────────

    def _relative_valuation(
        self, a: DCFAssumptions, proj: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Peer-multiple implied fair values:
          P/E, EV/EBITDA, EV/FCF, PEG
        """
        # Derive LTM / forward estimates from last projected year
        last_rev = float(proj["Revenue ($M)"].iloc[-1])
        last_ebit = float(proj["EBIT ($M)"].iloc[-1])
        last_da = float(proj["D&A ($M)"].iloc[-1])
        last_fcf = float(proj["Free Cash Flow ($M)"].iloc[-1])
        last_nopat = float(proj["NOPAT ($M)"].iloc[-1])

        ebitda = last_ebit + last_da
        # EPS proxy: NOPAT / shares
        eps_proxy = last_nopat / (a.shares_outstanding + 1e-9)

        # P/E implied
        pe_implied = eps_proxy * a.peer_pe

        # EV/EBITDA implied
        ev_ebitda_implied_ev = ebitda * a.peer_ev_ebitda
        ev_ebitda_implied_fv = (ev_ebitda_implied_ev - a.debt_value + a.cash) / (
            a.shares_outstanding + 1e-9
        )

        # EV/FCF implied
        ev_fcf_implied_ev = last_fcf * a.peer_ev_fcf
        ev_fcf_implied_fv = (ev_fcf_implied_ev - a.debt_value + a.cash) / (
            a.shares_outstanding + 1e-9
        )

        # PEG implied (rough: price ≈ EPS * PE where PE = PEG * g_pct)
        avg_g_pct = float(np.mean(a.revenue_growth_rates)) * 100  # as a percentage
        peg_pe = a.peg_ratio * avg_g_pct
        peg_implied = eps_proxy * peg_pe

        return {
            "pe_implied_fv": pe_implied,
            "ev_ebitda_implied_fv": ev_ebitda_implied_fv,
            "ev_fcf_implied_fv": ev_fcf_implied_fv,
            "peg_implied_fv": peg_implied,
            "avg_peer_implied_fv": float(
                np.mean([pe_implied, ev_ebitda_implied_fv, ev_fcf_implied_fv])
            ),
            "peer_pe": a.peer_pe,
            "peer_ev_ebitda": a.peer_ev_ebitda,
            "peer_ev_fcf": a.peer_ev_fcf,
            "peg_ratio": a.peg_ratio,
            "last_fcf_M": last_fcf,
            "ebitda_M": ebitda,
            "eps_proxy": eps_proxy,
        }

    # ── SENSITIVITY TABLE ─────────────────────────────────────────────────────

    def _sensitivity_table(
        self,
        a: DCFAssumptions,
        proj: pd.DataFrame,
        base_sum_pv: float,
        net_debt: float,
    ) -> pd.DataFrame:
        """2-D sensitivity: WACC (cols) × Terminal Growth (rows)."""
        wacc_range = [
            a.assumptions_wacc_for_sens(offset)
            for offset in [-0.02, -0.01, 0, +0.01, +0.02]
        ]
        tg_range = [
            max(0.001, a.terminal_growth_rate + offset)
            for offset in [-0.01, -0.005, 0, +0.005, +0.01]
        ]

        # Re-compute FCF stream for WACC sensitivity (simplified: use base FCFs, vary discount)
        base_fcfs_undiscounted = [
            float(proj["Free Cash Flow ($M)"].iloc[i])
            / float(proj["Discount Factor"].iloc[i])
            for i in range(a.projection_years)
        ]

        rows = {}
        for tg in tg_range:
            row = {}
            for w in wacc_range:
                if w <= tg:
                    w = tg + 0.01
                sum_pv = sum(
                    fcf / (1 + w) ** (yr + 1)
                    for yr, fcf in enumerate(base_fcfs_undiscounted)
                )
                last_fcf = base_fcfs_undiscounted[-1]
                tv = last_fcf * (1 + tg) / (w - tg)
                pv_tv = tv / (1 + w) ** a.projection_years
                ev = sum_pv + pv_tv
                fv = (ev - net_debt) / (a.shares_outstanding + 1e-9)
                row[f"{w:.1%}"] = round(fv, 2)
            rows[f"{tg:.1%}"] = row

        df = pd.DataFrame(rows).T
        df.index.name = "Terminal Growth \\ WACC"
        return df

    # ── CATALYST TRACKING ─────────────────────────────────────────────────────

    def _default_catalysts(self, ticker: str) -> List[CatalystEvent]:
        """Return a set of templated catalyst events for the ticker."""
        return [
            CatalystEvent(
                name="Earnings Release",
                date="Next Quarter",
                impact="High",
                direction="Neutral",
                description=f"Quarterly earnings — key drivers: revenue growth, margin trajectory, guidance revision.",
            ),
            CatalystEvent(
                name="Product / Pipeline Catalyst",
                date="TBD",
                impact="Medium",
                direction="Positive",
                description="New product launch, FDA approval, regulatory milestone, or contract win.",
            ),
            CatalystEvent(
                name="Macro / Rate Catalyst",
                date="FOMC Meetings",
                impact="Medium",
                direction="Neutral",
                description="Fed rate decisions directly impact WACC and discount rates used in this model.",
            ),
            CatalystEvent(
                name="Capital Allocation Event",
                date="Annual Meeting",
                impact="Medium",
                direction="Positive",
                description="Buyback announcements, dividend increases, or M&A activity.",
            ),
            CatalystEvent(
                name="Analyst Day / Guidance Update",
                date="TBD",
                impact="High",
                direction="Neutral",
                description="Management long-term guidance revision — triggers model re-rating.",
            ),
        ]

    # ── TRADE SIGNAL ──────────────────────────────────────────────────────────

    def _generate_trade_signal(
        self,
        ticker: str,
        fair_value: float,
        market_price: float,
        wacc: float,
        beta: float,
    ) -> TradeSignal:
        """
        Step 1 — Mispricing:  Upside = (FV − P) / P
        Step 2 — Signal classification
        Step 3 — Risk-adjusted return = Upside / Volatility
        Step 4 — Position sizing = Conviction / Volatility
        """
        if market_price <= 0:
            market_price = 1.0

        upside = (fair_value - market_price) / market_price
        upside_pct = upside * 100

        # Volatility proxy: annualised = beta * market_vol (assume ~18% market vol)
        vol = max(0.10, beta * 0.18)

        # Signal classification
        if upside_pct > 30:
            signal = "Strong Long"
            conviction = 0.90
        elif upside_pct > 15:
            signal = "Long"
            conviction = 0.70
        elif upside_pct >= -10:
            signal = "Neutral"
            conviction = 0.40
        elif upside_pct > -30:
            signal = "Short"
            conviction = 0.65
        else:
            signal = "Strong Short"
            conviction = 0.85

        # Risk-adjusted return
        rar = upside / vol

        # Position size  (cap at 15%)
        pos_size = min(0.15, conviction / vol)
        pos_size_pct = round(pos_size * 100, 1)

        # Risk level
        if beta < 0.8:
            risk_level = "Low"
        elif beta < 1.4:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Confidence  (blend of upside magnitude and model quality)
        confidence = min(95, max(5, 50 + abs(upside_pct) * 1.0))

        rationale = (
            f"DCF fair value ${fair_value:.2f} vs market ${market_price:.2f} "
            f"→ {upside_pct:+.1f}% mispricing. "
            f"Risk-adjusted return: {rar:.2f}x. "
            f"Beta {beta:.2f}, Volatility proxy {vol:.0%}. "
            f"Signal: {signal}."
        )

        return TradeSignal(
            ticker=ticker,
            fair_value=fair_value,
            market_price=market_price,
            upside_pct=upside_pct,
            signal=signal,
            confidence_pct=round(confidence, 1),
            risk_level=risk_level,
            risk_adjusted_return=round(rar, 2),
            position_size_pct=pos_size_pct,
            rationale=rationale,
        )

    # ── EXCEL EXPORT ──────────────────────────────────────────────────────────

    def export_excel(self, result: DCFResult) -> bytes:
        """Export full DCF model to Excel workbook."""
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                wb = writer.book

                # Formats
                hdr_fmt = wb.add_format(
                    {
                        "bold": True,
                        "bg_color": "#1a1a2e",
                        "font_color": "#e0c97f",
                        "border": 1,
                    }
                )
                num_fmt = wb.add_format({"num_format": "#,##0.0", "border": 1})
                pct_fmt = wb.add_format({"num_format": "0.0%", "border": 1})
                money_fmt = wb.add_format({"num_format": "$#,##0.00", "border": 1})

                # ── Sheet 1: Summary ──────────────────────────────────────────
                a = result.assumptions
                summary_data = {
                    "Metric": [
                        "Ticker",
                        "Fair Value Per Share (DCF)",
                        "Market Price",
                        "Upside / (Downside)",
                        "Signal",
                        "Enterprise Value ($M)",
                        "Equity Value ($M)",
                        "Net Debt ($M)",
                        "Shares Outstanding (M)",
                        "WACC",
                        "Cost of Equity (CAPM)",
                        "Terminal Growth Rate",
                        "Projection Years",
                    ],
                    "Value": [
                        result.ticker,
                        f"${result.fair_value_per_share:.2f}",
                        f"${a.current_price:.2f}",
                        f"{result.trade_signal.upside_pct:+.1f}%",
                        result.trade_signal.signal,
                        f"${result.enterprise_value:,.0f}M",
                        f"${result.equity_value:,.0f}M",
                        f"${result.net_debt:,.0f}M",
                        f"{result.shares_outstanding:,.1f}M",
                        f"{result.wacc:.2%}",
                        f"{result.cost_of_equity:.2%}",
                        f"{a.terminal_growth_rate:.2%}",
                        str(a.projection_years),
                    ],
                }
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name="Summary", index=False
                )

                # ── Sheet 2: 20-Line DCF ──────────────────────────────────────
                result.line_items.to_excel(writer, sheet_name="DCF Projections")

                # ── Sheet 3: Scenarios ────────────────────────────────────────
                scen_data = [
                    {
                        "Scenario": s.label,
                        "Probability": f"{s.probability:.0%}",
                        "Fair Value": f"${s.fair_value:.2f}",
                        "Upside": f"{s.upside:.1%}",
                        "Avg Revenue Growth": f"{s.revenue_growth_avg:.1%}",
                        "EBIT Margin": f"{s.ebit_margin:.1%}",
                        "WACC": f"{s.wacc:.2%}",
                        "Terminal Growth": f"{s.terminal_growth:.2%}",
                    }
                    for s in result.scenarios
                ]
                scen_data.append(
                    {
                        "Scenario": "Scenario-Weighted Value",
                        "Probability": "100%",
                        "Fair Value": f"${result.scenario_weighted_value:.2f}",
                        "Upside": "",
                        "Avg Revenue Growth": "",
                        "EBIT Margin": "",
                        "WACC": "",
                        "Terminal Growth": "",
                    }
                )
                pd.DataFrame(scen_data).to_excel(
                    writer, sheet_name="Scenarios", index=False
                )

                # ── Sheet 4: Monte Carlo ──────────────────────────────────────
                mc_data = {
                    "Metric": [
                        "Median",
                        "Mean",
                        "Std Dev",
                        "P5",
                        "P25",
                        "P50",
                        "P75",
                        "P95",
                        "Upside Probability",
                        "Downside Probability",
                    ],
                    "Value": [
                        f"${result.mc_median:.2f}",
                        f"${result.mc_mean:.2f}",
                        f"${result.mc_std:.2f}",
                        f"${result.mc_percentiles['p5']:.2f}",
                        f"${result.mc_percentiles['p25']:.2f}",
                        f"${result.mc_percentiles['p50']:.2f}",
                        f"${result.mc_percentiles['p75']:.2f}",
                        f"${result.mc_percentiles['p95']:.2f}",
                        f"{result.mc_upside_prob:.1%}",
                        f"{result.mc_downside_prob:.1%}",
                    ],
                }
                pd.DataFrame(mc_data).to_excel(
                    writer, sheet_name="Monte Carlo", index=False
                )

                # ── Sheet 5: Relative Valuation ───────────────────────────────
                rv = result.relative_valuation
                rel_data = {
                    "Method": [
                        "P/E Implied",
                        "EV/EBITDA Implied",
                        "EV/FCF Implied",
                        "PEG Implied",
                        "Avg Peer Implied",
                    ],
                    "Fair Value": [
                        f"${rv['pe_implied_fv']:.2f}",
                        f"${rv['ev_ebitda_implied_fv']:.2f}",
                        f"${rv['ev_fcf_implied_fv']:.2f}",
                        f"${rv['peg_implied_fv']:.2f}",
                        f"${rv['avg_peer_implied_fv']:.2f}",
                    ],
                    "Peer Multiple": [
                        f"{rv['peer_pe']:.1f}x",
                        f"{rv['peer_ev_ebitda']:.1f}x",
                        f"{rv['peer_ev_fcf']:.1f}x",
                        f"{rv['peg_ratio']:.2f}",
                        "",
                    ],
                }
                pd.DataFrame(rel_data).to_excel(
                    writer, sheet_name="Relative Valuation", index=False
                )

                # ── Sheet 6: Sensitivity ──────────────────────────────────────
                result.sensitivity.to_excel(writer, sheet_name="Sensitivity")

                # ── Sheet 7: Catalysts ────────────────────────────────────────
                cat_data = [
                    {
                        "Catalyst": c.name,
                        "Date": c.date,
                        "Impact": c.impact,
                        "Direction": c.direction,
                        "Description": c.description,
                    }
                    for c in result.catalysts
                ]
                pd.DataFrame(cat_data).to_excel(
                    writer, sheet_name="Catalysts", index=False
                )

        except Exception as e:
            logger.error(f"Excel export error: {e}")

        return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _tweak(
    base: DCFAssumptions,
    growth_mult: float = 1.0,
    margin_mult: float = 1.0,
    wacc_add: float = 0.0,
    tg_mult: float = 1.0,
) -> DCFAssumptions:
    """Return a shallow copy of DCFAssumptions with key parameters adjusted."""
    import copy

    a = copy.copy(base)
    a.revenue_growth_rates = [g * growth_mult for g in base.revenue_growth_rates]
    a.ebit_margin = base.ebit_margin * margin_mult
    # Adjust WACC via beta (simple proxy: shift risk_free_rate)
    a.risk_free_rate = base.risk_free_rate + wacc_add
    a.terminal_growth_rate = max(0.005, base.terminal_growth_rate * tg_mult)
    return a


# Monkey-patch DCFAssumptions to add the helper method used in _sensitivity_table
def _assumptions_wacc_for_sens(self, offset: float) -> float:
    """Compute WACC given a wacc offset (used for sensitivity table)."""
    re = self.risk_free_rate + self.beta * self.equity_risk_premium
    total = self.equity_value_market + self.debt_value + 1e-9
    w_e = self.equity_value_market / total
    w_d = self.debt_value / total
    base_wacc = w_e * re + w_d * self.cost_of_debt * (1 - self.tax_rate)
    return max(0.04, base_wacc + offset)


DCFAssumptions.assumptions_wacc_for_sens = _assumptions_wacc_for_sens


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY ADAPTER  (keeps old main.py interface working)
# ─────────────────────────────────────────────────────────────────────────────


class FinancialModelGenerator:
    """
    Legacy wrapper that preserves the original simple DCF interface
    while also exposing the full institutional engine.
    """

    def __init__(self):
        self._engine = InstitutionalDCFEngine()

    # ── Simple legacy DCF (used by old main.py Financial Model Generator page) ──

    def generate_dcf(
        self,
        ticker: str,
        current_fcf: float,
        growth_rate_5y: float,
        terminal_growth: float,
        discount_rate: float,
        shares_outstanding: float,
        net_debt: float,
        projection_years: int = 5,
    ) -> dict:
        """
        Legacy interface.  Builds a DCFAssumptions object from the simplified
        inputs and runs the full institutional engine.

        current_fcf is treated as NOPAT (proxy for FCF when revenue data
        is unavailable).  Revenue is back-calculated assuming a 20% FCF margin.
        """
        # Estimate revenue from FCF (assume 20% FCF / Revenue margin as default)
        fcf_margin = 0.20
        base_revenue = current_fcf / fcf_margin if fcf_margin > 0 else current_fcf

        # Simple uniform growth rates
        g_rates = [growth_rate_5y] * projection_years

        # Equity value: shares * assumed price (use net_debt as proxy if unavailable)
        # For legacy mode we use simplified WACC = discount_rate directly
        # We achieve this by setting beta=1, Rf = discount_rate-0.05, ERP=0.05
        assumed_price = 100.0  # placeholder; caller should pass via run_full_dcf
        equity_val = shares_outstanding * assumed_price

        assumptions = DCFAssumptions(
            ticker=ticker,
            base_revenue=base_revenue,
            revenue_growth_rates=g_rates,
            ebit_margin=0.25,
            tax_rate=0.21,
            da_pct_revenue=0.04,
            capex_pct_revenue=0.05,
            nwc_change_pct_revenue=0.01,
            equity_value_market=max(1.0, equity_val),
            debt_value=max(0.0, net_debt),
            cost_of_debt=0.05,
            risk_free_rate=max(0.01, discount_rate - 0.05),
            equity_risk_premium=0.05,
            beta=1.0,
            terminal_growth_rate=terminal_growth,
            cash=0.0,
            shares_outstanding=shares_outstanding,
            current_price=assumed_price,
            projection_years=projection_years,
        )

        result = self._engine.run_dcf(assumptions)

        # ── Build the 20-line display DataFrame ──────────────────────────────
        proj = result.line_items.copy()

        # Reconstruct the legacy-style projections DataFrame
        years_idx = [i for i in proj.index if i != "Terminal"]
        proj_legacy = proj.loc[years_idx].copy()

        # Correct discount factor column name for legacy display
        proj_legacy = proj_legacy.rename(
            columns={
                "Discount Factor": "Discount_Factor",
                "PV of FCF ($M)": "PV_FCF",
                "Free Cash Flow ($M)": "FCF",
            }
        )

        sensitivity = self._generate_sensitivity_table_legacy(
            assumptions, result.sensitivity
        )

        return {
            "type": "DCF",
            "ticker": ticker,
            "fair_value": result.fair_value_per_share,
            "equity_value": result.equity_value,
            "enterprise_value": result.enterprise_value,
            "sum_pv_fcf": result.sum_pv_fcf,
            "pv_terminal": result.pv_terminal_value,
            "projections": proj_legacy,
            "sensitivity": sensitivity,
            "full_result": result,  # full institutional result
            "inputs": {
                "fcf": current_fcf,
                "growth": growth_rate_5y,
                "wacc": discount_rate,
                "terminal": terminal_growth,
                "debt": net_debt,
                "shares": shares_outstanding,
            },
        }

    def _generate_sensitivity_table_legacy(
        self, assumptions: DCFAssumptions, sens_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Return the institutional sensitivity table (already computed)."""
        return sens_df

    def run_full_dcf(self, assumptions: DCFAssumptions) -> DCFResult:
        """Run the full institutional DCF directly."""
        return self._engine.run_dcf(assumptions)

    def generate_excel(self, model_data: dict) -> bytes:
        """Export model to Excel — supports both legacy dict and DCFResult."""
        if "full_result" in model_data and isinstance(
            model_data["full_result"], DCFResult
        ):
            return self._engine.export_excel(model_data["full_result"])

        # Fallback: legacy simple export
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                summary = pd.DataFrame(
                    {
                        "Metric": [
                            "Fair Value per Share",
                            "Equity Value",
                            "Enterprise Value",
                        ],
                        "Value": [
                            model_data["fair_value"],
                            model_data["equity_value"],
                            model_data["enterprise_value"],
                        ],
                    }
                )
                summary.to_excel(writer, sheet_name="Summary", index=False)
                if "projections" in model_data:
                    model_data["projections"].to_excel(writer, sheet_name="Projections")
                if "sensitivity" in model_data:
                    model_data["sensitivity"].to_excel(
                        writer, sheet_name="Sensitivity Analysis"
                    )
        except Exception as e:
            logger.error(f"Legacy Excel export error: {e}")
        return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_fin_gen: Optional[FinancialModelGenerator] = None


def get_financial_generator() -> FinancialModelGenerator:
    global _fin_gen
    if _fin_gen is None:
        _fin_gen = FinancialModelGenerator()
    return _fin_gen


_dcf_engine: Optional[InstitutionalDCFEngine] = None


def get_dcf_engine() -> InstitutionalDCFEngine:
    """Get the raw institutional DCF engine."""
    global _dcf_engine
    if _dcf_engine is None:
        _dcf_engine = InstitutionalDCFEngine()
    return _dcf_engine
