"""
buffer_etf_pricing.py — Buffer ETF Portfolio Pricing Engine

Models a Buffer ETF as a portfolio of options on an underlying asset.
Uses Black-Scholes-Merton to price each leg, then aggregates into a
normalised NAV (starting at $100) so the user can analyse how the
synthetic ETF behaves under different market scenarios.

Typical buffer-ETF replication (standard buffer on SPY, 1-year):
  +1 ATM call          capture upside
  -1 OTM call          cap the upside  (e.g. strike = 115% of spot)
  -1 ATM put           fund the structure
  +1 deep OTM put      limit downside  (e.g. strike = 91% of spot)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from option_pricing import BSMInputs, bsm_price


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    """A single option leg in the portfolio."""
    is_call: bool
    strike: float
    expiry_date: date
    position: int            # +1 long, -1 short
    implied_vol: float       # annualised decimal (0.20 = 20%)
    label: str = ""


@dataclass
class PortfolioDefinition:
    """Full definition of the synthetic Buffer ETF."""
    legs: list[OptionLeg]
    underlying_start: float          # starting underlying price
    start_date: date
    risk_free_rate: float            # annualised continuous
    dividend_yield: float            # annualised continuous


@dataclass
class ScenarioParams:
    """Market scenario to evaluate the portfolio under."""
    underlying_price: float
    analysis_date: date
    vol_shift: float = 0.0           # additive shift to every leg's IV
    rate_shift: float = 0.0          # additive shift to risk-free rate


@dataclass
class LegResult:
    """Pricing result for a single leg under a scenario."""
    label: str
    is_call: bool
    strike: float
    position: int
    initial_price: float             # per-unit BSM price at inception
    current_price: float             # per-unit BSM price under scenario
    price_change: float
    pct_change: float
    position_value: float            # position * current_price (signed)
    initial_position_value: float    # position * initial_price (signed)
    delta: float
    theta: float                     # per calendar day


@dataclass
class PortfolioResult:
    """Aggregated result for the whole portfolio."""
    nav: float
    nav_return: float                # (nav - 100) / 100
    underlying_return: float
    legs: list[LegResult]
    scale_factor: float              # multiplier used to normalise to $100


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict] = {
    "Standard Buffer": {
        "description": "~9% downside buffer, capped upside (~15%). Long call + short call (cap) + short put (buffer edge).",
        "legs": [
            {"is_call": True,  "strike_pct": 1.00, "position": +1, "vol": 0.18, "label": "Long ATM Call"},
            {"is_call": True,  "strike_pct": 1.15, "position": -1, "vol": 0.16, "label": "Short OTM Call (Cap)"},
            {"is_call": False, "strike_pct": 0.91, "position": -1, "vol": 0.22, "label": "Short Put (Buffer Edge)"},
        ],
    },
    "Ultra Buffer": {
        "description": "5% gap then 30% buffer zone (-5% to -35%). Short ATM put creates gap exposure, put spread buffers.",
        "legs": [
            {"is_call": True,  "strike_pct": 1.00, "position": +1, "vol": 0.18, "label": "Long ATM Call"},
            {"is_call": True,  "strike_pct": 1.10, "position": -1, "vol": 0.16, "label": "Short OTM Call (Cap)"},
            {"is_call": False, "strike_pct": 1.00, "position": -1, "vol": 0.18, "label": "Short ATM Put (Gap Exposure)"},
            {"is_call": False, "strike_pct": 0.95, "position": +1, "vol": 0.19, "label": "Long 95% Put (Buffer Start)"},
            {"is_call": False, "strike_pct": 0.65, "position": -1, "vol": 0.28, "label": "Short 65% Put (Buffer End)"},
        ],
    },
    "Accelerated Buffer": {
        "description": "2x upside participation capped at ~12%, ~9% buffer.",
        "legs": [
            {"is_call": True,  "strike_pct": 1.00, "position": +2, "vol": 0.18, "label": "Long 2x ATM Call"},
            {"is_call": True,  "strike_pct": 1.06, "position": -2, "vol": 0.17, "label": "Short 2x 106% Call (Cap)"},
            {"is_call": False, "strike_pct": 0.91, "position": -1, "vol": 0.22, "label": "Short Put (Buffer Edge)"},
        ],
    },
    "Enhanced Buffer": {
        "description": "~15% buffer with moderate cap (~12%). Deeper protection than standard.",
        "legs": [
            {"is_call": True,  "strike_pct": 1.00, "position": +1, "vol": 0.18, "label": "Long ATM Call"},
            {"is_call": True,  "strike_pct": 1.12, "position": -1, "vol": 0.16, "label": "Short OTM Call (Cap)"},
            {"is_call": False, "strike_pct": 0.85, "position": -1, "vol": 0.24, "label": "Short Put (Buffer Edge)"},
        ],
    },
}


def get_template(
    name: str,
    underlying: float,
    start_date: date,
    expiry_date: date,
) -> list[OptionLeg]:
    """Return preset option legs for a named template."""
    tmpl = TEMPLATES[name]
    legs: list[OptionLeg] = []
    for leg in tmpl["legs"]:
        legs.append(OptionLeg(
            is_call=leg["is_call"],
            strike=round(underlying * leg["strike_pct"], 2),
            expiry_date=expiry_date,
            position=leg["position"],
            implied_vol=leg["vol"],
            label=leg["label"],
        ))
    return legs


# ---------------------------------------------------------------------------
# Core pricing helpers
# ---------------------------------------------------------------------------

def _time_in_years(from_date: date, to_date: date) -> float:
    """Calendar-day fraction of a year.  Returns a small positive floor if same day."""
    days = (to_date - from_date).days
    return max(days / 365.0, 1e-6)


def _price_leg(
    leg: OptionLeg,
    underlying: float,
    as_of: date,
    risk_free_rate: float,
    dividend_yield: float,
    vol_shift: float = 0.0,
    rate_shift: float = 0.0,
) -> tuple[float, float, float]:
    """Return (price, delta, theta_per_day) for one leg using BSM."""
    T = _time_in_years(as_of, leg.expiry_date)
    sigma = max(leg.implied_vol + vol_shift, 0.001)
    r = risk_free_rate + rate_shift

    inputs = BSMInputs(
        S=underlying,
        K=leg.strike,
        T=T,
        r=r,
        sigma=sigma,
        q=dividend_yield,
        is_call=leg.is_call,
    )
    result = bsm_price(inputs)
    return result.price, result.delta, result.theta


# ---------------------------------------------------------------------------
# Portfolio pricing
# ---------------------------------------------------------------------------

def _compute_scale_factor(portfolio: PortfolioDefinition) -> float:
    """
    Compute the multiplier that maps raw option values to a $100-notional ETF.

    A Buffer ETF is modelled as:
        NAV_t = 100 + scale * (options_value_t - options_value_0)

    where scale = 100 / underlying_start.  This ensures that option payoffs
    (denominated in underlying-price units) are expressed as percentages of
    a $100 investment — e.g. a 50-point call payoff on a $500 underlying
    contributes $10 to NAV (= 10% return).
    """
    return 100.0 / portfolio.underlying_start


def price_portfolio(
    portfolio: PortfolioDefinition,
    scenario: ScenarioParams,
) -> PortfolioResult:
    """Price the full portfolio under a given scenario.

    NAV = 100 + scale * sum_over_legs(position * (current_price - initial_price))

    This is the standard ETF model: the investor's $100 is held in T-bills
    (approximated here as staying at par), and the self-financing options
    overlay adds or subtracts value.
    """
    scale = _compute_scale_factor(portfolio)

    leg_results: list[LegResult] = []
    options_pnl = 0.0  # sum of position * (cur - init)

    for leg in portfolio.legs:
        # Initial price (at inception)
        init_price, _, _ = _price_leg(
            leg,
            portfolio.underlying_start,
            portfolio.start_date,
            portfolio.risk_free_rate,
            portfolio.dividend_yield,
        )
        # Current price (under scenario)
        cur_price, delta, theta = _price_leg(
            leg,
            scenario.underlying_price,
            scenario.analysis_date,
            portfolio.risk_free_rate,
            portfolio.dividend_yield,
            vol_shift=scenario.vol_shift,
            rate_shift=scenario.rate_shift,
        )

        init_pos_val = leg.position * init_price
        cur_pos_val = leg.position * cur_price
        price_chg = cur_price - init_price
        pct_chg = price_chg / init_price if init_price > 1e-10 else 0.0
        leg_pnl = leg.position * price_chg

        leg_results.append(LegResult(
            label=leg.label or _auto_label(leg),
            is_call=leg.is_call,
            strike=leg.strike,
            position=leg.position,
            initial_price=init_price,
            current_price=cur_price,
            price_change=price_chg,
            pct_change=pct_chg,
            position_value=cur_pos_val * scale,
            initial_position_value=init_pos_val * scale,
            delta=delta * leg.position,
            theta=theta * leg.position,
        ))

        options_pnl += leg_pnl

    nav = 100.0 + options_pnl * scale

    underlying_ret = (
        (scenario.underlying_price - portfolio.underlying_start)
        / portfolio.underlying_start
    )

    return PortfolioResult(
        nav=nav,
        nav_return=(nav - 100.0) / 100.0,
        underlying_return=underlying_ret,
        legs=leg_results,
        scale_factor=scale,
    )


def _auto_label(leg: OptionLeg) -> str:
    direction = "Long" if leg.position > 0 else "Short"
    kind = "Call" if leg.is_call else "Put"
    qty = f"{abs(leg.position)}x " if abs(leg.position) > 1 else ""
    return f"{direction} {qty}{leg.strike:.0f} {kind}"


# ---------------------------------------------------------------------------
# Chart data helpers
# ---------------------------------------------------------------------------

def compute_nav_vs_underlying(
    portfolio: PortfolioDefinition,
    scenario: ScenarioParams,
    underlying_range: np.ndarray,
) -> np.ndarray:
    """NAV for each underlying price, holding date/vol/rate from scenario."""
    navs = np.empty(len(underlying_range))
    for i, s in enumerate(underlying_range):
        sc = ScenarioParams(
            underlying_price=float(s),
            analysis_date=scenario.analysis_date,
            vol_shift=scenario.vol_shift,
            rate_shift=scenario.rate_shift,
        )
        navs[i] = price_portfolio(portfolio, sc).nav
    return navs


def compute_payoff_at_expiry(
    portfolio: PortfolioDefinition,
    underlying_range: np.ndarray,
) -> np.ndarray:
    """Intrinsic-value P&L at expiration for each underlying price.

    Returns the change from the $100 starting NAV (i.e. P&L, not absolute NAV)
    so it can be compared against the underlying's P&L on the same chart.
    """
    scale = _compute_scale_factor(portfolio)

    # Initial options cost at inception
    initial_cost = 0.0
    for leg in portfolio.legs:
        init_price, _, _ = _price_leg(
            leg,
            portfolio.underlying_start,
            portfolio.start_date,
            portfolio.risk_free_rate,
            portfolio.dividend_yield,
        )
        initial_cost += leg.position * init_price

    payoffs = np.zeros(len(underlying_range))
    for i, s in enumerate(underlying_range):
        expiry_value = 0.0
        for leg in portfolio.legs:
            if leg.is_call:
                intrinsic = max(float(s) - leg.strike, 0.0)
            else:
                intrinsic = max(leg.strike - float(s), 0.0)
            expiry_value += leg.position * intrinsic
        # P&L = (expiry_value - initial_cost) * scale
        payoffs[i] = (expiry_value - initial_cost) * scale
    return payoffs


def compute_nav_over_time(
    portfolio: PortfolioDefinition,
    scenario: ScenarioParams,
    date_range: list[date],
) -> list[float]:
    """NAV at each date for a fixed underlying price."""
    navs: list[float] = []
    for d in date_range:
        sc = ScenarioParams(
            underlying_price=scenario.underlying_price,
            analysis_date=d,
            vol_shift=scenario.vol_shift,
            rate_shift=scenario.rate_shift,
        )
        navs.append(price_portfolio(portfolio, sc).nav)
    return navs


def compute_leg_pnl_vs_underlying(
    portfolio: PortfolioDefinition,
    scenario: ScenarioParams,
    underlying_range: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-leg P&L contribution (NAV-scaled) across a range of underlying prices."""
    scale = _compute_scale_factor(portfolio)
    results: dict[str, np.ndarray] = {}

    for leg in portfolio.legs:
        label = leg.label or _auto_label(leg)
        pnls = np.empty(len(underlying_range))
        # initial price per unit
        init_price, _, _ = _price_leg(
            leg,
            portfolio.underlying_start,
            portfolio.start_date,
            portfolio.risk_free_rate,
            portfolio.dividend_yield,
        )

        for i, s in enumerate(underlying_range):
            cur_price, _, _ = _price_leg(
                leg,
                float(s),
                scenario.analysis_date,
                portfolio.risk_free_rate,
                portfolio.dividend_yield,
                vol_shift=scenario.vol_shift,
                rate_shift=scenario.rate_shift,
            )
            # P&L = position * (current - initial) * scale
            pnls[i] = leg.position * (cur_price - init_price) * scale
        results[label] = pnls

    return results
