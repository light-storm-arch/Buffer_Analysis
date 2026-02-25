"""
option_pricing.py — Option Pricing Engine

This module implements two option pricing models:

  1. Black-Scholes-Merton (BSM) with continuous dividend yield
     - Closed-form European option pricing
     - All five Greeks (Delta, Gamma, Theta, Vega, Rho)
     - Implied volatility solver

  2. Cox-Ross-Rubinstein Binomial Tree
     - European and American option pricing
     - Numerical Greeks via input bumping

Mathematical Reference (BSM):

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    Put  = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)

Where N(x) is the standard normal CDF, n(x) is the standard normal PDF.

Binomial Tree (CRR):

    dt = T / N,  u = exp(sigma * sqrt(dt)),  d = 1/u
    p = (exp((r - q) * dt) - d) / (u - d)
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BSMInputs:
    """
    Parameters for option pricing.

    All rates are annualized decimals (e.g., 0.05 for 5%).

    Attributes:
        S: Current price of the underlying asset.
        K: Strike price.
        T: Time to expiration in years (e.g., 0.5 for 6 months).
        r: Risk-free interest rate (annualized, continuous compounding).
        sigma: Volatility (annualized standard deviation of returns).
        q: Continuous dividend yield (annualized). Default 0.0.
        is_call: True for call, False for put.
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    is_call: bool = True

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Underlying price must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiration must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")


@dataclass
class BSMResult:
    """
    Complete output from option pricing.

    Attributes:
        price: Theoretical option price.
        delta: Rate of change of price w.r.t. underlying price.
        gamma: Rate of change of delta w.r.t. underlying price.
        theta: Rate of change of price w.r.t. time (per calendar day).
        vega: Rate of change of price w.r.t. 1% change in volatility.
        rho: Rate of change of price w.r.t. 1% change in interest rate.
        d1: BSM intermediate variable d1.
        d2: BSM intermediate variable d2.
    """
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    d1: float
    d2: float


# ---------------------------------------------------------------------------
# Black-Scholes-Merton pricing
# ---------------------------------------------------------------------------

def bsm_price(inputs: BSMInputs) -> BSMResult:
    """
    Price a European option using the generalized Black-Scholes-Merton model
    with continuous dividend yield, and compute all five Greeks.

    Returns a BSMResult with price, delta, gamma, theta (per day),
    vega (per 1% vol change), and rho (per 1% rate change).
    """
    S, K, T, r, sigma, q = (
        inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
    )

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    exp_neg_qT = math.exp(-q * T)
    exp_neg_rT = math.exp(-r * T)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    if inputs.is_call:
        price = S * exp_neg_qT * Nd1 - K * exp_neg_rT * Nd2
        delta = exp_neg_qT * Nd1
        theta_annual = (
            -(S * sigma * exp_neg_qT * nd1) / (2 * sqrt_T)
            - r * K * exp_neg_rT * Nd2
            + q * S * exp_neg_qT * Nd1
        )
        rho_annual = K * T * exp_neg_rT * Nd2
    else:
        Nm_d1 = norm.cdf(-d1)
        Nm_d2 = norm.cdf(-d2)
        price = K * exp_neg_rT * Nm_d2 - S * exp_neg_qT * Nm_d1
        delta = exp_neg_qT * (Nd1 - 1.0)
        theta_annual = (
            -(S * sigma * exp_neg_qT * nd1) / (2 * sqrt_T)
            + r * K * exp_neg_rT * Nm_d2
            - q * S * exp_neg_qT * Nm_d1
        )
        rho_annual = -K * T * exp_neg_rT * Nm_d2

    # Gamma and vega are the same for calls and puts
    gamma = exp_neg_qT * nd1 / (S * sigma * sqrt_T)
    vega_annual = S * exp_neg_qT * nd1 * sqrt_T

    return BSMResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta_annual / 365.0,   # Per calendar day
        vega=vega_annual / 100.0,      # Per 1% vol change
        rho=rho_annual / 100.0,        # Per 1% rate change
        d1=d1,
        d2=d2,
    )


# ---------------------------------------------------------------------------
# Binomial tree pricing (Cox-Ross-Rubinstein)
# ---------------------------------------------------------------------------

def binomial_price(
    inputs: BSMInputs,
    american: bool = False,
    steps: int = 200,
) -> float:
    """
    Price an option using the Cox-Ross-Rubinstein binomial tree model.

    Supports both European and American exercise. For European options,
    the result converges to BSM as steps increases.

    Args:
        inputs: Option parameters (same as BSM).
        american: If True, allow early exercise at each node.
        steps: Number of time steps in the tree. Default 200.

    Returns:
        The theoretical option price.
    """
    S, K, T, r, sigma, q = (
        inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
    )

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    discount = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p_disc = discount * p
    q_disc = discount * (1.0 - p)

    # Build terminal asset prices
    asset_prices = np.array([
        S * (u ** (steps - j)) * (d ** j) for j in range(steps + 1)
    ])

    # Terminal option values
    if inputs.is_call:
        option_values = np.maximum(asset_prices - K, 0.0)
    else:
        option_values = np.maximum(K - asset_prices, 0.0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        # Asset prices at this step
        asset_at_step = np.array([
            S * (u ** (i - j)) * (d ** j) for j in range(i + 1)
        ])

        # Continuation value
        option_values = p_disc * option_values[:-1] + q_disc * option_values[1:]

        # Early exercise check for American options
        if american:
            if inputs.is_call:
                exercise = np.maximum(asset_at_step - K, 0.0)
            else:
                exercise = np.maximum(K - asset_at_step, 0.0)
            option_values = np.maximum(option_values, exercise)

    return float(option_values[0])


def binomial_greeks(
    inputs: BSMInputs,
    american: bool = False,
    steps: int = 200,
) -> dict:
    """
    Compute Greeks numerically for the binomial model by bumping inputs.

    Returns a dict with keys: price, delta, gamma, theta, vega, rho.
    Theta is per calendar day, vega per 1% vol change, rho per 1% rate change.
    """
    price = binomial_price(inputs, american, steps)

    bump_S = max(inputs.S * 0.05, 0.01)  # 5% bump for stable numerical gamma with binomial
    bump_sigma = 0.01         # 1 percentage point
    bump_r = 0.01             # 1 percentage point
    bump_T = 1.0 / 365.0     # 1 day

    # Delta and gamma via central difference on S
    inputs_up = BSMInputs(
        S=inputs.S + bump_S, K=inputs.K, T=inputs.T,
        r=inputs.r, sigma=inputs.sigma, q=inputs.q, is_call=inputs.is_call,
    )
    inputs_down = BSMInputs(
        S=inputs.S - bump_S, K=inputs.K, T=inputs.T,
        r=inputs.r, sigma=inputs.sigma, q=inputs.q, is_call=inputs.is_call,
    )
    price_up = binomial_price(inputs_up, american, steps)
    price_down = binomial_price(inputs_down, american, steps)
    delta = (price_up - price_down) / (2 * bump_S)
    gamma = (price_up - 2 * price + price_down) / (bump_S ** 2)

    # Theta: price change per 1-day decrease in T
    if inputs.T > bump_T:
        inputs_t = BSMInputs(
            S=inputs.S, K=inputs.K, T=inputs.T - bump_T,
            r=inputs.r, sigma=inputs.sigma, q=inputs.q, is_call=inputs.is_call,
        )
        theta = (binomial_price(inputs_t, american, steps) - price)
    else:
        theta = 0.0

    # Vega: per 1% vol change
    inputs_v_up = BSMInputs(
        S=inputs.S, K=inputs.K, T=inputs.T,
        r=inputs.r, sigma=inputs.sigma + bump_sigma, q=inputs.q,
        is_call=inputs.is_call,
    )
    inputs_v_down = BSMInputs(
        S=inputs.S, K=inputs.K, T=inputs.T,
        r=inputs.r, sigma=max(0.001, inputs.sigma - bump_sigma), q=inputs.q,
        is_call=inputs.is_call,
    )
    vega = (
        binomial_price(inputs_v_up, american, steps)
        - binomial_price(inputs_v_down, american, steps)
    ) / (inputs_v_up.sigma - inputs_v_down.sigma) / 100.0

    # Rho: per 1% rate change
    inputs_r_up = BSMInputs(
        S=inputs.S, K=inputs.K, T=inputs.T,
        r=inputs.r + bump_r, sigma=inputs.sigma, q=inputs.q,
        is_call=inputs.is_call,
    )
    inputs_r_down = BSMInputs(
        S=inputs.S, K=inputs.K, T=inputs.T,
        r=inputs.r - bump_r, sigma=inputs.sigma, q=inputs.q,
        is_call=inputs.is_call,
    )
    rho = (
        binomial_price(inputs_r_up, american, steps)
        - binomial_price(inputs_r_down, american, steps)
    ) / (2 * bump_r) / 100.0

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------

def implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    is_call: bool,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> Optional[float]:
    """
    Solve for implied volatility using Newton-Raphson with Brent fallback.

    Args:
        S, K, T, r, q: Standard BSM parameters (rates as decimals).
        market_price: Observed market price of the option.
        is_call: True for call, False for put.

    Returns:
        Implied volatility as a decimal (e.g., 0.20 for 20%), or None if
        no valid solution exists (e.g., price below intrinsic value).
    """
    # Check that market_price is positive
    if market_price <= 0:
        return None

    # Check intrinsic value bound
    exp_neg_qT = math.exp(-q * T)
    exp_neg_rT = math.exp(-r * T)
    if is_call:
        intrinsic = max(0.0, S * exp_neg_qT - K * exp_neg_rT)
    else:
        intrinsic = max(0.0, K * exp_neg_rT - S * exp_neg_qT)

    if market_price < intrinsic - tolerance:
        return None

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = market_price * math.sqrt(2 * math.pi / T) / S
    sigma = max(sigma, 0.01)

    # Newton-Raphson
    for _ in range(max_iterations):
        try:
            inputs = BSMInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q, is_call=is_call)
            result = bsm_price(inputs)
        except ValueError:
            break

        diff = result.price - market_price
        if abs(diff) < tolerance:
            return sigma

        # Convert vega back to per-1.0 vol change for Newton step
        vega_raw = result.vega * 100.0
        if abs(vega_raw) < 1e-12:
            break

        sigma -= diff / vega_raw
        if sigma <= 0:
            sigma = 0.001

    # Fallback: Brent's method
    def objective(vol):
        inp = BSMInputs(S=S, K=K, T=T, r=r, sigma=vol, q=q, is_call=is_call)
        return bsm_price(inp).price - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tolerance)
    except ValueError:
        return None
