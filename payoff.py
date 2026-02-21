"""
payoff.py — Buffer ETF Payoff Calculation Engine

This module models the payoff structure of buffer ETFs at expiry. These
products use FLEX options to create a "defined outcome" over a fixed period
(typically 1 year):

  - CAPPED UPSIDE: Gains are capped at the cap level. If the reference asset
    returns more than the cap, the ETF returns only the cap amount.

  - BUFFERED DOWNSIDE: Losses up to the buffer level are absorbed (the investor
    doesn't lose money). Losses beyond the buffer are passed through 1:1.

  - For "Ultra" buffer ETFs, there's a gap — the buffer doesn't start at 0%.
    For example, a 5-35% ultra buffer means the investor loses the first 5%,
    is protected from -5% to -35%, and loses 1:1 beyond -35%.

Payoff Diagram (Standard Buffer ETF, e.g., 15% cap, 9% buffer):

  ETF Return (at expiry, from period start)
    ^
    |         ___________  <- Cap (15%)
    |        /
    |       /
    |      /   <- 1:1 participation between 0% and cap
    |     /
    +----+-----------------> Reference Asset Return (from period start)
    |    |    |
    |    0%  -9%  <- Buffer absorbs losses
    |         \
    |          \   <- 1:1 loss beyond buffer
    |           \

IMPORTANT — Fund Return vs. Reference Asset Return:

  Mid-period, the fund NAV does NOT track the reference asset 1:1. If the
  S&P 500 is up 15% from period start, the fund might only be up 10% because
  the options still embed time value. The fund converges toward the at-expiry
  payoff as the outcome period end approaches.

  This means:
    - "Remaining cap" (in fund-NAV terms) can differ from the reference asset's
      distance to cap.
    - An investor who is up 10% in fund NAV has 10% of unrealized gain at risk
      BEFORE the buffer provides any protection.

  The correct way to model "hold to expiry" is:
    1. Compound the forward reference asset return with the current reference
       return to get the total return from period start.
    2. Apply the at-expiry payoff function using the STARTING cap and buffer.
    3. Convert the resulting fund NAV at expiry back to an investor return
       relative to the current fund NAV.
"""

from dataclasses import dataclass


@dataclass
class PayoffParams:
    """
    Parameters defining a buffer ETF's at-expiry payoff structure.

    All percentage values are stored as decimals (e.g., 0.15 for 15%).

    Attributes:
        cap: Maximum upside return (e.g., 0.15 for 15% cap).
        buffer: Downside protection level (e.g., 0.09 for 9% buffer).
        downside_before_buffer: For ultra buffers, the loss the investor
            absorbs before the buffer kicks in (e.g., 0.05 for 5%).
            Set to 0.0 for standard and power buffer ETFs.
    """
    cap: float
    buffer: float
    downside_before_buffer: float = 0.0

    def __post_init__(self):
        """Validate that payoff parameters are sensible."""
        if self.cap < 0:
            raise ValueError(f"Cap must be non-negative, got {self.cap}")
        if self.buffer < 0:
            raise ValueError(f"Buffer must be non-negative, got {self.buffer}")
        if self.downside_before_buffer < 0:
            raise ValueError(
                f"Downside before buffer must be non-negative, got {self.downside_before_buffer}"
            )


def calculate_buffer_etf_return(
    market_return: float,
    params: PayoffParams,
) -> float:
    """
    Calculate the buffer ETF return at expiry for a given reference asset return.

    This models the at-expiry piecewise linear payoff. Both the input
    (market_return) and output are returns measured from the SAME starting
    point (the outcome period start).

    For standard/power buffers (downside_before_buffer == 0):
      - market_return >= 0:  ETF return = min(market_return, cap)
      - -buffer <= market_return < 0:  ETF return = 0  (fully buffered)
      - market_return < -buffer:  ETF return = market_return + buffer

    For ultra buffers (downside_before_buffer > 0, e.g., 5%):
      - market_return >= 0:  ETF return = min(market_return, cap)
      - 0 > market_return >= -gap:  ETF return = market_return (1:1 loss in gap)
      - -gap > market_return >= -(gap + buffer):  ETF return = -gap (buffered)
      - market_return < -(gap + buffer):  ETF return = market_return + buffer

    Args:
        market_return: Reference asset return from period start to expiry (decimal).
        params: PayoffParams defining the cap, buffer, and any gap.

    Returns:
        The ETF's return from period start as a decimal.

    Examples:
        >>> params = PayoffParams(cap=0.15, buffer=0.09)
        >>> calculate_buffer_etf_return(0.10, params)
        0.10
        >>> calculate_buffer_etf_return(0.20, params)   # capped
        0.15
        >>> calculate_buffer_etf_return(-0.05, params)  # buffered
        0.0
        >>> calculate_buffer_etf_return(-0.15, params)  # beyond buffer
        -0.06
    """
    cap = params.cap
    buffer = params.buffer
    gap = params.downside_before_buffer

    # --- UPSIDE: Gains are capped ---
    if market_return >= 0:
        return min(market_return, cap)

    # --- DOWNSIDE: Zones depend on buffer type ---

    if gap > 0:
        # Ultra buffer: investor loses 1:1 in the gap, then buffer kicks in
        if market_return >= -gap:
            return market_return  # 1:1 loss in the gap
        if market_return >= -(gap + buffer):
            return -gap  # Loss frozen at the gap amount
        return market_return + buffer  # 1:1 beyond the buffer

    else:
        # Standard / Power buffer: protection starts at 0%
        if market_return >= -buffer:
            return 0.0  # Fully protected
        return market_return + buffer  # 1:1 beyond the buffer


def calculate_hold_investor_return(
    forward_ref_return: float,
    current_ref_return: float,
    fund_return: float,
    starting_cap: float,
    starting_buffer: float,
    downside_before_buffer: float = 0.0,
) -> float:
    """
    Calculate the investor's return from current fund NAV if holding to expiry.

    This is the core "hold" calculation. It correctly models the relationship
    between forward reference asset moves and the investor's actual outcome:

      1. Compound the forward reference return with the current reference
         return to get the total reference return from period start to expiry.
      2. Apply the at-expiry payoff using the STARTING cap and buffer.
      3. Express the result as a return from the CURRENT fund NAV.

    Why this matters: If the fund is up 10% and the reference is up 15%, a
    -10% forward reference move results in a total reference return of
    (1.15)(0.90) - 1 = +3.5%. The at-expiry payoff for +3.5% is +3.5%
    (below cap). The fund NAV at expiry is 1.0 * 1.035 = 1.035. From the
    current fund NAV of 1.10, the investor's return is 1.035/1.10 - 1 = -5.9%.

    Args:
        forward_ref_return: Expected reference asset return from today to
            expiry (decimal, e.g., -0.10 for -10%).
        current_ref_return: Reference asset return from period start to today
            (decimal, e.g., 0.15 for +15%).
        fund_return: Fund NAV return from period start to today
            (decimal, e.g., 0.10 for +10%).
        starting_cap: Cap set at the beginning of the outcome period.
        starting_buffer: Buffer set at the beginning of the outcome period.
        downside_before_buffer: Gap before buffer for ultra products.

    Returns:
        Investor's return from current fund NAV (decimal).

    Examples:
        >>> # Ref up 15%, fund up 10%, 18% cap, 9% buffer, forward ref -10%
        >>> calculate_hold_investor_return(-0.10, 0.15, 0.10, 0.18, 0.09)
        -0.059...  # Fund NAV goes from 1.10 to 1.035
    """
    # Step 1: Total reference return from period start to expiry
    total_ref_return = (1.0 + current_ref_return) * (1.0 + forward_ref_return) - 1.0

    # Step 2: At-expiry payoff based on starting cap/buffer
    params = PayoffParams(
        cap=starting_cap,
        buffer=starting_buffer,
        downside_before_buffer=downside_before_buffer,
    )
    fund_return_at_expiry = calculate_buffer_etf_return(total_ref_return, params)

    # Step 3: Convert to investor return from current fund NAV
    # Starting NAV is normalized to 1.0
    # Fund NAV at expiry = 1.0 * (1 + fund_return_at_expiry)
    # Current fund NAV = 1.0 * (1 + fund_return)
    # Investor return = (expiry_nav / current_nav) - 1
    current_nav = 1.0 + fund_return
    expiry_nav = 1.0 + fund_return_at_expiry

    return (expiry_nav / current_nav) - 1.0


def calculate_roll_investor_return(
    forward_ref_return: float,
    new_cap: float,
    new_buffer: float,
    new_downside_before_buffer: float = 0.0,
) -> float:
    """
    Calculate the new ETF's at-expiry return for a given forward reference move.

    The roll strategy starts a fresh outcome period, so the forward reference
    return IS the total reference return from the new period start. The
    at-expiry payoff applies directly.

    Note: This returns the ETF's return only. Transaction costs and tax drag
    are applied separately at the position-value level in the scenario engine.

    Args:
        forward_ref_return: Expected reference asset return from today (decimal).
        new_cap: The new ETF's starting cap (decimal).
        new_buffer: The new ETF's starting buffer (decimal).
        new_downside_before_buffer: Gap before buffer for ultra products.

    Returns:
        The new ETF's return at expiry (decimal).
    """
    params = PayoffParams(
        cap=new_cap,
        buffer=new_buffer,
        downside_before_buffer=new_downside_before_buffer,
    )
    return calculate_buffer_etf_return(forward_ref_return, params)


def calculate_downside_metrics(
    fund_return: float,
    current_ref_return: float,
    starting_cap: float,
    starting_buffer: float,
    downside_before_buffer: float = 0.0,
) -> dict:
    """
    Calculate key downside risk metrics from the investor's current position.

    These metrics help advisors communicate the REAL risk to clients. The
    critical insight: if the fund is up 10%, the client's first ~9.1% of
    downside from current NAV is just giving back gains — the buffer doesn't
    protect against that.

    Metrics calculated:
      - downside_to_period_start: How much the investor loses from current NAV
            if the reference asset returns to 0% (flat from period start).
            This is the "unprotected" portion of their current gain.
      - downside_to_buffer_exhaustion: How much the investor loses from current
            NAV when the reference asset hits -buffer% from period start. The
            buffer absorbs losses in this zone, but the investor still loses
            their accumulated gain.
      - ref_forward_to_zero: How much the reference asset must fall from its
            current level to return to its period-start level.
      - ref_forward_to_buffer_edge: How much the reference asset must fall from
            its current level to exhaust the buffer.
      - remaining_cap_from_nav: Maximum additional return from current fund NAV.

    Args:
        fund_return: Fund NAV return from period start (decimal).
        current_ref_return: Reference asset return from period start (decimal).
        starting_cap: Cap at period start (decimal).
        starting_buffer: Buffer at period start (decimal).
        downside_before_buffer: Gap for ultra buffers (decimal).

    Returns:
        Dictionary of downside risk metrics.

    Example:
        >>> metrics = calculate_downside_metrics(0.10, 0.15, 0.18, 0.09)
        >>> metrics["downside_to_period_start"]
        -0.0909...  # Lose ~9.1% from current NAV to get back to start
    """
    current_nav = 1.0 + fund_return
    starting_nav = 1.0

    params = PayoffParams(
        cap=starting_cap,
        buffer=starting_buffer,
        downside_before_buffer=downside_before_buffer,
    )

    # --- Fund value when reference returns to 0% from period start ---
    # At ref = 0%, fund return from start = payoff(0%) = 0% → fund NAV = 1.0
    fund_at_ref_zero = starting_nav  # Always 1.0 for standard buffers
    investor_return_at_ref_zero = (fund_at_ref_zero / current_nav) - 1.0

    # --- Fund value at buffer exhaustion ---
    # Standard: ref = -buffer%, fund = 0% from start
    # Ultra: ref = -(gap + buffer), fund = -gap from start
    if downside_before_buffer > 0:
        buffer_edge_ref = -(downside_before_buffer + starting_buffer)
        fund_return_at_edge = calculate_buffer_etf_return(buffer_edge_ref, params)
    else:
        buffer_edge_ref = -starting_buffer
        fund_return_at_edge = 0.0  # By definition, at buffer edge fund = 0%
    fund_at_buffer_edge = starting_nav * (1.0 + fund_return_at_edge)
    investor_return_at_buffer_edge = (fund_at_buffer_edge / current_nav) - 1.0

    # --- Fund value at cap ---
    fund_at_cap = starting_nav * (1.0 + starting_cap)
    remaining_cap_from_nav = (fund_at_cap / current_nav) - 1.0

    # --- Forward reference returns to reach key levels ---
    # Forward return needed for ref to go from current to 0%
    ref_forward_to_zero = -current_ref_return / (1.0 + current_ref_return)

    # Forward return needed for ref to reach buffer edge from period start
    ref_at_buffer_edge = buffer_edge_ref
    ref_forward_to_buffer_edge = (
        (1.0 + ref_at_buffer_edge) / (1.0 + current_ref_return) - 1.0
    )

    # Forward return needed for ref to reach cap
    ref_forward_to_cap = (1.0 + starting_cap) / (1.0 + current_ref_return) - 1.0

    return {
        # Investor returns from current NAV at key reference levels
        "downside_to_period_start": investor_return_at_ref_zero,
        "downside_to_buffer_exhaustion": investor_return_at_buffer_edge,
        "remaining_cap_from_nav": remaining_cap_from_nav,

        # Forward reference moves needed to reach key levels
        "ref_forward_to_zero": ref_forward_to_zero,
        "ref_forward_to_buffer_edge": ref_forward_to_buffer_edge,
        "ref_forward_to_cap": ref_forward_to_cap,
    }
