"""
payoff.py — Buffer ETF Payoff Calculation Engine

This module models the payoff structure of buffer ETFs. These products use
options to create a "defined outcome" over a fixed period (typically 1 year):

  - CAPPED UPSIDE: Gains are capped at the cap level. If the underlying index
    returns more than the cap, the ETF returns only the cap amount.

  - BUFFERED DOWNSIDE: Losses up to the buffer level are absorbed (the investor
    doesn't lose money). Losses beyond the buffer are passed through 1:1.

  - For "Ultra" buffer ETFs, there's a gap — the buffer doesn't start at 0%.
    For example, a 5-35% ultra buffer means the investor loses the first 5%,
    is protected from -5% to -35%, and loses 1:1 beyond -35%.

Payoff Diagram (Standard Buffer ETF, e.g., 15% cap, 9% buffer):

  ETF Return
    ^
    |         ___________  <- Cap (15%)
    |        /
    |       /
    |      /   <- 1:1 participation between 0% and cap
    |     /
    +----+-----------------> Underlying Return
    |    |    |
    |    0%  -9%  <- Buffer absorbs losses
    |         \
    |          \   <- 1:1 loss beyond buffer
    |           \

This module calculates the expected ETF return for any given underlying
market return, accounting for:
  - Where we are in the outcome period (remaining cap/buffer vs. starting)
  - Whether we're modeling a "hold to reset" or "roll into new ETF" strategy

Key Concepts:
  - "Starting" values: The cap/buffer at the beginning of the outcome period.
  - "Remaining" values: The cap/buffer available from the current price/date.
    As the underlying index rises, remaining cap decreases and remaining buffer
    increases (and vice versa).
  - The payoff from "here to reset" uses remaining cap and remaining buffer.
  - A new ETF uses its fresh starting cap and starting buffer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PayoffParams:
    """
    Parameters defining a buffer ETF's payoff structure.

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
    Calculate the buffer ETF return for a given underlying market return.

    This is the core payoff function. It models the three-segment piecewise
    linear payoff structure of a buffer ETF:

      1. If market goes up: ETF return = min(market_return, cap)
      2. If market goes down within buffer: ETF return = max(market_return + buffer, 0)
         (but accounting for downside_before_buffer for ultra products)
      3. If market goes down beyond buffer: ETF return follows 1:1

    More precisely for standard/power buffers (downside_before_buffer == 0):
      - market_return >= 0:  ETF return = min(market_return, cap)
      - -buffer <= market_return < 0:  ETF return = 0  (fully buffered)
      - market_return < -buffer:  ETF return = market_return + buffer

    For ultra buffers (downside_before_buffer > 0, e.g., 5%):
      - market_return >= 0:  ETF return = min(market_return, cap)
      - 0 > market_return >= -downside_before_buffer:  ETF return = market_return (1:1 loss)
      - -downside_before_buffer > market_return >= -(downside_before_buffer + buffer):
            ETF return = -downside_before_buffer  (buffered zone)
      - market_return < -(downside_before_buffer + buffer):
            ETF return = market_return + buffer

    Args:
        market_return: The forward return of the underlying index as a decimal
            (e.g., 0.10 for +10%, -0.20 for -20%).
        params: PayoffParams defining the cap, buffer, and any gap.

    Returns:
        The ETF's return as a decimal (e.g., 0.12 for +12%).

    Examples:
        >>> params = PayoffParams(cap=0.15, buffer=0.09)
        >>> calculate_buffer_etf_return(0.10, params)   # +10% market
        0.10
        >>> calculate_buffer_etf_return(0.20, params)   # +20% market, capped
        0.15
        >>> calculate_buffer_etf_return(-0.05, params)  # -5% market, buffered
        0.0
        >>> calculate_buffer_etf_return(-0.15, params)  # -15% market, beyond buffer
        -0.06
    """
    cap = params.cap
    buffer = params.buffer
    gap = params.downside_before_buffer

    # --- UPSIDE: Gains are capped ---
    if market_return >= 0:
        return min(market_return, cap)

    # --- DOWNSIDE: Three zones depending on buffer type ---

    if gap > 0:
        # Ultra buffer: investor loses 1:1 in the gap, then buffer kicks in
        # Zone 1: Loss within the gap (0 to -gap)
        if market_return >= -gap:
            return market_return  # 1:1 loss in the gap

        # Zone 2: Loss within the buffer zone (-gap to -(gap + buffer))
        if market_return >= -(gap + buffer):
            return -gap  # Loss is frozen at the gap amount

        # Zone 3: Loss beyond the buffer
        # The investor loses 1:1 for the portion beyond (gap + buffer)
        return market_return + buffer

    else:
        # Standard / Power buffer: protection starts at 0%
        # Zone 1: Loss within the buffer (0 to -buffer)
        if market_return >= -buffer:
            return 0.0  # Fully protected

        # Zone 2: Loss beyond the buffer — 1:1 participation
        # If market is down 15% and buffer is 9%, ETF return = -15% + 9% = -6%
        return market_return + buffer


def calculate_hold_value(
    position_value: float,
    market_return: float,
    remaining_cap: float,
    remaining_buffer: float,
    downside_before_buffer: float = 0.0,
) -> float:
    """
    Calculate the ending position value if holding the current ETF to reset.

    Uses the remaining cap and remaining buffer (from the current price to
    the reset date) to model the payoff.

    Args:
        position_value: Current dollar value of the position.
        market_return: Expected forward return of the underlying index from
            now until the reset date.
        remaining_cap: Remaining upside cap from current price (decimal).
        remaining_buffer: Remaining downside buffer from current price (decimal).
        downside_before_buffer: Gap before buffer for ultra products (decimal).

    Returns:
        Ending dollar value of the position at the reset date.

    Example:
        >>> calculate_hold_value(100000, 0.05, 0.10, 0.08)
        105000.0  # 5% gain on $100k
        >>> calculate_hold_value(100000, 0.15, 0.10, 0.08)
        110000.0  # Capped at 10%
    """
    params = PayoffParams(
        cap=remaining_cap,
        buffer=remaining_buffer,
        downside_before_buffer=downside_before_buffer,
    )
    etf_return = calculate_buffer_etf_return(market_return, params)
    return position_value * (1.0 + etf_return)


def calculate_roll_value(
    position_value: float,
    market_return: float,
    new_cap: float,
    new_buffer: float,
    transaction_cost: float = 0.0,
    tax_drag: float = 0.0,
    new_downside_before_buffer: float = 0.0,
) -> float:
    """
    Calculate the ending position value if rolling into a new ETF.

    The roll strategy:
      1. Sell the current position (incurring transaction costs and possibly taxes).
      2. Buy the new ETF with fresh cap and buffer.
      3. The new ETF's payoff applies to the post-cost/tax position value.

    Args:
        position_value: Current dollar value of the position.
        market_return: Expected forward return of the underlying index over
            the new ETF's full outcome period.
        new_cap: The new ETF's starting cap (decimal).
        new_buffer: The new ETF's starting buffer (decimal).
        transaction_cost: Total cost to execute the roll as a decimal of
            position value (e.g., 0.001 for 10 basis points).
        tax_drag: Tax cost from realizing gains, as a dollar amount.
            (Calculated separately by tax.py based on gains and tax rate.)
        new_downside_before_buffer: Gap for ultra buffer products (decimal).

    Returns:
        Ending dollar value of the position after rolling.

    Example:
        >>> calculate_roll_value(100000, 0.10, 0.18, 0.15, 0.001, 500)
        # Sell: $100,000 - $100 (0.1% cost) - $500 (tax) = $99,400
        # New ETF return at +10%: min(10%, 18%) = 10%
        # Ending value: $99,400 * 1.10 = $109,340
    """
    # Step 1: Calculate proceeds after transaction costs and taxes
    proceeds = position_value * (1.0 - transaction_cost) - tax_drag

    # Step 2: Apply the new ETF's payoff structure
    params = PayoffParams(
        cap=new_cap,
        buffer=new_buffer,
        downside_before_buffer=new_downside_before_buffer,
    )
    etf_return = calculate_buffer_etf_return(market_return, params)

    return proceeds * (1.0 + etf_return)


def estimate_remaining_params(
    starting_cap: float,
    starting_buffer: float,
    current_return_since_start: float,
    downside_before_buffer: float = 0.0,
) -> tuple[float, float]:
    """
    Estimate remaining cap and remaining buffer from current position.

    When the underlying index has moved since the outcome period started,
    the remaining cap and buffer shift accordingly:

      - If the market is UP since start:
        * Remaining cap decreases (less upside room)
        * Remaining buffer increases (more downside protection)

      - If the market is DOWN since start:
        * Remaining cap increases (more upside room)
        * Remaining buffer decreases (less protection left)
        * If the market has fallen past the buffer, remaining buffer is 0

    This is a simplified linear estimate. The actual remaining values depend
    on the ETF's NAV and the options pricing, but this gives a reasonable
    approximation.

    Args:
        starting_cap: The cap at the beginning of the outcome period.
        starting_buffer: The buffer at the beginning of the outcome period.
        current_return_since_start: The underlying index return since the
            outcome period started (decimal).
        downside_before_buffer: Gap before buffer for ultra products.

    Returns:
        Tuple of (remaining_cap, remaining_buffer) as decimals.

    Examples:
        >>> estimate_remaining_params(0.15, 0.09, 0.05)
        (0.10, 0.14)  # Market up 5%: cap shrinks, buffer grows
        >>> estimate_remaining_params(0.15, 0.09, -0.03)
        (0.18, 0.06)  # Market down 3%: cap grows, buffer shrinks
    """
    r = current_return_since_start

    if r >= 0:
        # Market is up: cap has been partially consumed, buffer has grown
        remaining_cap = max(0.0, starting_cap - r)
        remaining_buffer = starting_buffer + r
    else:
        # Market is down: cap has grown, buffer has been partially consumed
        remaining_cap = starting_cap + abs(r)

        if downside_before_buffer > 0:
            # Ultra buffer: the gap absorbs first
            if abs(r) <= downside_before_buffer:
                # Still in the gap zone — buffer hasn't been touched
                remaining_buffer = starting_buffer
            elif abs(r) <= downside_before_buffer + starting_buffer:
                # In the buffer zone — some buffer consumed
                remaining_buffer = starting_buffer - (abs(r) - downside_before_buffer)
            else:
                # Beyond the buffer — no protection left
                remaining_buffer = 0.0
        else:
            # Standard/power buffer
            remaining_buffer = max(0.0, starting_buffer - abs(r))

    return remaining_cap, remaining_buffer
