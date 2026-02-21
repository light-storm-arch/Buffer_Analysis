"""
scenarios.py — Scenario Runner and Breakeven Calculator

This module generates a grid of forward market return scenarios and computes
the ending position value under both "hold" and "roll" strategies for each.
It then finds the breakeven point — the market return at which rolling into
a new ETF becomes more advantageous than holding the current one.

How it works:
  1. Define a range of forward market returns (e.g., -30% to +30%).
  2. For each return scenario:
     a. Calculate ending value under HOLD (using remaining cap/buffer).
     b. Calculate ending value under ROLL (sell, pay tax/costs, buy new ETF).
  3. Find where the two curves cross — that's the breakeven.
  4. Generate a recommendation based on the breakeven and current market outlook.

The breakeven is important because:
  - If you expect the market to return MORE than the breakeven, rolling is better
    (assuming the new ETF has a higher cap).
  - If you expect the market to return LESS than the breakeven, holding is better
    (because you avoid transaction costs and tax drag).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from payoff import PayoffParams, calculate_buffer_etf_return
from tax import TaxParams, calculate_tax_drag


@dataclass
class ScenarioInputs:
    """
    All inputs needed to run the hold-vs-roll scenario analysis.

    This groups together the current position details, the replacement ETF
    parameters, and the account/tax information.

    Attributes:
        # Current ETF position
        position_value: Current dollar value of the position.
        remaining_cap: Remaining upside cap from current price (decimal).
        remaining_buffer: Remaining downside protection from current price (decimal).
        current_downside_before_buffer: Gap before buffer for current ultra buffer (decimal).

        # Replacement ETF
        new_cap: The new ETF's starting cap (decimal).
        new_buffer: The new ETF's starting buffer (decimal).
        new_downside_before_buffer: Gap before buffer for new ultra buffer (decimal).

        # Costs and taxes
        transaction_cost_rate: Cost to execute the roll (decimal, e.g., 0.001 = 10bps).
        tax_params: Tax parameters for the account.

        # Scenario range
        min_return: Minimum forward market return to model (decimal, e.g., -0.30).
        max_return: Maximum forward market return to model (decimal, e.g., 0.30).
        return_step: Step size for the return grid (decimal, e.g., 0.01 = 1%).
    """
    # Current position
    position_value: float = 100_000.0
    remaining_cap: float = 0.10
    remaining_buffer: float = 0.09
    current_downside_before_buffer: float = 0.0

    # New ETF
    new_cap: float = 0.15
    new_buffer: float = 0.09
    new_downside_before_buffer: float = 0.0

    # Costs
    transaction_cost_rate: float = 0.001
    tax_params: TaxParams = None

    # Scenario range
    min_return: float = -0.30
    max_return: float = 0.30
    return_step: float = 0.01

    def __post_init__(self):
        """Set default tax params if none provided."""
        if self.tax_params is None:
            self.tax_params = TaxParams(is_taxable=False)


@dataclass
class ScenarioResult:
    """
    Result of a single scenario (one forward return assumption).

    Attributes:
        market_return: The assumed forward market return (decimal).
        hold_value: Ending position value under the hold strategy.
        roll_value: Ending position value under the roll strategy.
        hold_return: Return percentage under hold.
        roll_return: Return percentage under roll.
        advantage: Dollar advantage of rolling vs. holding (positive = roll wins).
        winner: "hold" or "roll" — which strategy produces a higher ending value.
    """
    market_return: float
    hold_value: float
    roll_value: float
    hold_return: float
    roll_return: float
    advantage: float
    winner: str


@dataclass
class BreakevenResult:
    """
    Summary of the breakeven analysis.

    Attributes:
        breakeven_return: The market return at which hold and roll produce
            equal outcomes. None if the curves don't cross.
        roll_wins_above: True if rolling is better when the market returns
            MORE than the breakeven. False if rolling is better below.
        recommendation: A human-readable recommendation string.
        scenario_df: DataFrame with full scenario grid results.
        scenarios: List of individual ScenarioResult objects.
    """
    breakeven_return: Optional[float]
    roll_wins_above: Optional[bool]
    recommendation: str
    scenario_df: pd.DataFrame
    scenarios: list


def run_scenarios(inputs: ScenarioInputs) -> BreakevenResult:
    """
    Run the full hold-vs-roll scenario analysis across a grid of market returns.

    This is the main entry point for the scenario engine. It:
      1. Creates the return grid from min_return to max_return.
      2. Computes hold and roll values for each scenario.
      3. Finds the breakeven point(s).
      4. Generates a recommendation.

    Args:
        inputs: ScenarioInputs containing all position and replacement ETF details.

    Returns:
        BreakevenResult with the full analysis, including the scenario DataFrame,
        breakeven return, and recommendation.
    """
    # Generate the grid of forward market returns
    # np.arange can have floating-point precision issues, so we round
    returns = np.arange(
        inputs.min_return,
        inputs.max_return + inputs.return_step / 2,  # include endpoint
        inputs.return_step,
    )
    returns = np.round(returns, 4)

    # Pre-calculate the tax drag (it's the same regardless of forward return,
    # because the tax is on the current gain, not the future gain)
    tax_drag = calculate_tax_drag(inputs.position_value, inputs.tax_params)

    # Build payoff parameter objects for hold and roll
    hold_params = PayoffParams(
        cap=inputs.remaining_cap,
        buffer=inputs.remaining_buffer,
        downside_before_buffer=inputs.current_downside_before_buffer,
    )
    roll_params = PayoffParams(
        cap=inputs.new_cap,
        buffer=inputs.new_buffer,
        downside_before_buffer=inputs.new_downside_before_buffer,
    )

    # Calculate proceeds available for the new ETF after costs and taxes
    roll_proceeds = (
        inputs.position_value * (1.0 - inputs.transaction_cost_rate) - tax_drag
    )

    # Run each scenario
    scenarios = []
    for mkt_ret in returns:
        # Hold strategy: apply remaining cap/buffer to current position
        hold_etf_return = calculate_buffer_etf_return(float(mkt_ret), hold_params)
        hold_value = inputs.position_value * (1.0 + hold_etf_return)

        # Roll strategy: apply new cap/buffer to post-cost/tax proceeds
        roll_etf_return = calculate_buffer_etf_return(float(mkt_ret), roll_params)
        roll_value = roll_proceeds * (1.0 + roll_etf_return)

        # Calculate returns relative to current position value
        hold_return = (hold_value / inputs.position_value) - 1.0
        roll_return = (roll_value / inputs.position_value) - 1.0

        advantage = roll_value - hold_value
        winner = "roll" if advantage > 0 else "hold"

        scenarios.append(ScenarioResult(
            market_return=float(mkt_ret),
            hold_value=hold_value,
            roll_value=roll_value,
            hold_return=hold_return,
            roll_return=roll_return,
            advantage=advantage,
            winner=winner,
        ))

    # Build the DataFrame for display and charting
    scenario_df = pd.DataFrame([
        {
            "Market Return": s.market_return,
            "Hold Value": s.hold_value,
            "Roll Value": s.roll_value,
            "Hold Return": s.hold_return,
            "Roll Return": s.roll_return,
            "Roll Advantage ($)": s.advantage,
            "Winner": s.winner,
        }
        for s in scenarios
    ])

    # Find the breakeven point(s) — where the advantage crosses zero
    breakeven_return, roll_wins_above = _find_breakeven(scenarios)

    # Generate recommendation
    recommendation = _generate_recommendation(
        breakeven_return=breakeven_return,
        roll_wins_above=roll_wins_above,
        inputs=inputs,
        tax_drag=tax_drag,
    )

    return BreakevenResult(
        breakeven_return=breakeven_return,
        roll_wins_above=roll_wins_above,
        recommendation=recommendation,
        scenario_df=scenario_df,
        scenarios=scenarios,
    )


def _find_breakeven(scenarios: list[ScenarioResult]) -> tuple[Optional[float], Optional[bool]]:
    """
    Find the breakeven market return where hold and roll produce equal outcomes.

    We look for where the "advantage" (roll_value - hold_value) changes sign.
    When it crosses zero, we interpolate to find the exact breakeven return.

    If the curves never cross (one strategy always dominates), we return None.

    Args:
        scenarios: List of ScenarioResult objects, sorted by market_return.

    Returns:
        Tuple of:
          - breakeven_return: The interpolated market return at breakeven, or None.
          - roll_wins_above: True if roll is better above the breakeven return.
    """
    if len(scenarios) < 2:
        return None, None

    # Look for sign changes in the advantage
    for i in range(1, len(scenarios)):
        prev = scenarios[i - 1]
        curr = scenarios[i]

        # Check if advantage changed sign
        if prev.advantage * curr.advantage < 0:
            # Linear interpolation to find the exact crossing point
            # advantage = a + (b - a) * t, solve for t where advantage = 0
            a = prev.advantage
            b = curr.advantage
            t = -a / (b - a) if (b - a) != 0 else 0.5

            breakeven = prev.market_return + t * (curr.market_return - prev.market_return)
            breakeven = round(breakeven, 4)

            # Determine which side roll wins on
            # If advantage goes from negative to positive, roll wins above breakeven
            roll_wins_above = b > a

            return breakeven, roll_wins_above

    # No crossover found — one strategy dominates everywhere
    # Check which one
    if all(s.advantage > 0 for s in scenarios):
        # Roll always wins — breakeven is below our range
        return None, None
    elif all(s.advantage < 0 for s in scenarios):
        # Hold always wins — breakeven is above our range
        return None, None

    return None, None


def _generate_recommendation(
    breakeven_return: Optional[float],
    roll_wins_above: Optional[bool],
    inputs: ScenarioInputs,
    tax_drag: float,
) -> str:
    """
    Generate a human-readable recommendation based on the breakeven analysis.

    The recommendation explains:
      1. What the breakeven return is.
      2. What that means for the hold vs. roll decision.
      3. Key factors influencing the decision (tax drag, cap differential, etc.).

    Args:
        breakeven_return: The breakeven market return, or None.
        roll_wins_above: Whether roll wins above the breakeven.
        inputs: The scenario inputs for context.
        tax_drag: Dollar amount of tax drag from rolling.

    Returns:
        A multi-line recommendation string.
    """
    cap_differential = inputs.new_cap - inputs.remaining_cap
    buffer_differential = inputs.new_buffer - inputs.remaining_buffer

    lines = []

    if breakeven_return is None:
        # Check if one strategy always dominates
        # Run a quick check at 0% return
        hold_params = PayoffParams(
            cap=inputs.remaining_cap, buffer=inputs.remaining_buffer,
            downside_before_buffer=inputs.current_downside_before_buffer
        )
        roll_params = PayoffParams(
            cap=inputs.new_cap, buffer=inputs.new_buffer,
            downside_before_buffer=inputs.new_downside_before_buffer
        )
        hold_val = inputs.position_value * (1.0 + calculate_buffer_etf_return(0.0, hold_params))
        roll_proceeds = inputs.position_value * (1.0 - inputs.transaction_cost_rate) - tax_drag
        roll_val = roll_proceeds * (1.0 + calculate_buffer_etf_return(0.0, roll_params))

        if roll_val > hold_val:
            lines.append(
                "ROLL is advantageous across all modeled scenarios. "
                "The new ETF's superior cap/buffer profile more than offsets "
                "any transaction costs and tax drag."
            )
        else:
            lines.append(
                "HOLD is advantageous across all modeled scenarios. "
                "The costs of rolling (transaction costs"
                + (" and tax drag" if tax_drag > 0 else "")
                + ") outweigh the benefits of the new ETF's cap/buffer profile."
            )
    else:
        breakeven_pct = breakeven_return * 100

        if roll_wins_above:
            lines.append(
                f"Breakeven forward market return: {breakeven_pct:+.1f}%"
            )
            lines.append(
                f"If you expect the market to return MORE than {breakeven_pct:+.1f}% "
                f"over the outcome period, ROLLING is advantageous."
            )
            lines.append(
                f"If you expect the market to return LESS than {breakeven_pct:+.1f}%, "
                f"HOLDING the current ETF is better."
            )
        else:
            lines.append(
                f"Breakeven forward market return: {breakeven_pct:+.1f}%"
            )
            lines.append(
                f"If you expect the market to return LESS than {breakeven_pct:+.1f}% "
                f"over the outcome period, ROLLING is advantageous."
            )
            lines.append(
                f"If you expect the market to return MORE than {breakeven_pct:+.1f}%, "
                f"HOLDING the current ETF is better."
            )

    # Add context about the key drivers
    lines.append("")
    lines.append("Key factors:")

    if cap_differential > 0:
        lines.append(
            f"  • New cap is {cap_differential:.1%} higher than remaining cap "
            f"({inputs.new_cap:.1%} vs {inputs.remaining_cap:.1%}), "
            f"favoring ROLL in up markets."
        )
    elif cap_differential < 0:
        lines.append(
            f"  • Remaining cap is {abs(cap_differential):.1%} higher than new cap "
            f"({inputs.remaining_cap:.1%} vs {inputs.new_cap:.1%}), "
            f"favoring HOLD in up markets."
        )
    else:
        lines.append(f"  • Cap levels are equal ({inputs.remaining_cap:.1%}).")

    if buffer_differential > 0:
        lines.append(
            f"  • New buffer is {buffer_differential:.1%} higher than remaining buffer "
            f"({inputs.new_buffer:.1%} vs {inputs.remaining_buffer:.1%}), "
            f"favoring ROLL in down markets."
        )
    elif buffer_differential < 0:
        lines.append(
            f"  • Remaining buffer is {abs(buffer_differential):.1%} higher than new buffer "
            f"({inputs.remaining_buffer:.1%} vs {inputs.new_buffer:.1%}), "
            f"favoring HOLD in down markets."
        )
    else:
        lines.append(f"  • Buffer levels are equal ({inputs.remaining_buffer:.1%}).")

    total_cost = inputs.position_value * inputs.transaction_cost_rate + max(0, tax_drag)
    if total_cost > 0:
        cost_components = []
        txn_cost = inputs.position_value * inputs.transaction_cost_rate
        if txn_cost > 0:
            cost_components.append(f"${txn_cost:,.0f} transaction cost")
        if tax_drag > 0:
            cost_components.append(f"${tax_drag:,.0f} tax drag")
        lines.append(
            f"  • Rolling costs: {' + '.join(cost_components)} = "
            f"${total_cost:,.0f} total, favoring HOLD."
        )
    elif tax_drag < 0:
        lines.append(
            f"  • Tax loss harvesting benefit of ${abs(tax_drag):,.0f} partially "
            f"offsets transaction costs, favoring ROLL."
        )

    return "\n".join(lines)
