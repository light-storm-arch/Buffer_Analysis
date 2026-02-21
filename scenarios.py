"""
scenarios.py — Scenario Runner and Breakeven Calculator

This module generates a grid of forward reference asset return scenarios and
computes the investor's return from current NAV under both "hold" and "roll"
strategies. It then finds ALL breakeven points and generates a recommendation.

How it works:
  1. Define a range of forward reference asset returns (e.g., -30% to +30%).
  2. For each return scenario:
     a. HOLD: Compound the forward return with the current reference return,
        apply the at-expiry payoff using starting cap/buffer, then convert
        to an investor return from the current fund NAV.
     b. ROLL: Sell current position (pay costs/taxes), buy new ETF, apply
        the new ETF's payoff to the forward return.
  3. Find ALL breakeven points where the two curves cross.
  4. Generate a recommendation.

Key Design Decision — Investor Return from Current NAV:

  All returns in the scenario table and chart are expressed as the investor's
  return from their CURRENT fund NAV. This is what matters to the client:
  "From where I am right now, how do these two strategies compare?"

  For the hold strategy, this correctly accounts for the fact that mid-period
  gains are at risk — the fund NAV can decline back toward (or below) its
  period-start value even though the buffer hasn't been touched yet.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from payoff import (
    calculate_hold_investor_return,
    calculate_roll_investor_return,
)
from tax import TaxParams, calculate_tax_drag


@dataclass
class ScenarioInputs:
    """
    All inputs needed to run the hold-vs-roll scenario analysis.

    This groups together the current position details, the replacement ETF
    parameters, and the account/tax information.

    Key fields for the CURRENT ETF:
      - fund_return: Fund NAV change from period start (e.g., 0.10 for +10%).
      - ref_return: Reference asset change from period start (e.g., 0.15 for +15%).
      - starting_cap: Cap at period start (derived from ref_return + ref_return_to_cap).
      - starting_buffer: Buffer at period start (derived from remaining_buffer
            and current ref position).
      - remaining_cap: Max additional return from current fund NAV (from Innovator).
      - remaining_buffer: Remaining protection (from Innovator).
      - downside_before_buffer: Gap for ultra buffers.

    Key fields for the REPLACEMENT ETF:
      - new_cap: Starting cap for the new ETF.
      - new_buffer: Starting buffer for the new ETF.
      - new_downside_before_buffer: Gap for ultra buffers.
    """
    # Current ETF position
    position_value: float = 100_000.0
    fund_return: float = 0.10
    ref_return: float = 0.15
    starting_cap: float = 0.18
    starting_buffer: float = 0.09
    remaining_cap: float = 0.073
    remaining_buffer: float = 0.09
    downside_before_buffer: float = 0.0

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
    Result of a single scenario (one forward reference asset return).

    All return values are investor returns from current fund NAV.

    Attributes:
        forward_return: The assumed forward reference asset return (decimal).
        hold_return: Investor return from current NAV under hold strategy.
        roll_return: Investor return from current NAV under roll strategy.
        hold_value: Ending position value under hold.
        roll_value: Ending position value under roll.
        advantage: Dollar advantage of rolling vs. holding (positive = roll wins).
        winner: "hold" or "roll".
    """
    forward_return: float
    hold_return: float
    roll_return: float
    hold_value: float
    roll_value: float
    advantage: float
    winner: str


@dataclass
class BreakevenResult:
    """
    Summary of the full breakeven analysis.

    Attributes:
        breakevens: List of (breakeven_return, roll_wins_above) tuples for ALL
            crossover points found. Each tuple contains the forward reference
            return at which hold and roll produce equal outcomes, and whether
            roll wins above that point.
        breakeven_return: Primary breakeven (first crossover), or None.
        roll_wins_above: Direction for the primary breakeven.
        recommendation: Human-readable recommendation string.
        scenario_df: DataFrame with full scenario grid results.
        scenarios: List of individual ScenarioResult objects.
    """
    breakevens: list
    breakeven_return: Optional[float]
    roll_wins_above: Optional[bool]
    recommendation: str
    scenario_df: pd.DataFrame
    scenarios: list


def run_scenarios(inputs: ScenarioInputs) -> BreakevenResult:
    """
    Run the full hold-vs-roll scenario analysis.

    For each forward reference asset return in the grid:
      - HOLD: Uses compounded returns (current ref + forward) and at-expiry
        payoff to compute investor return from current fund NAV.
      - ROLL: Sells at current NAV, pays costs/taxes, buys new ETF, applies
        new ETF payoff directly to the forward return.

    Args:
        inputs: ScenarioInputs with all position and replacement ETF details.

    Returns:
        BreakevenResult with full analysis.
    """
    # Generate the grid of forward reference asset returns
    returns = np.arange(
        inputs.min_return,
        inputs.max_return + inputs.return_step / 2,
        inputs.return_step,
    )
    returns = np.round(returns, 4)

    # Pre-calculate tax drag and roll proceeds
    tax_drag = calculate_tax_drag(inputs.position_value, inputs.tax_params)
    roll_proceeds = (
        inputs.position_value * (1.0 - inputs.transaction_cost_rate) - tax_drag
    )

    # Run each scenario
    scenarios = []
    for fwd_ret in returns:
        fwd = float(fwd_ret)

        # --- HOLD: Compound forward ref with current ref, apply at-expiry payoff ---
        hold_inv_return = calculate_hold_investor_return(
            forward_ref_return=fwd,
            current_ref_return=inputs.ref_return,
            fund_return=inputs.fund_return,
            starting_cap=inputs.starting_cap,
            starting_buffer=inputs.starting_buffer,
            downside_before_buffer=inputs.downside_before_buffer,
        )
        hold_value = inputs.position_value * (1.0 + hold_inv_return)

        # --- ROLL: Apply new ETF payoff to forward return ---
        roll_etf_return = calculate_roll_investor_return(
            forward_ref_return=fwd,
            new_cap=inputs.new_cap,
            new_buffer=inputs.new_buffer,
            new_downside_before_buffer=inputs.new_downside_before_buffer,
        )
        roll_value = roll_proceeds * (1.0 + roll_etf_return)
        roll_inv_return = (roll_value / inputs.position_value) - 1.0

        advantage = roll_value - hold_value
        winner = "roll" if advantage > 0 else "hold"

        scenarios.append(ScenarioResult(
            forward_return=fwd,
            hold_return=hold_inv_return,
            roll_return=roll_inv_return,
            hold_value=hold_value,
            roll_value=roll_value,
            advantage=advantage,
            winner=winner,
        ))

    # Build the DataFrame
    scenario_df = pd.DataFrame([
        {
            "Forward Ref Return": s.forward_return,
            "Hold Return": s.hold_return,
            "Roll Return": s.roll_return,
            "Hold Value": s.hold_value,
            "Roll Value": s.roll_value,
            "Roll Advantage ($)": s.advantage,
            "Winner": s.winner,
        }
        for s in scenarios
    ])

    # Find ALL breakeven points
    breakevens = _find_all_breakevens(scenarios)
    primary_be = breakevens[0] if breakevens else (None, None)

    # Generate recommendation
    recommendation = _generate_recommendation(
        breakevens=breakevens,
        inputs=inputs,
        tax_drag=tax_drag,
        scenarios=scenarios,
    )

    return BreakevenResult(
        breakevens=breakevens,
        breakeven_return=primary_be[0],
        roll_wins_above=primary_be[1],
        recommendation=recommendation,
        scenario_df=scenario_df,
        scenarios=scenarios,
    )


def _find_all_breakevens(
    scenarios: list[ScenarioResult],
) -> list[tuple[float, bool]]:
    """
    Find ALL breakeven points where hold and roll produce equal outcomes.

    Returns a list of (breakeven_return, roll_wins_above) tuples, one for
    each crossover point found. Crossovers are detected where the roll
    advantage changes sign.

    Args:
        scenarios: List of ScenarioResult objects, sorted by forward_return.

    Returns:
        List of (breakeven_return, roll_wins_above) tuples. Empty if no
        crossovers are found.
    """
    if len(scenarios) < 2:
        return []

    breakevens = []
    for i in range(1, len(scenarios)):
        prev = scenarios[i - 1]
        curr = scenarios[i]

        if prev.advantage * curr.advantage < 0:
            # Linear interpolation
            a = prev.advantage
            b = curr.advantage
            t = -a / (b - a) if (b - a) != 0 else 0.5

            be_return = prev.forward_return + t * (
                curr.forward_return - prev.forward_return
            )
            be_return = round(be_return, 4)

            # Roll wins above if advantage goes from negative to positive
            roll_wins_above = b > a

            breakevens.append((be_return, roll_wins_above))

    return breakevens


def _generate_recommendation(
    breakevens: list[tuple[float, bool]],
    inputs: ScenarioInputs,
    tax_drag: float,
    scenarios: list[ScenarioResult],
) -> str:
    """
    Generate a human-readable recommendation based on the breakeven analysis.

    Args:
        breakevens: List of (breakeven_return, roll_wins_above) tuples.
        inputs: Scenario inputs for context.
        tax_drag: Dollar tax drag from rolling.
        scenarios: Full list of scenarios for dominance check.

    Returns:
        Multi-line recommendation string.
    """
    lines = []

    if not breakevens:
        # One strategy dominates across all scenarios
        if all(s.advantage > 0 for s in scenarios):
            lines.append(
                "ROLL is advantageous across all modeled scenarios. "
                "The new ETF's superior cap/buffer profile more than offsets "
                "any transaction costs and tax drag."
            )
        elif all(s.advantage < 0 for s in scenarios):
            lines.append(
                "HOLD is advantageous across all modeled scenarios. "
                "The costs of rolling (transaction costs"
                + (" and tax drag" if tax_drag > 0 else "")
                + ") outweigh the benefits of the new ETF's cap/buffer profile."
            )
        else:
            lines.append(
                "No clear breakeven found. Results are very close across "
                "the modeled range."
            )
    else:
        # Report each breakeven
        for idx, (be_return, roll_wins_above) in enumerate(breakevens, 1):
            be_pct = be_return * 100
            prefix = "" if len(breakevens) == 1 else f"Breakeven #{idx}: "

            lines.append(f"{prefix}Breakeven forward ref return: {be_pct:+.1f}%")

            if roll_wins_above:
                lines.append(
                    f"  If the reference asset returns MORE than {be_pct:+.1f}%, "
                    f"ROLLING is advantageous."
                )
                lines.append(
                    f"  If the reference asset returns LESS than {be_pct:+.1f}%, "
                    f"HOLDING is better."
                )
            else:
                lines.append(
                    f"  If the reference asset returns LESS than {be_pct:+.1f}%, "
                    f"ROLLING is advantageous."
                )
                lines.append(
                    f"  If the reference asset returns MORE than {be_pct:+.1f}%, "
                    f"HOLDING is better."
                )

            if idx < len(breakevens):
                lines.append("")

    # Key factors
    lines.append("")
    lines.append("Key factors:")

    # Downside risk context
    if inputs.fund_return > 0:
        downside_pct = -inputs.fund_return / (1.0 + inputs.fund_return) * 100
        lines.append(
            f"  * Downside to period start: {downside_pct:+.1f}% from current NAV. "
            f"The fund is up {inputs.fund_return:.1%} — this gain is at risk "
            f"before the buffer provides any protection."
        )

    # Cap comparison
    remaining_cap = inputs.remaining_cap
    if inputs.new_cap > remaining_cap:
        lines.append(
            f"  * New cap ({inputs.new_cap:.1%}) is higher than remaining cap "
            f"({remaining_cap:.1%}), favoring ROLL in up markets."
        )
    elif inputs.new_cap < remaining_cap:
        lines.append(
            f"  * Remaining cap ({remaining_cap:.1%}) is higher than new cap "
            f"({inputs.new_cap:.1%}), favoring HOLD in up markets."
        )
    else:
        lines.append(f"  * Cap levels are equal ({remaining_cap:.1%}).")

    # Buffer comparison
    if inputs.new_buffer > inputs.starting_buffer:
        lines.append(
            f"  * New buffer ({inputs.new_buffer:.1%}) is larger than current "
            f"starting buffer ({inputs.starting_buffer:.1%}), favoring ROLL in "
            f"down markets."
        )
    elif inputs.new_buffer < inputs.starting_buffer:
        lines.append(
            f"  * Current starting buffer ({inputs.starting_buffer:.1%}) is larger "
            f"than new buffer ({inputs.new_buffer:.1%}), favoring HOLD in down "
            f"markets."
        )
    else:
        lines.append(
            f"  * Buffer levels are equal ({inputs.starting_buffer:.1%})."
        )

    # Rolling costs
    total_cost = inputs.position_value * inputs.transaction_cost_rate + max(0, tax_drag)
    if total_cost > 0:
        cost_components = []
        txn_cost = inputs.position_value * inputs.transaction_cost_rate
        if txn_cost > 0:
            cost_components.append(f"${txn_cost:,.0f} transaction cost")
        if tax_drag > 0:
            cost_components.append(f"${tax_drag:,.0f} tax drag")
        lines.append(
            f"  * Rolling costs: {' + '.join(cost_components)} = "
            f"${total_cost:,.0f} total, favoring HOLD."
        )
    elif tax_drag < 0:
        lines.append(
            f"  * Tax loss harvesting benefit of ${abs(tax_drag):,.0f} partially "
            f"offsets transaction costs, favoring ROLL."
        )

    return "\n".join(lines)
