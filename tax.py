"""
tax.py — Tax Overlay for Taxable Account Adjustments

When a buffer ETF position is sold in a taxable account, the realized gain
or loss triggers a tax event. This module calculates the tax drag (or tax
benefit) from selling the current position before rolling into a new ETF.

Key concepts:
  - Cost basis: What the investor originally paid for the shares.
  - Current value: The market value of the position today.
  - Realized gain: Current value minus cost basis. If positive, taxes are owed.
    If negative, it's a tax loss that can offset other gains.
  - Tax rate: Depends on holding period (short-term vs. long-term capital gains)
    and the investor's tax bracket. We let the user specify this directly.

For tax-advantaged accounts (IRA, 401k, etc.), there is no tax drag, and
this module returns zero.

Important disclaimer: This is a simplified model for scenario analysis only.
It does not constitute tax advice. Actual tax treatment depends on the
investor's full tax situation, wash sale rules, state taxes, etc.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaxParams:
    """
    Parameters for tax drag calculation.

    Attributes:
        is_taxable: Whether the account is taxable (True) or tax-advantaged (False).
        cost_basis: The original purchase price of the position (total dollars).
        tax_rate: The applicable capital gains tax rate as a decimal (e.g., 0.20 for 20%).
            For short-term gains (held < 1 year), this is typically the investor's
            ordinary income rate. For long-term gains, it's typically 0%, 15%, or 20%
            depending on income.
        state_tax_rate: Optional additional state capital gains tax rate (decimal).
            Set to 0.0 if the state has no income tax or for simplicity.
    """
    is_taxable: bool = False
    cost_basis: float = 0.0
    tax_rate: float = 0.20
    state_tax_rate: float = 0.0

    @property
    def combined_tax_rate(self) -> float:
        """Total tax rate combining federal and state."""
        return self.tax_rate + self.state_tax_rate


def calculate_tax_drag(
    current_value: float,
    tax_params: TaxParams,
) -> float:
    """
    Calculate the dollar amount of tax owed (or saved) when selling a position.

    If the position has a gain, the tax drag is positive (taxes reduce proceeds).
    If the position has a loss, the tax drag is negative (tax benefit increases
    effective proceeds through loss harvesting).

    For tax-advantaged accounts, this always returns 0.

    Args:
        current_value: Current market value of the position in dollars.
        tax_params: Tax parameters including basis, rate, and account type.

    Returns:
        Dollar amount of tax drag. Positive means taxes owed on gains.
        Negative means tax benefit from harvesting losses.

    Examples:
        >>> params = TaxParams(is_taxable=True, cost_basis=90000, tax_rate=0.20)
        >>> calculate_tax_drag(100000, params)
        2000.0  # $10k gain * 20% = $2k tax

        >>> params = TaxParams(is_taxable=True, cost_basis=105000, tax_rate=0.20)
        >>> calculate_tax_drag(100000, params)
        -1000.0  # $5k loss * 20% = -$1k (tax benefit)

        >>> params = TaxParams(is_taxable=False, cost_basis=90000, tax_rate=0.20)
        >>> calculate_tax_drag(100000, params)
        0.0  # Tax-advantaged: no tax
    """
    # No tax in tax-advantaged accounts (IRA, 401k, Roth, etc.)
    if not tax_params.is_taxable:
        return 0.0

    # Calculate realized gain or loss
    realized_gain = current_value - tax_params.cost_basis

    # Apply combined tax rate
    tax_amount = realized_gain * tax_params.combined_tax_rate

    return tax_amount


def calculate_after_tax_proceeds(
    current_value: float,
    tax_params: TaxParams,
    transaction_cost_rate: float = 0.0,
) -> float:
    """
    Calculate net proceeds after selling a position, accounting for both
    taxes and transaction costs.

    This is the amount available to invest in the new ETF after the roll.

    Args:
        current_value: Current market value of the position in dollars.
        tax_params: Tax parameters for the account.
        transaction_cost_rate: Transaction cost as a fraction of position value
            (e.g., 0.001 for 10 basis points).

    Returns:
        Net dollar proceeds available for reinvestment.

    Example:
        >>> params = TaxParams(is_taxable=True, cost_basis=90000, tax_rate=0.20)
        >>> calculate_after_tax_proceeds(100000, params, 0.001)
        97900.0  # $100k - $100 transaction cost - $2000 tax = $97,900
    """
    transaction_cost = current_value * transaction_cost_rate
    tax_drag = calculate_tax_drag(current_value, tax_params)
    return current_value - transaction_cost - tax_drag


def format_tax_summary(
    current_value: float,
    tax_params: TaxParams,
) -> dict:
    """
    Generate a human-readable summary of the tax implications of selling.

    Useful for displaying in the Streamlit UI so the user understands
    the tax impact before deciding to roll.

    Args:
        current_value: Current market value of the position.
        tax_params: Tax parameters for the account.

    Returns:
        Dictionary with formatted tax summary information:
          - realized_gain: Dollar gain/loss from selling
          - tax_amount: Dollar tax owed (or benefit)
          - effective_rate: Effective tax rate on the position
          - is_gain: Whether the position has a gain (True) or loss (False)
          - description: Human-readable summary string
    """
    if not tax_params.is_taxable:
        return {
            "realized_gain": 0.0,
            "tax_amount": 0.0,
            "effective_rate": 0.0,
            "is_gain": False,
            "description": "Tax-advantaged account — no tax impact from rolling.",
        }

    realized_gain = current_value - tax_params.cost_basis
    tax_amount = calculate_tax_drag(current_value, tax_params)
    is_gain = realized_gain > 0

    # Effective tax rate relative to the total position
    effective_rate = abs(tax_amount) / current_value if current_value > 0 else 0.0

    if is_gain:
        description = (
            f"Selling triggers a ${realized_gain:,.2f} capital gain. "
            f"At a {tax_params.combined_tax_rate:.1%} tax rate, "
            f"tax owed is ${tax_amount:,.2f} "
            f"({effective_rate:.2%} of position value)."
        )
    elif realized_gain < 0:
        description = (
            f"Selling realizes a ${abs(realized_gain):,.2f} capital loss. "
            f"At a {tax_params.combined_tax_rate:.1%} tax rate, "
            f"the tax benefit is ${abs(tax_amount):,.2f}. "
            f"Note: Wash sale rules may apply if buying a substantially "
            f"identical replacement within 30 days."
        )
    else:
        description = "Position is at breakeven — no tax impact from selling."

    return {
        "realized_gain": realized_gain,
        "tax_amount": tax_amount,
        "effective_rate": effective_rate,
        "is_gain": is_gain,
        "description": description,
    }
