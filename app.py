"""
app.py — Streamlit UI and Visualization Layer

This is the main entry point for the Buffer ETF Rollover Analysis Tool.
Run with: streamlit run app.py

The UI is organized into sections:
  1. Ticker lookup and auto-population via web scraping
  2. Current ETF position inputs (editable, pre-filled if scraped)
  3. Replacement ETF inputs
  4. Account and tax settings
  5. Scenario analysis results:
     - Current ETF status summary
     - Hold vs. Roll comparison chart (Plotly)
     - Scenario table with conditional formatting
     - Breakeven and recommendation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date, timedelta

from scraper import scrape_etf_page, get_ticker_suggestions, get_ticker_description, INNOVATOR_TICKERS
from payoff import PayoffParams, calculate_buffer_etf_return, estimate_remaining_params
from scenarios import ScenarioInputs, run_scenarios
from tax import TaxParams, format_tax_summary


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Buffer ETF Rollover Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Buffer ETF Rollover Analyzer")
st.markdown(
    "Compare the expected outcomes of **holding** your current buffer ETF through "
    "its reset date versus **rolling** into a fresh ETF with a new cap and buffer."
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
# We use session state to persist scraped data and avoid re-scraping on every
# Streamlit rerun (which happens on any widget interaction).
if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None
if "scrape_attempted" not in st.session_state:
    st.session_state.scrape_attempted = False


# ---------------------------------------------------------------------------
# Sidebar: Ticker Lookup & Scraping
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ETF Lookup")
    st.markdown(
        "Enter an Innovator ETF ticker to auto-populate fields from "
        "[innovatoretfs.com](https://www.innovatoretfs.com)."
    )

    # Ticker input with suggestions
    ticker = st.text_input(
        "Ticker Symbol",
        value="BJUL",
        max_chars=10,
        help="Enter an Innovator Buffer ETF ticker (e.g., BJUL, PAPR, UOCT).",
    ).upper().strip()

    # Show ticker description if known
    if ticker in INNOVATOR_TICKERS:
        st.caption(f"_{get_ticker_description(ticker)}_")

    # Scrape button
    if st.button("Fetch Data from Innovator", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            st.session_state.scraped_data = scrape_etf_page(ticker)
            st.session_state.scrape_attempted = True

    # Show scrape status
    if st.session_state.scrape_attempted:
        data = st.session_state.scraped_data
        if data and data.scrape_successful:
            st.success("Data auto-populated from Innovator ETFs.")
            if data.scrape_timestamp:
                st.caption(f"Fetched at {data.scrape_timestamp.strftime('%I:%M %p')}")
        else:
            msg = data.scrape_message if data else "Scraping was not attempted."
            st.warning(msg)
            st.info("Please enter values manually below. All fields are editable.")

    st.divider()

    # Quick reference: common ticker table
    with st.expander("Common Innovator ETF Tickers"):
        ticker_df = pd.DataFrame([
            {"Ticker": t, "Type": d}
            for t, d in sorted(INNOVATOR_TICKERS.items())
        ])
        st.dataframe(ticker_df, hide_index=True, use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Helper: get scraped value or default
# ---------------------------------------------------------------------------
def scraped_or_default(field_name: str, default):
    """
    Return the scraped value for a field if available, otherwise the default.

    This lets us pre-fill input fields with scraped data while still allowing
    the user to override anything.
    """
    data = st.session_state.scraped_data
    if data and data.scrape_successful:
        value = getattr(data, field_name, None)
        if value is not None:
            return value
    return default


# ---------------------------------------------------------------------------
# Section 1: Current ETF Position
# ---------------------------------------------------------------------------
st.header("Current ETF Position")

# Show a badge if values were auto-populated
if st.session_state.scraped_data and st.session_state.scraped_data.scrape_successful:
    st.info(
        "Fields below have been auto-populated from Innovator ETFs. "
        "All values are editable — adjust anything that appears incorrect."
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Outcome Period")

    # Default dates: assume a standard 1-year outcome period
    default_start = scraped_or_default(
        "outcome_period_start",
        date.today() - timedelta(days=180),
    )
    default_reset = scraped_or_default(
        "reset_date",
        date.today() + timedelta(days=185),
    )

    outcome_start = st.date_input(
        "Outcome Period Start Date",
        value=default_start,
        help="The date the current outcome period began.",
    )
    reset_date = st.date_input(
        "Reset Date (End of Outcome Period)",
        value=default_reset,
        help="The date the current outcome period ends and the ETF resets.",
    )

    days_remaining = max(0, (reset_date - date.today()).days)
    days_elapsed = max(0, (date.today() - outcome_start).days)
    total_days = max(1, (reset_date - outcome_start).days)
    st.metric("Days Remaining", f"{days_remaining}")
    st.progress(days_elapsed / total_days, text=f"{days_elapsed}/{total_days} days elapsed")

with col2:
    st.subheader("Cap & Buffer Levels")

    # Starting values (at period inception)
    starting_cap_pct = st.number_input(
        "Starting Cap (%)",
        min_value=0.0,
        max_value=100.0,
        value=scraped_or_default("starting_cap", 0.15) * 100,
        step=0.1,
        format="%.2f",
        help="The cap level set at the beginning of the outcome period.",
    )

    starting_buffer_pct = st.number_input(
        "Starting Buffer (%)",
        min_value=0.0,
        max_value=100.0,
        value=scraped_or_default("starting_buffer", 0.09) * 100,
        step=0.1,
        format="%.2f",
        help="The buffer level set at the beginning of the outcome period.",
    )

    # Remaining values (from current price)
    remaining_cap_pct = st.number_input(
        "Remaining Cap (%)",
        min_value=0.0,
        max_value=200.0,
        value=scraped_or_default("remaining_cap", 0.10) * 100,
        step=0.1,
        format="%.2f",
        help="Maximum additional upside from the current ETF price to the cap.",
    )

    remaining_buffer_pct = st.number_input(
        "Remaining Buffer (%)",
        min_value=0.0,
        max_value=100.0,
        value=scraped_or_default("remaining_buffer", 0.09) * 100,
        step=0.1,
        format="%.2f",
        help="Remaining downside protection from the current ETF price.",
    )

with col3:
    st.subheader("NAV & Position")

    starting_nav = st.number_input(
        "Starting NAV ($)",
        min_value=0.01,
        value=scraped_or_default("starting_nav", 30.00),
        step=0.01,
        format="%.2f",
        help="NAV at the beginning of the outcome period.",
    )

    current_nav = st.number_input(
        "Current NAV ($)",
        min_value=0.01,
        value=scraped_or_default("current_nav", 31.50),
        step=0.01,
        format="%.2f",
        help="Current net asset value of the ETF.",
    )

    position_value = st.number_input(
        "Position Size ($)",
        min_value=0.0,
        value=100_000.0,
        step=1000.0,
        format="%.0f",
        help="Total dollar value of your current position.",
    )

    # ETF type selector for ultra buffers
    etf_type = st.selectbox(
        "Buffer Type",
        options=["Standard (9%)", "Power (15%)", "Ultra (5-35%)", "Custom"],
        index=0,
        help="Determines the buffer structure. Ultra buffers have a gap before protection starts.",
    )

    current_gap_pct = 0.0
    if etf_type == "Ultra (5-35%)":
        current_gap_pct = 5.0
    elif etf_type == "Custom":
        current_gap_pct = st.number_input(
            "Downside Before Buffer (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            format="%.1f",
            help="For ultra buffers: the loss absorbed by the investor before buffer protection starts.",
        )


# ---------------------------------------------------------------------------
# Section 2: Replacement ETF
# ---------------------------------------------------------------------------
st.header("Replacement ETF (New Position)")

col_new1, col_new2 = st.columns(2)

with col_new1:
    new_cap_pct = st.number_input(
        "New ETF Cap (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1,
        format="%.2f",
        help="The cap being offered on the new ETF.",
    )

    new_buffer_pct = st.number_input(
        "New ETF Buffer (%)",
        min_value=0.0,
        max_value=100.0,
        value=9.0,
        step=0.1,
        format="%.2f",
        help="The buffer level on the new ETF.",
    )

    new_etf_type = st.selectbox(
        "New Buffer Type",
        options=["Standard (9%)", "Power (15%)", "Ultra (5-35%)", "Custom"],
        index=0,
        key="new_etf_type",
        help="Buffer structure of the replacement ETF.",
    )

    new_gap_pct = 0.0
    if new_etf_type == "Ultra (5-35%)":
        new_gap_pct = 5.0
    elif new_etf_type == "Custom":
        new_gap_pct = st.number_input(
            "New Downside Before Buffer (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            format="%.1f",
            key="new_gap",
        )

with col_new2:
    transaction_cost_bps = st.number_input(
        "Transaction Costs (bps)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=1.0,
        format="%.0f",
        help="Total cost to execute the sell and buy, in basis points (1 bp = 0.01%).",
    )


# ---------------------------------------------------------------------------
# Section 3: Account & Tax Settings
# ---------------------------------------------------------------------------
st.header("Account & Tax Settings")

col_tax1, col_tax2 = st.columns(2)

with col_tax1:
    account_type = st.radio(
        "Account Type",
        options=["Tax-Advantaged (IRA, 401k, Roth)", "Taxable"],
        index=0,
        help="Tax-advantaged accounts have no tax drag when selling.",
    )
    is_taxable = account_type == "Taxable"

with col_tax2:
    if is_taxable:
        cost_basis = st.number_input(
            "Cost Basis ($)",
            min_value=0.0,
            value=95_000.0,
            step=1000.0,
            format="%.0f",
            help="What you originally paid for the position.",
        )
        tax_rate_pct = st.number_input(
            "Capital Gains Tax Rate (%)",
            min_value=0.0,
            max_value=60.0,
            value=20.0,
            step=0.5,
            format="%.1f",
            help=(
                "Your applicable capital gains tax rate. Short-term (held < 1 year) "
                "is typically your ordinary income rate. Long-term is 0%, 15%, or 20%."
            ),
        )
        state_tax_pct = st.number_input(
            "State Tax Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            format="%.1f",
            help="Additional state capital gains tax rate, if applicable.",
        )
    else:
        cost_basis = position_value
        tax_rate_pct = 0.0
        state_tax_pct = 0.0


# ---------------------------------------------------------------------------
# Section 4: Scenario Settings
# ---------------------------------------------------------------------------
st.header("Scenario Settings")

col_sc1, col_sc2, col_sc3 = st.columns(3)

with col_sc1:
    min_return_pct = st.number_input(
        "Min Forward Return (%)",
        min_value=-80.0,
        max_value=0.0,
        value=-30.0,
        step=5.0,
        format="%.0f",
    )

with col_sc2:
    max_return_pct = st.number_input(
        "Max Forward Return (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=5.0,
        format="%.0f",
    )

with col_sc3:
    step_pct = st.number_input(
        "Step Size (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.5,
        format="%.1f",
    )


# ---------------------------------------------------------------------------
# Run Analysis
# ---------------------------------------------------------------------------
st.divider()

if st.button("Run Analysis", type="primary", use_container_width=True):

    # Convert percentage inputs to decimals
    remaining_cap = remaining_cap_pct / 100.0
    remaining_buffer = remaining_buffer_pct / 100.0
    new_cap = new_cap_pct / 100.0
    new_buffer = new_buffer_pct / 100.0
    transaction_cost_rate = transaction_cost_bps / 10_000.0
    current_gap = current_gap_pct / 100.0
    new_gap = new_gap_pct / 100.0

    # Build tax params
    tax_params = TaxParams(
        is_taxable=is_taxable,
        cost_basis=cost_basis,
        tax_rate=tax_rate_pct / 100.0,
        state_tax_rate=state_tax_pct / 100.0,
    )

    # Build scenario inputs
    inputs = ScenarioInputs(
        position_value=position_value,
        remaining_cap=remaining_cap,
        remaining_buffer=remaining_buffer,
        current_downside_before_buffer=current_gap,
        new_cap=new_cap,
        new_buffer=new_buffer,
        new_downside_before_buffer=new_gap,
        transaction_cost_rate=transaction_cost_rate,
        tax_params=tax_params,
        min_return=min_return_pct / 100.0,
        max_return=max_return_pct / 100.0,
        return_step=step_pct / 100.0,
    )

    # -----------------------------------------------------------------------
    # Run the scenario engine
    # -----------------------------------------------------------------------
    result = run_scenarios(inputs)

    # -----------------------------------------------------------------------
    # Output Section A: Current ETF Status
    # -----------------------------------------------------------------------
    st.header("Results")

    st.subheader("Current ETF Status")
    current_return = (current_nav / starting_nav - 1.0) if starting_nav > 0 else 0.0

    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric(
            "Remaining Cap",
            f"{remaining_cap_pct:.2f}%",
            help="Maximum additional upside from here.",
        )
    with status_cols[1]:
        st.metric(
            "Remaining Buffer",
            f"{remaining_buffer_pct:.2f}%",
            help="Remaining downside protection.",
        )
    with status_cols[2]:
        st.metric(
            "Days Left",
            f"{days_remaining}",
            help="Calendar days until the outcome period resets.",
        )
    with status_cols[3]:
        st.metric(
            "Current Gain/Loss",
            f"{current_return:+.2%}",
            delta=f"${position_value * current_return:+,.0f}",
            help="Return since the outcome period started.",
        )

    # Show tax impact if taxable
    if is_taxable:
        tax_summary = format_tax_summary(position_value, tax_params)
        st.info(f"**Tax Impact if Rolling:** {tax_summary['description']}")

    # -----------------------------------------------------------------------
    # Output Section B: Hold vs. Roll Comparison Chart
    # -----------------------------------------------------------------------
    st.subheader("Hold vs. Roll Comparison")

    df = result.scenario_df

    fig = go.Figure()

    # Hold line
    fig.add_trace(go.Scatter(
        x=df["Market Return"] * 100,
        y=df["Hold Value"],
        mode="lines",
        name="Hold Current ETF",
        line=dict(color="#1f77b4", width=3),
        hovertemplate="Market: %{x:.1f}%<br>Hold Value: $%{y:,.0f}<extra></extra>",
    ))

    # Roll line
    fig.add_trace(go.Scatter(
        x=df["Market Return"] * 100,
        y=df["Roll Value"],
        mode="lines",
        name="Roll to New ETF",
        line=dict(color="#ff7f0e", width=3),
        hovertemplate="Market: %{x:.1f}%<br>Roll Value: $%{y:,.0f}<extra></extra>",
    ))

    # Breakeven marker
    if result.breakeven_return is not None:
        # Interpolate the value at breakeven
        be_pct = result.breakeven_return * 100
        # Find the approximate hold value at breakeven
        hold_params_temp = PayoffParams(
            cap=remaining_cap, buffer=remaining_buffer,
            downside_before_buffer=current_gap,
        )
        be_etf_ret = calculate_buffer_etf_return(result.breakeven_return, hold_params_temp)
        be_value = position_value * (1.0 + be_etf_ret)

        fig.add_trace(go.Scatter(
            x=[be_pct],
            y=[be_value],
            mode="markers+text",
            name=f"Breakeven ({be_pct:+.1f}%)",
            marker=dict(color="red", size=14, symbol="diamond"),
            text=[f"Breakeven: {be_pct:+.1f}%"],
            textposition="top center",
            textfont=dict(size=12, color="red"),
            hovertemplate=f"Breakeven: {be_pct:+.1f}%<br>Value: ${be_value:,.0f}<extra></extra>",
        ))

        # Add vertical line at breakeven
        fig.add_vline(
            x=be_pct, line_dash="dash", line_color="red",
            annotation_text=f"Breakeven: {be_pct:+.1f}%",
            annotation_position="top",
        )

    # Add horizontal line at current position value
    fig.add_hline(
        y=position_value, line_dash="dot", line_color="gray",
        annotation_text=f"Current Value: ${position_value:,.0f}",
        annotation_position="bottom right",
    )

    # Add vertical line at 0% return
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Ending Position Value: Hold vs. Roll",
        xaxis_title="Forward Market Return (%)",
        yaxis_title="Ending Position Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=500,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Output Section C: Scenario Table
    # -----------------------------------------------------------------------
    st.subheader("Scenario Table")

    # Format the table for display
    display_df = df.copy()
    display_df["Market Return"] = display_df["Market Return"].apply(lambda x: f"{x:+.1%}")
    display_df["Hold Value"] = display_df["Hold Value"].apply(lambda x: f"${x:,.0f}")
    display_df["Roll Value"] = display_df["Roll Value"].apply(lambda x: f"${x:,.0f}")
    display_df["Hold Return"] = display_df["Hold Return"].apply(lambda x: f"{x:+.2%}")
    display_df["Roll Return"] = display_df["Roll Return"].apply(lambda x: f"{x:+.2%}")
    display_df["Roll Advantage ($)"] = display_df["Roll Advantage ($)"].apply(
        lambda x: f"${x:+,.0f}"
    )

    # Apply conditional formatting using Streamlit's native styling
    # We use a custom function to highlight rows
    def highlight_winner(row):
        """Apply green/red background based on which strategy wins."""
        if row["Winner"] == "roll":
            return ["background-color: rgba(0, 180, 0, 0.15)"] * len(row)
        else:
            return ["background-color: rgba(220, 0, 0, 0.10)"] * len(row)

    styled_df = display_df.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    st.caption(
        "Green rows: Rolling is advantageous. Red rows: Holding is advantageous."
    )

    # -----------------------------------------------------------------------
    # Output Section D: Breakeven & Recommendation
    # -----------------------------------------------------------------------
    st.subheader("Breakeven Analysis & Recommendation")

    if result.breakeven_return is not None:
        be_col1, be_col2 = st.columns([1, 2])
        with be_col1:
            st.metric(
                "Breakeven Market Return",
                f"{result.breakeven_return * 100:+.1f}%",
                help="The forward market return at which hold and roll produce equal outcomes.",
            )
        with be_col2:
            if result.roll_wins_above:
                st.markdown(
                    f"Rolling wins if the market returns **above** "
                    f"{result.breakeven_return * 100:+.1f}%.  \n"
                    f"Holding wins if the market returns **below** "
                    f"{result.breakeven_return * 100:+.1f}%."
                )
            else:
                st.markdown(
                    f"Rolling wins if the market returns **below** "
                    f"{result.breakeven_return * 100:+.1f}%.  \n"
                    f"Holding wins if the market returns **above** "
                    f"{result.breakeven_return * 100:+.1f}%."
                )
    else:
        st.info(
            "No breakeven found within the modeled range — one strategy dominates "
            "across all scenarios."
        )

    # Full recommendation
    st.text_area(
        "Detailed Recommendation",
        value=result.recommendation,
        height=250,
        disabled=True,
    )

    # -----------------------------------------------------------------------
    # Output Section E: Payoff Diagram (educational)
    # -----------------------------------------------------------------------
    with st.expander("Understanding the Buffer ETF Payoff Structure"):
        st.markdown("""
        **How Buffer ETFs Work:**

        Buffer ETFs use options to create a defined outcome over a fixed period (usually 1 year):

        - **Capped Upside:** Gains are limited to the cap level. If the S&P 500 returns 25%
          but the cap is 15%, the ETF returns 15%.
        - **Buffered Downside:** Losses up to the buffer level are absorbed. If the S&P 500
          drops 7% and the buffer is 9%, the ETF returns 0% (no loss).
        - **Beyond the Buffer:** Losses exceeding the buffer pass through 1:1. If the S&P 500
          drops 20% and the buffer is 9%, the ETF returns -11% (20% - 9%).

        **Why the Remaining Cap/Buffer Matter:**

        As the underlying index moves during the outcome period, the remaining cap and buffer
        shift. If the market has already risen 5% and the starting cap was 15%, the remaining
        cap is approximately 10%. The remaining buffer also adjusts.

        **The Roll Decision:**

        When a new ETF launches with a fresh cap (say 16%), it may be advantageous to sell
        the current ETF (with only 10% remaining cap) and buy the new one — provided the
        benefits outweigh the transaction costs and any tax drag.
        """)

        # Draw the payoff diagram
        payoff_returns = np.arange(-0.35, 0.35, 0.005)

        hold_payoffs = [
            calculate_buffer_etf_return(r, PayoffParams(
                cap=remaining_cap, buffer=remaining_buffer,
                downside_before_buffer=current_gap,
            )) for r in payoff_returns
        ]
        roll_payoffs = [
            calculate_buffer_etf_return(r, PayoffParams(
                cap=new_cap, buffer=new_buffer,
                downside_before_buffer=new_gap,
            )) for r in payoff_returns
        ]

        payoff_fig = go.Figure()
        payoff_fig.add_trace(go.Scatter(
            x=payoff_returns * 100,
            y=[p * 100 for p in hold_payoffs],
            mode="lines",
            name=f"Current ETF (Cap: {remaining_cap_pct:.1f}%, Buffer: {remaining_buffer_pct:.1f}%)",
            line=dict(color="#1f77b4", width=2),
        ))
        payoff_fig.add_trace(go.Scatter(
            x=payoff_returns * 100,
            y=[p * 100 for p in roll_payoffs],
            mode="lines",
            name=f"New ETF (Cap: {new_cap_pct:.1f}%, Buffer: {new_buffer_pct:.1f}%)",
            line=dict(color="#ff7f0e", width=2),
        ))
        # 1:1 reference line
        payoff_fig.add_trace(go.Scatter(
            x=payoff_returns * 100,
            y=payoff_returns * 100,
            mode="lines",
            name="Unprotected (1:1)",
            line=dict(color="gray", width=1, dash="dot"),
        ))

        payoff_fig.update_layout(
            title="Payoff Structure Comparison",
            xaxis_title="Underlying Market Return (%)",
            yaxis_title="ETF Return (%)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(payoff_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "**Disclaimer:** This tool is for educational and informational purposes only. "
    "It does not constitute investment or tax advice. Buffer ETF payoffs are approximations "
    "based on simplified models. Actual outcomes depend on options pricing, fund expenses, "
    "tracking error, and market conditions. Consult a qualified financial advisor before "
    "making investment decisions."
)
