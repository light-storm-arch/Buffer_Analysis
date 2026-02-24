"""
app.py — Streamlit UI and Visualization Layer

This is the main entry point for the Buffer ETF Rollover Analysis Tool.
Run with: streamlit run app.py

The UI is organized into sections:
  1. Ticker lookup and auto-population via web scraping
  2. Current ETF position inputs (fund return, ref return, cap/buffer levels)
  3. Replacement ETF inputs
  4. Account and tax settings
  5. Scenario analysis results:
     - Current position risk metrics (downside to buffer, remaining cap)
     - Hold vs. Roll comparison chart (Plotly)
     - Scenario table with investor returns from current NAV
     - ALL breakeven points and recommendation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date, timedelta

from scraper import scrape_etf_page, get_ticker_suggestions, get_ticker_description, INNOVATOR_TICKERS
from payoff import calculate_downside_metrics, calculate_hold_investor_return
from scenarios import ScenarioInputs, run_scenarios
from tax import TaxParams, format_tax_summary, calculate_tax_drag


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
    "its reset date versus **rolling** into a fresh ETF with a new cap and buffer. "
    "All returns are shown as **investor returns from current fund NAV**."
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
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

    ticker = st.text_input(
        "Ticker Symbol",
        value="BJUL",
        max_chars=10,
        help="Enter an Innovator Buffer ETF ticker (e.g., BJUL, PAPR, UOCT).",
    ).upper().strip()

    if ticker in INNOVATOR_TICKERS:
        st.caption(f"_{get_ticker_description(ticker)}_")

    if st.button("Fetch Data from Innovator", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            st.session_state.scraped_data = scrape_etf_page(ticker)
            st.session_state.scrape_attempted = True

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
    """Return the scraped value for a field if available, otherwise the default."""
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

if st.session_state.scraped_data and st.session_state.scraped_data.scrape_successful:
    st.info(
        "Fields below have been auto-populated from Innovator ETFs. "
        "All values are editable — adjust anything that appears incorrect."
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Outcome Period")

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
    st.subheader("Returns & Levels")

    fund_return_pct = st.number_input(
        "Fund Return from Period Start (%)",
        min_value=-99.0,
        max_value=200.0,
        value=scraped_or_default("fund_return", 0.10) * 100,
        step=0.1,
        format="%.2f",
        help=(
            "How much the fund NAV has changed since the outcome period started. "
            "Example: if the fund started at $30 and is now $33, enter 10.00."
        ),
    )

    ref_return_pct = st.number_input(
        "Reference Asset Return from Period Start (%)",
        min_value=-99.0,
        max_value=500.0,
        value=scraped_or_default("ref_return", 0.15) * 100,
        step=0.1,
        format="%.2f",
        help=(
            "How much the reference index (e.g., S&P 500) has changed since "
            "the outcome period started. This is NOT the same as the fund return."
        ),
    )

    ref_return_to_cap_pct = st.number_input(
        "Ref Return to Cap (%)",
        min_value=0.0,
        max_value=200.0,
        value=scraped_or_default("ref_return_to_cap", 0.03) * 100,
        step=0.1,
        format="%.2f",
        help=(
            "How much more the reference asset can return (from its current level) "
            "before the cap is hit. Starting cap = ref return + this value."
        ),
    )

    remaining_cap_pct = st.number_input(
        "Remaining Cap — from Fund NAV (%)",
        min_value=0.0,
        max_value=200.0,
        value=scraped_or_default("remaining_cap", 0.073) * 100,
        step=0.1,
        format="%.2f",
        help=(
            "Maximum additional return the fund can achieve from its current NAV. "
            "This is published by Innovator and differs from the reference-asset "
            "distance to cap."
        ),
    )

    remaining_buffer_pct = st.number_input(
        "Remaining Buffer (%)",
        min_value=0.0,
        max_value=100.0,
        value=scraped_or_default("remaining_buffer", 0.09) * 100,
        step=0.1,
        format="%.2f",
        help=(
            "Remaining downside protection as published by Innovator. "
            "For standard buffers: equals starting buffer when ref is positive."
        ),
    )

with col3:
    st.subheader("Position & Type")

    position_value = st.number_input(
        "Position Size ($)",
        min_value=0.0,
        value=100_000.0,
        step=1000.0,
        format="%.0f",
        help="Total dollar value of your current position.",
    )

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
        "Min Forward Ref Return (%)",
        min_value=-80.0,
        max_value=0.0,
        value=-30.0,
        step=5.0,
        format="%.0f",
    )

with col_sc2:
    max_return_pct = st.number_input(
        "Max Forward Ref Return (%)",
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
    fund_return = fund_return_pct / 100.0
    ref_return = ref_return_pct / 100.0
    ref_return_to_cap = ref_return_to_cap_pct / 100.0
    remaining_cap = remaining_cap_pct / 100.0
    remaining_buffer = remaining_buffer_pct / 100.0
    new_cap = new_cap_pct / 100.0
    new_buffer = new_buffer_pct / 100.0
    transaction_cost_rate = transaction_cost_bps / 10_000.0
    current_gap = current_gap_pct / 100.0
    new_gap = new_gap_pct / 100.0

    # Derive starting cap from reference return + distance to cap
    starting_cap = ref_return + ref_return_to_cap

    # Derive starting buffer from remaining buffer and current ref position
    # If ref >= 0: buffer hasn't been consumed, starting_buffer = remaining_buffer
    # If ref < 0: some buffer consumed, starting_buffer = remaining_buffer + consumed
    if ref_return >= 0:
        starting_buffer = remaining_buffer
    else:
        consumed = max(0.0, abs(ref_return) - current_gap)
        starting_buffer = remaining_buffer + consumed

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
        fund_return=fund_return,
        ref_return=ref_return,
        starting_cap=starting_cap,
        starting_buffer=starting_buffer,
        remaining_cap=remaining_cap,
        remaining_buffer=remaining_buffer,
        downside_before_buffer=current_gap,
        new_cap=new_cap,
        new_buffer=new_buffer,
        new_downside_before_buffer=new_gap,
        transaction_cost_rate=transaction_cost_rate,
        tax_params=tax_params,
        min_return=min_return_pct / 100.0,
        max_return=max_return_pct / 100.0,
        return_step=step_pct / 100.0,
    )

    # -------------------------------------------------------------------
    # Run the scenario engine
    # -------------------------------------------------------------------
    result = run_scenarios(inputs)

    # -------------------------------------------------------------------
    # Output Section A: Current Position Risk Metrics
    # -------------------------------------------------------------------
    st.header("Results")

    st.subheader("Current Position Risk Metrics")

    metrics = calculate_downside_metrics(
        fund_return=fund_return,
        current_ref_return=ref_return,
        starting_cap=starting_cap,
        starting_buffer=starting_buffer,
        downside_before_buffer=current_gap,
    )

    # Cross-validation: compare derived remaining cap with user input
    derived_remaining_cap = metrics["remaining_cap_from_nav"]
    cap_diff = abs(derived_remaining_cap - remaining_cap)
    if cap_diff > 0.005:  # More than 0.5% difference
        st.warning(
            f"The remaining cap you entered ({remaining_cap:.2%}) differs from "
            f"the value derived from your other inputs ({derived_remaining_cap:.2%}). "
            f"This may indicate a data entry inconsistency."
        )

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric(
            "Remaining Cap (Fund NAV)",
            f"{remaining_cap_pct:.2f}%",
            help="Maximum additional return from current fund NAV to cap.",
        )
    with metric_cols[1]:
        st.metric(
            "Remaining Buffer",
            f"{remaining_buffer_pct:.2f}%",
            help="Remaining downside protection from period start level.",
        )
    with metric_cols[2]:
        st.metric(
            "Downside to Period Start",
            f"{metrics['downside_to_period_start']:+.2%}",
            help=(
                "How much the fund NAV would drop from current if the reference "
                "asset returns to its period-start level (0% from start). This is "
                "the unprotected portion of your gain."
            ),
        )
    with metric_cols[3]:
        st.metric(
            "Downside to Buffer Exhaustion",
            f"{metrics['downside_to_buffer_exhaustion']:+.2%}",
            help=(
                "Investor return from current NAV when the reference asset has "
                "fallen enough to exhaust the entire buffer. Beyond this, losses "
                "are 1:1."
            ),
        )
    with metric_cols[4]:
        st.metric(
            "Days Left",
            f"{days_remaining}",
            help="Calendar days until the outcome period resets.",
        )

    # Derived values info box
    with st.expander("Derived Values"):
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Starting Cap (derived) | {starting_cap:.2%} |
        | Starting Buffer (derived) | {starting_buffer:.2%} |
        | Remaining Cap from NAV (derived) | {derived_remaining_cap:.2%} |
        | Ref Forward to Period Start | {metrics['ref_forward_to_zero']:+.2%} |
        | Ref Forward to Buffer Edge | {metrics['ref_forward_to_buffer_edge']:+.2%} |
        | Ref Forward to Cap | {metrics['ref_forward_to_cap']:+.2%} |
        """)

    # Show tax impact if taxable
    if is_taxable:
        tax_summary = format_tax_summary(position_value, tax_params)
        st.info(f"**Tax Impact if Rolling:** {tax_summary['description']}")

    # -------------------------------------------------------------------
    # Output Section B: Hold vs. Roll Comparison Chart
    # -------------------------------------------------------------------
    st.subheader("Hold vs. Roll: Investor Return from Current NAV")

    df = result.scenario_df

    fig = go.Figure()

    # Hold line
    fig.add_trace(go.Scatter(
        x=df["Forward Ref Return"] * 100,
        y=df["Hold Return"] * 100,
        mode="lines",
        name="Hold Current ETF",
        line=dict(color="#1f77b4", width=3),
        hovertemplate=(
            "Forward Ref: %{x:.1f}%<br>"
            "Hold Return: %{y:.2f}%<extra></extra>"
        ),
    ))

    # Roll line
    fig.add_trace(go.Scatter(
        x=df["Forward Ref Return"] * 100,
        y=df["Roll Return"] * 100,
        mode="lines",
        name="Roll to New ETF",
        line=dict(color="#ff7f0e", width=3),
        hovertemplate=(
            "Forward Ref: %{x:.1f}%<br>"
            "Roll Return: %{y:.2f}%<extra></extra>"
        ),
    ))

    # Plot ALL breakeven markers
    for idx, (be_return, be_above) in enumerate(result.breakevens):
        be_pct = be_return * 100
        # Calculate hold return at breakeven for the marker position
        be_hold_return = calculate_hold_investor_return(
            forward_ref_return=be_return,
            current_ref_return=ref_return,
            fund_return=fund_return,
            starting_cap=starting_cap,
            starting_buffer=starting_buffer,
            downside_before_buffer=current_gap,
        ) * 100

        label = f"BE{idx + 1}" if len(result.breakevens) > 1 else "BE"

        fig.add_trace(go.Scatter(
            x=[be_pct],
            y=[be_hold_return],
            mode="markers+text",
            name=f"{label}: {be_pct:+.1f}%",
            marker=dict(color="red", size=14, symbol="diamond"),
            text=[f"{label}: {be_pct:+.1f}%"],
            textposition="top center",
            textfont=dict(size=12, color="red"),
            hovertemplate=(
                f"{label}: {be_pct:+.1f}%<br>"
                f"Return: {be_hold_return:.2f}%<extra></extra>"
            ),
        ))

        fig.add_vline(
            x=be_pct, line_dash="dash", line_color="red",
            annotation_text=f"{label}: {be_pct:+.1f}%",
            annotation_position="top",
        )

    # Add horizontal line at 0% (no change from current NAV)
    fig.add_hline(
        y=0, line_dash="dot", line_color="gray",
        annotation_text="No change from current NAV",
        annotation_position="bottom right",
    )

    # Add vertical line at 0% forward return
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Investor Return from Current NAV: Hold vs. Roll",
        xaxis_title="Forward Reference Asset Return (%)",
        yaxis_title="Investor Return from Current NAV (%)",
        yaxis_ticksuffix="%",
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

    # -------------------------------------------------------------------
    # Output Section C: Scenario Table
    # -------------------------------------------------------------------
    st.subheader("Scenario Table")

    display_df = df.copy()
    display_df["Forward Ref Return"] = display_df["Forward Ref Return"].apply(
        lambda x: f"{x:+.1%}"
    )
    display_df["Hold Return"] = display_df["Hold Return"].apply(
        lambda x: f"{x:+.2%}"
    )
    display_df["Roll Return"] = display_df["Roll Return"].apply(
        lambda x: f"{x:+.2%}"
    )
    display_df["Hold Value"] = display_df["Hold Value"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_df["Roll Value"] = display_df["Roll Value"].apply(
        lambda x: f"${x:,.0f}"
    )
    display_df["Roll Advantage ($)"] = display_df["Roll Advantage ($)"].apply(
        lambda x: f"${x:+,.0f}"
    )

    def highlight_winner(row):
        """Apply green/red background based on which strategy wins."""
        if row["Winner"] == "roll":
            return ["background-color: rgba(0, 180, 0, 0.15)"] * len(row)
        else:
            return ["background-color: rgba(220, 0, 0, 0.10)"] * len(row)

    styled_df = display_df.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    st.caption(
        "Green rows: Rolling is advantageous. Red rows: Holding is advantageous. "
        "All returns are from the investor's current fund NAV."
    )

    # -------------------------------------------------------------------
    # Output Section D: Breakeven & Recommendation
    # -------------------------------------------------------------------
    st.subheader("Breakeven Analysis & Recommendation")

    if result.breakevens:
        be_cols = st.columns(min(len(result.breakevens), 4))
        for idx, (be_return, be_above) in enumerate(result.breakevens):
            with be_cols[idx % len(be_cols)]:
                label = f"Breakeven #{idx + 1}" if len(result.breakevens) > 1 else "Breakeven"
                direction = "above" if be_above else "below"
                st.metric(
                    label,
                    f"{be_return * 100:+.1f}%",
                    help=f"Roll wins {direction} this forward ref return.",
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

    # -------------------------------------------------------------------
    # Output Section E: Breakeven Background Calculations
    # -------------------------------------------------------------------
    if result.breakevens:
        st.subheader("Background Calculations")

        # Pre-compute roll costs — mirrors logic in run_scenarios
        _tax_drag = calculate_tax_drag(position_value, tax_params)
        _txn_cost = position_value * transaction_cost_rate
        _roll_proceeds = position_value * (1.0 - transaction_cost_rate) - _tax_drag

        def _payoff_zone(ref_total: float, cap: float, buffer: float, gap: float = 0.0):
            """Return (payoff_decimal, zone_label) for a reference return."""
            if ref_total >= 0:
                if ref_total >= cap:
                    return cap, f"**Capped** — total ref ({ref_total:+.2%}) ≥ cap ({cap:.2%})"
                return ref_total, (
                    f"**Upside participation** — 0% ≤ total ref ({ref_total:+.2%}) < cap ({cap:.2%})"
                )
            if gap > 0:
                if ref_total >= -gap:
                    return ref_total, (
                        f"**Ultra gap (1:1 loss)** — total ref ({ref_total:+.2%}) "
                        f"in [{-gap:.2%}, 0%)"
                    )
                if ref_total >= -(gap + buffer):
                    return -gap, (
                        f"**Buffered (frozen at {-gap:.2%})** — total ref ({ref_total:+.2%}) "
                        f"in [{-(gap + buffer):.2%}, {-gap:.2%})"
                    )
                return ref_total + buffer, (
                    f"**Beyond buffer** — total ref ({ref_total:+.2%}) "
                    f"< {-(gap + buffer):.2%}"
                )
            if ref_total >= -buffer:
                return 0.0, (
                    f"**Buffered (0% return)** — total ref ({ref_total:+.2%}) "
                    f"in [{-buffer:.2%}, 0%)"
                )
            return ref_total + buffer, (
                f"**Beyond buffer** — total ref ({ref_total:+.2%}) < {-buffer:.2%}"
            )

        num_bes = len(result.breakevens)
        for idx, (be_return, be_above) in enumerate(result.breakevens):
            expander_label = (
                f"Breakeven #{idx + 1} — Calculation Details"
                if num_bes > 1
                else "Breakeven — Calculation Details"
            )

            with st.expander(expander_label):
                direction = "above" if be_above else "below"
                st.markdown(
                    f"**Breakeven forward reference return: {be_return * 100:+.2f}%** — "
                    f"Roll wins {direction} this return. At this return the two strategies "
                    f"produce equal portfolio outcomes."
                )
                st.divider()

                # ---- Compute Hold at breakeven ----
                total_ref = (1.0 + ref_return) * (1.0 + be_return) - 1.0
                hold_payoff, hold_zone = _payoff_zone(
                    total_ref, starting_cap, starting_buffer, current_gap
                )
                hold_inv_return = (1.0 + hold_payoff) / (1.0 + fund_return) - 1.0
                hold_value_at_be = position_value * (1.0 + hold_inv_return)

                # ---- Compute Roll at breakeven ----
                roll_etf_return, roll_zone = _payoff_zone(
                    be_return, new_cap, new_buffer, new_gap
                )
                roll_value_at_be = _roll_proceeds * (1.0 + roll_etf_return)
                roll_inv_return = (roll_value_at_be / position_value) - 1.0

                col_h, col_r = st.columns(2)

                with col_h:
                    st.markdown("##### Hold Calculation")
                    st.markdown(
                        f"**Step 1 — Total ref return from period start:**\n\n"
                        f"The current ref return is compounded with the forward return "
                        f"to get the full return the ETF payoff will be based on.\n\n"
                        f"> (1 + {ref_return:.4f}) × (1 + {be_return:.4f}) − 1\n\n"
                        f"> = **{total_ref:+.4f} ({total_ref:.2%})**"
                    )
                    st.markdown(
                        f"**Step 2 — At-expiry payoff:**\n\n"
                        f"Starting cap: {starting_cap:.2%} | Starting buffer: {starting_buffer:.2%}"
                        + (f" | Ultra gap: {current_gap:.2%}" if current_gap > 0 else "")
                        + f"\n\nZone: {hold_zone}\n\n"
                        f"> At-expiry fund return (from period start) = **{hold_payoff:+.2%}**"
                    )
                    st.markdown(
                        f"**Step 3 — Investor return from current fund NAV:**\n\n"
                        f"The at-expiry fund NAV is divided by the current fund NAV "
                        f"to express the outcome from the investor's current cost basis.\n\n"
                        f"> (1 + {hold_payoff:.4f}) / (1 + {fund_return:.4f}) − 1\n\n"
                        f"> = **{hold_inv_return:+.2%}**\n\n"
                        f"> Hold ending value: ${position_value:,.0f} × "
                        f"(1 + {hold_inv_return:.4f}) = **${hold_value_at_be:,.2f}**"
                    )

                with col_r:
                    st.markdown("##### Roll Calculation")
                    st.markdown(
                        f"**Step 1 — Transaction cost:**\n\n"
                        f"> ${position_value:,.0f} × {transaction_cost_rate:.4f} "
                        f"= **${_txn_cost:,.2f}**"
                    )

                    if is_taxable:
                        gain_loss = position_value - cost_basis
                        combined_rate = (tax_rate_pct + state_tax_pct) / 100.0
                        gl_label = "gain" if gain_loss >= 0 else "loss"
                        st.markdown(
                            f"**Step 2 — Tax drag:**\n\n"
                            f"> Realized {gl_label}: "
                            f"${position_value:,.0f} − ${cost_basis:,.0f} "
                            f"= **${gain_loss:+,.2f}**\n\n"
                            f"> Combined tax rate: {tax_rate_pct:.1f}% fed "
                            f"+ {state_tax_pct:.1f}% state = {combined_rate:.2%}\n\n"
                            f"> Tax = ${gain_loss:+,.2f} × {combined_rate:.4f} "
                            f"= **${_tax_drag:+,.2f}**"
                        )
                    else:
                        st.markdown(
                            "**Step 2 — Tax drag:** $0 (tax-advantaged account)"
                        )

                    st.markdown(
                        f"**Step 3 — Roll proceeds after costs and taxes:**\n\n"
                        f"> ${position_value:,.0f} × (1 − {transaction_cost_rate:.4f}) "
                        f"− ${_tax_drag:,.2f}\n\n"
                        f"> = **${_roll_proceeds:,.2f}**"
                    )
                    st.markdown(
                        f"**Step 4 — New ETF payoff:**\n\n"
                        f"New cap: {new_cap:.2%} | New buffer: {new_buffer:.2%}"
                        + (f" | Ultra gap: {new_gap:.2%}" if new_gap > 0 else "")
                        + f"\n\nZone: {roll_zone}\n\n"
                        f"> New ETF return = **{roll_etf_return:+.2%}**"
                    )
                    st.markdown(
                        f"**Step 5 — Roll ending value and investor return:**\n\n"
                        f"> ${_roll_proceeds:,.2f} × (1 + {roll_etf_return:.4f}) "
                        f"= **${roll_value_at_be:,.2f}**\n\n"
                        f"> Roll investor return: "
                        f"${roll_value_at_be:,.2f} / ${position_value:,.0f} − 1 "
                        f"= **{roll_inv_return:+.2%}**"
                    )

                st.divider()
                st.caption(
                    f"At {be_return * 100:+.2f}% forward ref return — "
                    f"Hold: ${hold_value_at_be:,.2f} ({hold_inv_return:+.2%}) | "
                    f"Roll: ${roll_value_at_be:,.2f} ({roll_inv_return:+.2%}). "
                    f"Small differences are expected: the breakeven is found by "
                    f"linear interpolation between 1% grid points, so the two "
                    f"values may not be exactly equal at the interpolated point."
                )

    # -------------------------------------------------------------------
    # Output Section F: Payoff Explanation (educational)
    # -------------------------------------------------------------------
    with st.expander("Understanding the Buffer ETF Payoff Structure"):
        st.markdown("""
        **How Buffer ETFs Work:**

        Buffer ETFs use FLEX options to create a defined outcome over a fixed period
        (usually 1 year):

        - **Capped Upside:** Gains are limited to the cap level. If the S&P 500 returns
          25% but the cap is 15%, the ETF returns 15%.
        - **Buffered Downside:** Losses up to the buffer level are absorbed. If the S&P 500
          drops 7% and the buffer is 9%, the ETF returns 0% (no loss).
        - **Beyond the Buffer:** Losses exceeding the buffer pass through 1:1. If the S&P 500
          drops 20% and the buffer is 9%, the ETF returns -11% (20% - 9%).

        **Why Fund Return != Reference Asset Return Mid-Period:**

        Mid-period, the fund NAV does NOT track the reference asset 1:1. The fund's
        options still embed time value. If the S&P is up 15%, the fund might only be up
        10%. This difference narrows as the outcome period end approaches.

        **The "Downside to Period Start" Risk:**

        If the fund is up 10%, the client has 10% of unrealized gain. If the reference
        asset returns to 0% from period start, the at-expiry payoff is 0% — meaning the
        fund NAV goes back to its starting value. The client loses their 10% gain. The
        buffer only protects against losses BELOW the period-start level.

        **The Roll Decision:**

        When a new ETF launches with a fresh cap (say 16%), it may be advantageous to sell
        the current ETF and buy the new one — provided the benefits outweigh the transaction
        costs, tax drag, and the risk of giving up the current position's characteristics.
        """)


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
