"""
Performance Tracker Page for Streamlit App.

This module provides the Performance Tracker page that displays:
- Portfolio NAV with and without overlays
- Performance metrics comparison
- Overlay signal breakdowns
- Historical allocation charts

IMPORTANT: This page is isolated with error handling to ensure
the main Streamlit app remains functional even if this page has issues.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def show_performance_tracker_ui():
    """
    Main entry point for the Performance Tracker page.

    This function is wrapped with error handling to ensure
    the app doesn't crash if there are issues.
    """
    try:
        _render_performance_tracker()
    except Exception as e:
        logger.exception("Error in Performance Tracker page")
        st.error(
            f"An error occurred in the Performance Tracker: {str(e)}\n\n"
            "The rest of the application should continue to work normally. "
            "Please check the logs for more details."
        )
        st.info(
            "If this error persists, please ensure:\n"
            "1. Database tables are migrated (run `python scripts/migrate_tracking_tables.py`)\n"
            "2. S3 credentials are configured for asm-models data access\n"
            "3. At least one portfolio has been registered and tracked"
        )


def _render_performance_tracker():
    """Internal function that renders the Performance Tracker page."""
    # Import tracking components (lazy import to avoid startup issues)
    from ..tracking import (
        PortfolioTracker,
        get_tracker,
        Variants,
        PeriodTypes,
    )
    from .styles import (
        get_dashboard_css,
        COLORS,
        VARIANT_COLORS,
        get_variant_display_name,
        get_period_display_name,
        format_percentage,
        format_ratio,
    )
    from .components import (
        render_kpi_grid,
        render_comparison_table,
        render_section_header,
        format_metrics_for_display,
    )
    from .charts import (
        create_nav_chart,
        create_drawdown_chart,
        create_allocation_chart,
        create_returns_bar_chart,
    )

    # Inject custom CSS with reduced header spacing
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)
    st.markdown("""
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        header[data-testid="stHeader"] {
            height: 2.5rem;
        }
        h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Page header
    st.title("Portfolio Performance Tracker")

    # Initialize tracker
    tracker = get_tracker()

    # Get list of portfolios
    portfolios = tracker.list_portfolios(active_only=True)

    if not portfolios:
        st.warning(
            "No portfolios are currently being tracked. "
            "Register a portfolio to start tracking performance."
        )
        _render_demo_mode()
        return

    # Sidebar controls
    st.sidebar.header("Portfolio Selection")

    # Helper to clean portfolio name for display
    def clean_display_name(name: str) -> str:
        return name.replace("_EqualWeight", "").replace("_", " ")

    # Portfolio selector - sorted alphabetically, display clean names
    sorted_portfolios = sorted(portfolios, key=lambda x: x.name)
    portfolio_id_map = {p.name: p.id for p in sorted_portfolios}
    display_to_actual = {clean_display_name(p.name): p.name for p in sorted_portfolios}

    selected_display_name = st.sidebar.selectbox(
        "Select Portfolio",
        options=list(display_to_actual.keys()),
        index=0,
    )
    selected_portfolio_name = display_to_actual[selected_display_name]
    selected_portfolio_id = portfolio_id_map[selected_portfolio_name]

    # Get portfolio details
    portfolio = tracker.get_portfolio(selected_portfolio_id)
    if portfolio:
        st.sidebar.markdown(f"**Source:** {portfolio.source or 'N/A'}")
        st.sidebar.markdown(f"**Tracking since:** {portfolio.start_date}")

    # Date range selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range")

    # Get available date range from NAV data
    raw_nav_df = tracker.get_nav_series(selected_portfolio_id, Variants.RAW)
    if raw_nav_df.empty:
        st.warning(f"No NAV data found for portfolio '{selected_portfolio_name}'.")
        return

    min_date = raw_nav_df.index.min().date()
    max_date = raw_nav_df.index.max().date()

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input(
        "Start",
        value=max(min_date, max_date - timedelta(days=365)),
        min_value=min_date,
        max_value=max_date,
    )
    end_date = col2.date_input(
        "End",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    # Variant selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("Variants to Compare")

    available_variants = Variants.all()
    selected_variants = st.sidebar.multiselect(
        "Select variants",
        options=available_variants,
        default=available_variants,
        format_func=get_variant_display_name,
    )

    if not selected_variants:
        st.warning("Please select at least one variant to display.")
        return

    # Load NAV data for selected variants
    nav_data = {}
    for variant in selected_variants:
        nav_df = tracker.get_nav_series(
            selected_portfolio_id, variant, start_date, end_date
        )
        if not nav_df.empty:
            nav_data[variant] = nav_df["nav"]

    if not nav_data:
        st.warning("No NAV data available for the selected date range and variants.")
        return

    # Main content area
    tabs = st.tabs([
        "Overview",
        "Performance Comparison",
        "Multi-Portfolio Compare",
        "Allocation History",
        "Signal Analysis",
        "Scraper View",
    ])

    # ===== Tab 1: Overview =====
    with tabs[0]:
        _render_overview_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date, nav_data
        )

    # ===== Tab 2: Performance Comparison =====
    with tabs[1]:
        _render_comparison_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date
        )

    # ===== Tab 3: Multi-Portfolio Comparison =====
    with tabs[2]:
        _render_multi_portfolio_comparison_tab(tracker, start_date, end_date)

    # ===== Tab 4: Allocation History =====
    with tabs[3]:
        _render_allocation_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date
        )

    # ===== Tab 5: Signal Analysis =====
    with tabs[4]:
        _render_signals_tab(tracker, start_date, end_date)

    # ===== Tab 6: Scraper View =====
    with tabs[5]:
        _render_scraper_view_tab(tracker, start_date, end_date)


def _render_overview_tab(
    tracker, portfolio_id, variants, start_date, end_date, nav_data
):
    """Render the Overview tab."""
    from .components import render_kpi_grid, format_metrics_for_display
    from .charts import create_nav_chart, create_drawdown_chart
    from ..tracking.metrics import calculate_drawdown_series

    st.markdown("### Portfolio Performance Overview")

    # KPI cards for the "best" variant (or first selected)
    primary_variant = variants[0]
    perf = tracker.get_portfolio_performance(
        portfolio_id, primary_variant, start_date, end_date
    )

    kpis = format_metrics_for_display(perf)
    st.markdown(render_kpi_grid(kpis), unsafe_allow_html=True)

    # NAV Chart
    st.markdown("---")
    st.markdown("### NAV Comparison")

    nav_chart = create_nav_chart(
        nav_data,
        title="Normalized NAV (Base = 100)",
        normalize=True,
        height=450,
    )
    st.plotly_chart(nav_chart, use_container_width=True)

    # Drawdown Chart
    st.markdown("---")
    st.markdown("### Drawdown Analysis")

    drawdown_data = {}
    for variant, nav_series in nav_data.items():
        drawdown_data[variant] = calculate_drawdown_series(nav_series)

    dd_chart = create_drawdown_chart(
        drawdown_data,
        title="Drawdown (%)",
        height=300,
    )
    st.plotly_chart(dd_chart, use_container_width=True)


def _render_comparison_tab(tracker, portfolio_id, variants, start_date, end_date):
    """Render the Performance Comparison tab."""
    from .components import render_comparison_table
    from .charts import create_returns_bar_chart, create_comparison_chart
    from ..tracking import PeriodTypes

    st.markdown("### Variant Comparison")

    # Get metrics for all variants
    comparison_df = tracker.compare_variants(portfolio_id, start_date, end_date)

    if comparison_df.empty:
        st.info("No comparison data available.")
        return

    # Filter to selected variants
    comparison_df = comparison_df[[v for v in variants if v in comparison_df.columns]]

    # Render comparison table
    st.markdown(
        render_comparison_table(comparison_df.T),  # Transpose so variants are columns
        unsafe_allow_html=True,
    )

    # Returns bar chart
    st.markdown("---")
    st.markdown("### Total Return Comparison")

    if "total_return" in comparison_df.index:
        returns_data = comparison_df.loc["total_return"].to_dict()
        returns_chart = create_returns_bar_chart(
            returns_data,
            title="Total Return by Variant",
            height=350,
        )
        st.plotly_chart(returns_chart, use_container_width=True)

    # Period-based comparison
    st.markdown("---")
    st.markdown("### Performance by Period")

    period_options = ["week", "month", "quarter", "year", "ytd"]
    selected_period = st.selectbox(
        "Select period for detailed comparison",
        options=period_options,
        index=1,  # Default to month
        format_func=lambda x: {
            "week": "1 Week",
            "month": "1 Month",
            "quarter": "3 Months",
            "year": "1 Year",
            "ytd": "Year to Date",
        }.get(x, x),
    )

    # Get stored metrics for the selected period
    metrics = tracker.get_stored_metrics(
        portfolio_id,
        period_type=selected_period,
    )

    if metrics:
        # Build dataframe for display
        period_data = []
        for m in metrics:
            period_data.append({
                "variant": m.variant,
                "total_return": float(m.total_return) if m.total_return else 0,
                "sharpe_ratio": float(m.sharpe_ratio) if m.sharpe_ratio else 0,
                "max_drawdown": float(m.max_drawdown) if m.max_drawdown else 0,
            })

        if period_data:
            period_df = pd.DataFrame(period_data)
            st.dataframe(
                period_df.style.format({
                    "total_return": "{:.2%}",
                    "sharpe_ratio": "{:.2f}",
                    "max_drawdown": "{:.2%}",
                }),
                use_container_width=True,
            )


def _render_allocation_tab(tracker, portfolio_id, variants, start_date, end_date):
    """Render the Allocation History tab."""
    from .charts import create_allocation_chart
    from ..tracking import Variants

    st.markdown("### Equity Allocation Over Time")

    # Only overlay variants have allocation data
    overlay_variants = [v for v in variants if v != Variants.RAW]

    if not overlay_variants:
        st.info(
            "Select an overlay variant (Conservative or Trend Regime V2) "
            "to see allocation history."
        )
        return

    for variant in overlay_variants:
        nav_df = tracker.get_nav_series(portfolio_id, variant, start_date, end_date)

        if nav_df.empty or "equity_allocation" not in nav_df.columns:
            continue

        st.markdown(f"#### {variant.replace('_', ' ').title()}")

        alloc_chart = create_allocation_chart(
            nav_df["equity_allocation"],
            title=f"Equity Allocation - {variant.replace('_', ' ').title()}",
            height=250,
        )
        st.plotly_chart(alloc_chart, use_container_width=True)

        # Show current allocation
        current_alloc = nav_df["equity_allocation"].iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("Current Equity", f"{current_alloc:.1%}")
        col2.metric("Current Cash", f"{1 - current_alloc:.1%}")

        st.markdown("---")


def _render_signals_tab(tracker, start_date, end_date):
    """Render the Signal Analysis tab."""
    from ..tracking import OVERLAY_REGISTRY
    from .styles import COLORS

    st.markdown("### Overlay Signal Breakdown")

    model_options = list(OVERLAY_REGISTRY.keys())
    selected_model = st.selectbox(
        "Select overlay model",
        options=model_options,
        format_func=lambda x: OVERLAY_REGISTRY[x].display_name,
    )

    signals_df = tracker.get_overlay_signals(selected_model, start_date, end_date)

    if signals_df.empty:
        st.info(f"No signal data available for {selected_model}.")
        return

    # Allocation chart
    if "actual_allocation" in signals_df.columns:
        st.markdown("#### Target Allocation Over Time")

        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=signals_df.index,
            y=signals_df["actual_allocation"] * 100,
            name="Actual Allocation",
            mode="lines",
            fill="tozeroy",
            line={"color": COLORS["primary"], "width": 2},
        ))

        if "target_allocation" in signals_df.columns:
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df["target_allocation"] * 100,
                name="Target Allocation",
                mode="lines",
                line={"color": COLORS["chart_2"], "width": 1.5, "dash": "dot"},
            ))

        fig.update_layout(
            height=300,
            yaxis_title="Allocation (%)",
            yaxis_ticksuffix="%",
            yaxis_range=[0, 105],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Signal values table
    st.markdown("#### Signal Values (Latest)")

    # Get signal columns
    signal_cols = [c for c in signals_df.columns if c.startswith("signal_")]
    impact_cols = [c for c in signals_df.columns if c.startswith("impact_")]

    if signal_cols:
        latest = signals_df.iloc[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Signals**")
            signal_data = {
                c.replace("signal_", ""): latest[c]
                for c in signal_cols
                if pd.notna(latest[c])
            }
            st.dataframe(pd.Series(signal_data).to_frame("Value"))

        with col2:
            st.markdown("**Impacts**")
            impact_data = {
                c.replace("impact_", ""): latest[c]
                for c in impact_cols
                if pd.notna(latest[c])
            }
            st.dataframe(pd.Series(impact_data).to_frame("Impact"))

    # Trade history
    st.markdown("---")
    st.markdown("#### Trade History")

    if "trade_required" in signals_df.columns:
        trades = signals_df[signals_df["trade_required"] == True]
        if not trades.empty:
            trade_df = trades[["actual_allocation", "target_allocation"]].copy()
            trade_df["allocation_change"] = trade_df["actual_allocation"].diff()
            st.dataframe(
                trade_df.style.format({
                    "actual_allocation": "{:.1%}",
                    "target_allocation": "{:.1%}",
                    "allocation_change": "{:+.1%}",
                }),
                use_container_width=True,
            )
            st.info(f"Total trades in period: {len(trades)}")
        else:
            st.info("No trades executed in the selected period.")


def _render_multi_portfolio_comparison_tab(tracker, sidebar_start_date, sidebar_end_date):
    """Render the Multi-Portfolio Comparison tab with expandable month/day returns."""
    from ..tracking import Variants
    from .styles import format_percentage, get_variant_display_name, VARIANT_COLORS, COLORS
    from datetime import date, timedelta
    import pandas as pd

    st.markdown("### Multi-Portfolio Returns Comparison")
    st.markdown("Compare returns across multiple portfolios with expandable month/day details.")

    # Get all portfolios
    all_portfolios = tracker.list_portfolios(active_only=True)

    if not all_portfolios:
        st.warning("No portfolios available.")
        return

    # Sort portfolios alphabetically
    all_portfolios_sorted = sorted(all_portfolios, key=lambda x: x.name)
    portfolio_options = {p.name: p.id for p in all_portfolios_sorted}

    # Multi-select for portfolios and variants side by side
    col1, col2 = st.columns(2)

    with col1:
        selected_portfolio_names = st.multiselect(
            "Select Portfolios",
            options=list(portfolio_options.keys()),
            default=list(portfolio_options.keys())[:3] if len(portfolio_options) >= 3 else list(portfolio_options.keys()),
            help="Select portfolios to compare",
        )

    with col2:
        variant_options = Variants.all()
        selected_variants = st.multiselect(
            "Select Variants",
            options=variant_options,
            default=[Variants.RAW],  # Default to raw only
            format_func=get_variant_display_name,
            help="Select one or more variants to compare",
        )

    if not selected_portfolio_names:
        st.info("Please select at least one portfolio.")
        return

    if not selected_variants:
        st.info("Please select at least one variant.")
        return

    # Find the date range available across all selected portfolios
    # This ensures the date picker allows selecting dates that have data
    all_min_dates = []
    all_max_dates = []

    for portfolio_name in selected_portfolio_names:
        portfolio_id = portfolio_options[portfolio_name]
        for variant in selected_variants:
            nav_df = tracker.get_nav_series(portfolio_id, variant)
            if not nav_df.empty:
                all_min_dates.append(nav_df.index.min().date())
                all_max_dates.append(nav_df.index.max().date())

    if not all_min_dates or not all_max_dates:
        st.warning("No NAV data available for the selected portfolios and variants.")
        return

    # Use the earliest min date and latest max date across all selected portfolios
    available_min_date = min(all_min_dates)
    available_max_date = max(all_max_dates)
    today = date.today()
    current_month = today.strftime("%Y-%m")

    # Independent date range selector for this tab
    st.markdown("---")
    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "Start Date",
            value=max(available_min_date, available_max_date - timedelta(days=365)),
            min_value=available_min_date,
            max_value=available_max_date,
            key="multi_portfolio_start_date",
        )

    with date_col2:
        end_date = st.date_input(
            "End Date",
            value=available_max_date,
            min_value=available_min_date,
            max_value=available_max_date,
            key="multi_portfolio_end_date",
        )

    # Helper to clean portfolio name (remove _EqualWeight suffix)
    def clean_name(name: str) -> str:
        return name.replace("_EqualWeight", "").replace("_", " ")

    # Variant abbreviations for compact display
    variant_abbrev = {
        "raw": "Raw",
        "conservative": "Cons.",
        "trend_regime_v2": "Trend",
    }

    # Build returns data organized by portfolio -> variant
    # Structure: {portfolio_name: {variant: {month: {total, days}}}}
    portfolio_data = {}

    for portfolio_name in selected_portfolio_names:
        portfolio_id = portfolio_options[portfolio_name]
        portfolio_data[portfolio_name] = {}

        for variant in selected_variants:
            nav_df = tracker.get_nav_series(portfolio_id, variant, start_date, end_date)

            if nav_df.empty:
                continue

            # Use pre-computed daily_return from database (accounts for overlay allocation)
            # This is important for overlay variants where return = raw_return * allocation
            if "daily_return" in nav_df.columns:
                returns = nav_df["daily_return"]
            else:
                # Fallback to pct_change if daily_return not available
                returns = nav_df["nav"].pct_change()

            # Group by month
            monthly_data = {}
            for date_idx, ret in returns.items():
                if pd.isna(ret):
                    continue
                month_key = date_idx.strftime("%Y-%m")
                if month_key not in monthly_data:
                    monthly_data[month_key] = {"days": [], "total": 0}
                monthly_data[month_key]["days"].append({
                    "date": date_idx.strftime("%Y-%m-%d"),
                    "return": ret,
                })

            # Calculate monthly totals using compounding
            for month_key in monthly_data:
                daily_returns = [d["return"] for d in monthly_data[month_key]["days"]]
                monthly_total = 1
                for r in daily_returns:
                    monthly_total *= (1 + r)
                monthly_data[month_key]["total"] = monthly_total - 1

            portfolio_data[portfolio_name][variant] = monthly_data

    # Filter out portfolios with no data
    portfolio_data = {k: v for k, v in portfolio_data.items() if v}

    if not portfolio_data:
        st.warning("No NAV data available for the selected portfolios and variants.")
        return

    # Get all months across all portfolios
    all_months = set()
    for port_variants in portfolio_data.values():
        for variant_data in port_variants.values():
            all_months.update(variant_data.keys())
    all_months = sorted(all_months, reverse=True)

    st.markdown("---")

    # Build DataFrame for display - grouped by portfolio with variant sub-columns
    portfolios_with_data = [p for p in selected_portfolio_names if p in portfolio_data]

    # Helper function to color values
    def color_value(val):
        if val is None or val == "-" or val == "•":
            return "color: #666"
        try:
            num = float(str(val).replace("%", ""))
            if num > 0:
                return "color: #22c55e; font-weight: 600"
            elif num < 0:
                return "color: #ef4444; font-weight: 600"
        except (ValueError, TypeError):
            pass
        return ""

    # Build monthly returns table
    monthly_table_data = []
    for month in all_months:
        row = {"Month": f"{month} (MTD)" if month == current_month else month}

        for portfolio_name in portfolios_with_data:
            port_variants = portfolio_data[portfolio_name]
            clean_pname = clean_name(portfolio_name)

            for variant in selected_variants:
                if variant not in port_variants:
                    continue
                col_name = f"{clean_pname} {variant_abbrev.get(variant, variant)}"
                if month in port_variants[variant]:
                    val = port_variants[variant][month]["total"] * 100
                    row[col_name] = f"{val:.2f}%"
                else:
                    row[col_name] = "•"

        monthly_table_data.append(row)

    if not monthly_table_data:
        st.info("No data to display.")
        return

    monthly_df = pd.DataFrame(monthly_table_data)

    # Style the monthly dataframe
    value_cols = [c for c in monthly_df.columns if c != "Month"]
    styled_monthly = monthly_df.style.applymap(
        color_value, subset=value_cols
    ).set_properties(**{"text-align": "center"}, subset=value_cols)

    st.markdown("##### Monthly Returns")
    st.dataframe(styled_monthly, use_container_width=True, hide_index=True, height=min(400, 35 * len(monthly_df) + 38))

    # Daily details section
    st.markdown("---")
    st.markdown("##### Daily Details")

    selected_month = st.selectbox(
        "Select month:",
        options=all_months,
        format_func=lambda x: f"{x} (MTD)" if x == current_month else x,
        key="daily_details_month",
    )

    if selected_month:
        # Get all days for this month across all portfolios
        all_days = set()
        for port_variants in portfolio_data.values():
            for variant_data in port_variants.values():
                if selected_month in variant_data:
                    for d in variant_data[selected_month]["days"]:
                        all_days.add(d["date"])
        all_days = sorted(all_days)

        if all_days:
            daily_table_data = []
            for day in all_days:
                row = {"Date": day[-5:]}  # Show only MM-DD
                for portfolio_name in portfolios_with_data:
                    clean_pname = clean_name(portfolio_name)
                    for variant in selected_variants:
                        if variant not in portfolio_data[portfolio_name]:
                            continue
                        col_name = f"{clean_pname} {variant_abbrev.get(variant, variant)}"
                        if selected_month in portfolio_data[portfolio_name][variant]:
                            day_ret = next(
                                (d["return"] for d in portfolio_data[portfolio_name][variant][selected_month]["days"] if d["date"] == day),
                                None,
                            )
                            if day_ret is not None:
                                row[col_name] = f"{day_ret*100:.2f}%"
                            else:
                                row[col_name] = "-"
                        else:
                            row[col_name] = "-"
                daily_table_data.append(row)

            if daily_table_data:
                daily_df = pd.DataFrame(daily_table_data)
                value_cols = [c for c in daily_df.columns if c != "Date"]
                styled_daily = daily_df.style.applymap(
                    color_value, subset=value_cols
                ).set_properties(**{"text-align": "center"}, subset=value_cols)
                # Show full month without scrollbar (up to 31 days)
                st.dataframe(styled_daily, use_container_width=True, hide_index=True, height=35 * len(daily_df) + 38)
        else:
            st.info("No daily data available for this month.")


def _render_scraper_view_tab(tracker, sidebar_start_date, sidebar_end_date):
    """Render the Scraper View tab - pivot table of daily returns across all portfolios."""
    from ..tracking import Variants
    from .styles import format_percentage
    from datetime import date, timedelta
    import pandas as pd
    import numpy as np

    st.markdown("### Daily Returns - Scraper View")
    st.markdown("Daily performance percentage for all portfolios (similar to Google Sheets view).")

    # Get all portfolios
    all_portfolios = tracker.list_portfolios(active_only=True)

    if not all_portfolios:
        st.warning("No portfolios available.")
        return

    # Sort portfolios alphabetically
    all_portfolios_sorted = sorted(all_portfolios, key=lambda x: x.name)

    # Helper to clean portfolio name
    def clean_name(name: str) -> str:
        return name.replace("_EqualWeight", "").replace("_", " ")

    # Default portfolio filter keywords
    default_keywords = ["SA_LargeCaps", "SA_MidCaps", "SPY", "TR10_LargeCapsX", "TR10", "TopWeights", "TW30"]

    # Determine default selected portfolios
    def matches_default(name: str) -> bool:
        for keyword in default_keywords:
            if keyword.lower() in name.lower():
                return True
        return False

    default_selected = [p.name for p in all_portfolios_sorted if matches_default(p.name)]
    all_portfolio_names = [p.name for p in all_portfolios_sorted]

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        selected_variant = st.selectbox(
            "Select Variant",
            options=Variants.all(),
            index=0,  # Default to RAW
            key="scraper_view_variant",
        )

    with col2:
        # Default to last 30 days
        default_start = max(sidebar_start_date, sidebar_end_date - timedelta(days=30))
        date_range = st.date_input(
            "Date Range",
            value=(default_start, sidebar_end_date),
            key="scraper_view_dates",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_start, sidebar_end_date

    # Portfolio filter
    selected_portfolios = st.multiselect(
        "Select Portfolios",
        options=all_portfolio_names,
        default=default_selected if default_selected else all_portfolio_names[:5],
        key="scraper_view_portfolios",
    )

    if not selected_portfolios:
        st.warning("Please select at least one portfolio.")
        return

    # Filter to selected portfolios
    filtered_portfolios = [p for p in all_portfolios_sorted if p.name in selected_portfolios]

    # Collect daily returns for selected portfolios
    returns_data = {}

    for portfolio in filtered_portfolios:
        nav_df = tracker.get_nav_series(portfolio.id, selected_variant, start_date, end_date)

        if nav_df.empty:
            continue

        # Use pre-computed daily_return from database
        if "daily_return" in nav_df.columns:
            returns = nav_df["daily_return"]
        else:
            returns = nav_df["nav"].pct_change()

        # Store returns with clean name
        clean_portfolio_name = clean_name(portfolio.name)
        returns_data[clean_portfolio_name] = returns

    if not returns_data:
        st.warning("No data available for the selected date range and variant.")
        return

    # Create DataFrame with portfolios as rows, dates as columns
    df = pd.DataFrame(returns_data).T

    # Sort columns (dates) chronologically
    df = df.reindex(sorted(df.columns), axis=1)

    # Add total row (sum of all portfolios)
    total_row = df.sum(axis=0)
    df.loc["Gesamtsumme"] = total_row

    # Format column headers as short dates (e.g., "Nov. 03")
    date_labels = {}
    for col in df.columns:
        if hasattr(col, 'strftime'):
            date_labels[col] = col.strftime("%b %d")
        else:
            date_labels[col] = str(col)

    df.columns = [date_labels.get(c, str(c)) for c in df.columns]

    # Convert to percentages for display
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""
        )

    # Reset index to make portfolio name a column
    display_df = display_df.reset_index()
    display_df.rename(columns={"index": "Portfolio"}, inplace=True)

    # Style function for color coding
    def color_returns(val):
        """Color cells based on return value - green for positive, red for negative."""
        if val == "" or val == "Portfolio":
            return ""
        try:
            num = float(val.replace("%", ""))
            if num > 5:
                return "background-color: #1e7b1e; color: white"  # Dark green
            elif num > 2:
                return "background-color: #28a745; color: white"  # Green
            elif num > 0:
                return "background-color: #90EE90; color: black"  # Light green
            elif num == 0:
                return "background-color: #f8f9fa; color: black"  # Neutral gray
            elif num > -2:
                return "background-color: #ffcccb; color: black"  # Light red
            elif num > -5:
                return "background-color: #dc3545; color: white"  # Red
            else:
                return "background-color: #8b0000; color: white"  # Dark red
        except (ValueError, AttributeError):
            return ""

    # Apply styling
    value_cols = [c for c in display_df.columns if c != "Portfolio"]
    styled_df = display_df.style.map(
        color_returns, subset=value_cols
    ).set_properties(
        **{"text-align": "center"}, subset=value_cols
    ).set_properties(
        **{"text-align": "left", "font-weight": "bold"}, subset=["Portfolio"]
    )

    # Make the last row (Gesamtsumme) bold
    styled_df = styled_df.apply(
        lambda x: ["font-weight: bold" if x.name == len(display_df) - 1 else "" for _ in x],
        axis=1
    )

    # Calculate height based on number of rows (no cap to avoid scrollbar)
    height = 35 * len(display_df) + 40

    # Display the table
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=height,
    )

    # Show summary stats
    st.markdown("---")
    st.markdown("#### Period Summary")

    # Calculate period totals using compounding
    summary_data = []
    for portfolio_name, returns in returns_data.items():
        valid_returns = returns.dropna()
        if len(valid_returns) > 0:
            # Compounded return
            total_return = (1 + valid_returns).prod() - 1
            avg_daily = valid_returns.mean()
            summary_data.append({
                "Portfolio": portfolio_name,
                "Total Return": f"{total_return*100:.2f}%",
                "Avg Daily": f"{avg_daily*100:.3f}%",
                "Trading Days": len(valid_returns),
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("Total Return", ascending=False, key=lambda x: x.str.replace("%", "").astype(float))
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _render_demo_mode():
    """Render demo content when no portfolios are available."""
    st.markdown("---")
    st.markdown("### Demo Mode")
    st.info(
        "To start tracking portfolios:\n\n"
        "1. Run the migration script: `python scripts/migrate_tracking_tables.py`\n"
        "2. Register a portfolio using the PortfolioTracker API\n"
        "3. Run the scheduled NAV update script: `python scripts/scheduled_nav_update.py`\n\n"
        "Example registration:\n"
        "```python\n"
        "from AlphaMachine_core.tracking import PortfolioTracker\n\n"
        "tracker = PortfolioTracker()\n"
        "portfolio = tracker.register_portfolio(\n"
        "    name='TopWeights_20_MVO',\n"
        "    config={'num_stocks': 20, 'optimizer': 'mvo'},\n"
        "    source='Topweights',\n"
        "    start_date=date(2024, 1, 1),\n"
        ")\n"
        "```"
    )
