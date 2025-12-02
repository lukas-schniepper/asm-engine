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
        create_monthly_returns_heatmap,
    )

    # Inject custom CSS
    st.markdown(get_dashboard_css(), unsafe_allow_html=True)

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

    # Portfolio selector
    portfolio_options = {p.name: p.id for p in portfolios}
    selected_portfolio_name = st.sidebar.selectbox(
        "Select Portfolio",
        options=list(portfolio_options.keys()),
        index=0,
    )
    selected_portfolio_id = portfolio_options[selected_portfolio_name]

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
        "Allocation History",
        "Signal Analysis",
        "Monthly Returns",
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

    # ===== Tab 3: Allocation History =====
    with tabs[2]:
        _render_allocation_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date
        )

    # ===== Tab 4: Signal Analysis =====
    with tabs[3]:
        _render_signals_tab(tracker, start_date, end_date)

    # ===== Tab 5: Monthly Returns =====
    with tabs[4]:
        _render_monthly_returns_tab(nav_data)


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


def _render_monthly_returns_tab(nav_data):
    """Render the Monthly Returns tab."""
    from .charts import create_monthly_returns_heatmap
    from .styles import get_variant_display_name

    st.markdown("### Monthly Returns Heatmap")

    # Create heatmap for each variant
    for variant, nav_series in nav_data.items():
        st.markdown(f"#### {get_variant_display_name(variant)}")

        try:
            heatmap = create_monthly_returns_heatmap(
                nav_series,
                title="",
                height=350,
            )
            st.plotly_chart(heatmap, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate heatmap for {variant}: {e}")

        st.markdown("---")


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
