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

    # Filter to only portfolios with NAV data
    sorted_portfolios = sorted(portfolios, key=lambda x: x.name)
    portfolios_with_data = []
    for p in sorted_portfolios:
        nav_df = tracker.get_nav_series(p.id, Variants.RAW)
        if not nav_df.empty:
            portfolios_with_data.append(p)

    if not portfolios_with_data:
        st.warning(
            "No portfolios have NAV data yet. "
            "Run NAV update to generate performance data."
        )
        return

    # Portfolio selector - sorted alphabetically, display clean names
    portfolio_id_map = {p.name: p.id for p in portfolios_with_data}
    display_to_actual = {clean_display_name(p.name): p.name for p in portfolios_with_data}

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
        "Risk Analytics",
        "Performance Comparison",
        "Benchmark Comparison",
        "Multi-Portfolio Compare",
        "Drawdown Analysis",
        "Allocation History",
        "Signal Analysis",
        "Scraper View",
        "eToro Compare",
    ])

    # ===== Tab 1: Overview =====
    with tabs[0]:
        _render_overview_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date, nav_data
        )

    # ===== Tab 2: Risk Analytics (NEW) =====
    with tabs[1]:
        _render_risk_analytics_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date, nav_data
        )

    # ===== Tab 3: Performance Comparison =====
    with tabs[2]:
        _render_comparison_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date
        )

    # ===== Tab 4: Benchmark Comparison =====
    with tabs[3]:
        _render_benchmark_comparison_tab(
            tracker, selected_portfolio_id, portfolio, selected_variants,
            start_date, end_date
        )

    # ===== Tab 5: Multi-Portfolio Comparison =====
    with tabs[4]:
        _render_multi_portfolio_comparison_tab(tracker, start_date, end_date)

    # ===== Tab 6: Drawdown Analysis (NEW) =====
    with tabs[5]:
        _render_drawdown_analysis_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date, nav_data
        )

    # ===== Tab 7: Allocation History =====
    with tabs[6]:
        _render_allocation_tab(
            tracker, selected_portfolio_id, selected_variants,
            start_date, end_date
        )

    # ===== Tab 8: Signal Analysis =====
    with tabs[7]:
        _render_signals_tab(tracker, start_date, end_date)

    # ===== Tab 9: Scraper View =====
    with tabs[8]:
        _render_scraper_view_tab(tracker, start_date, end_date)

    # ===== Tab 10: eToro Compare =====
    with tabs[9]:
        _render_etoro_compare_tab()


def _render_overview_tab(
    tracker, portfolio_id, variants, start_date, end_date, nav_data
):
    """Render the Overview tab."""
    from .components import render_kpi_grid, format_metrics_for_display
    from .charts import create_nav_chart, create_drawdown_chart
    from ..tracking.metrics import (
        calculate_drawdown_series,
        calculate_returns,
        calculate_var,
        calculate_cvar,
        calculate_beta,
        calculate_alpha,
        calculate_information_ratio,
    )
    from ..tracking.benchmark_adapter import get_benchmark_adapter

    st.markdown("### Portfolio Performance Overview")

    # KPI cards for the "best" variant (or first selected)
    primary_variant = variants[0]
    perf = tracker.get_portfolio_performance(
        portfolio_id, primary_variant, start_date, end_date
    )

    # Initialize risk metrics (calculated below if enough data)
    beta = None
    alpha = None
    info_ratio = None
    var_95 = None
    cvar_95 = None

    if primary_variant in nav_data:
        portfolio_nav = nav_data[primary_variant]
        portfolio_returns = calculate_returns(portfolio_nav)

        # Get SPY benchmark data
        try:
            benchmark_adapter = get_benchmark_adapter()
            benchmark_nav = benchmark_adapter.get_benchmark_nav(
                "SPY", start_date, end_date, normalize=False
            )
            benchmark_returns = benchmark_adapter.get_benchmark_returns(
                "SPY", start_date, end_date
            )

            # Align returns
            aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
            if len(aligned) >= 30:
                aligned_port = aligned.iloc[:, 0]
                aligned_bench = aligned.iloc[:, 1]

                # Calculate metrics
                beta = calculate_beta(aligned_port, aligned_bench)
                alpha = calculate_alpha(portfolio_nav, benchmark_nav)
                info_ratio = calculate_information_ratio(portfolio_nav, benchmark_nav)
                var_95 = calculate_var(aligned_port, 0.95)
                cvar_95 = calculate_cvar(aligned_port, 0.95)
        except Exception:
            pass  # Will show N/A in KPI grid

    # Add info ratio to perf dict for KPI grid (always show, even if N/A)
    perf["information_ratio"] = info_ratio

    kpis = format_metrics_for_display(perf)
    st.markdown(render_kpi_grid(kpis), unsafe_allow_html=True)

    # Second row: Institutional Risk Metrics
    st.markdown("#### Risk Metrics (vs SPY)")

    if beta is not None:
        # Display second row of KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Beta", f"{beta:.2f}")
        with col2:
            st.metric("Alpha (Ann.)", f"{alpha*100:+.2f}%")
        with col3:
            st.metric("Info Ratio", f"{info_ratio:.2f}")
        with col4:
            st.metric("VaR (95%)", f"{var_95*100:.2f}%")
        with col5:
            st.metric("CVaR (95%)", f"{cvar_95*100:.2f}%")
    elif primary_variant in nav_data:
        st.caption("Insufficient data for risk metrics (need 30+ days).")
    else:
        st.caption("No NAV data available for risk metrics.")

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
    """Render the Performance Comparison tab with institutional metrics."""
    from .components import render_comparison_table
    from .charts import create_returns_bar_chart, create_comparison_chart
    from ..tracking import PeriodTypes
    from ..tracking.metrics import (
        calculate_returns,
        calculate_var,
        calculate_cvar,
        calculate_beta,
        calculate_alpha,
        calculate_tracking_error,
        calculate_information_ratio,
        calculate_institutional_metrics,
    )
    from ..tracking.benchmark_adapter import get_benchmark_adapter

    st.markdown("### Variant Comparison")

    # Get basic metrics for all variants
    comparison_df = tracker.compare_variants(portfolio_id, start_date, end_date)

    if comparison_df.empty:
        st.info("No comparison data available.")
        return

    # Filter to selected variants
    comparison_df = comparison_df[[v for v in variants if v in comparison_df.columns]]

    # Add institutional metrics section
    st.markdown("#### Core Performance Metrics")

    # Render basic comparison table
    st.markdown(
        render_comparison_table(comparison_df.T),  # Transpose so variants are columns
        unsafe_allow_html=True,
    )

    # ===== Enhanced Institutional Metrics Table =====
    st.markdown("---")
    st.markdown("#### Institutional Risk Metrics (vs SPY)")

    # Get NAV data for each variant and calculate institutional metrics
    try:
        benchmark_adapter = get_benchmark_adapter()
        benchmark_nav = benchmark_adapter.get_benchmark_nav(
            "SPY", start_date, end_date, normalize=False
        )
        benchmark_returns = benchmark_adapter.get_benchmark_returns(
            "SPY", start_date, end_date
        )

        if benchmark_nav.empty:
            st.caption("Could not load SPY benchmark data.")
        else:
            institutional_data = []

            for variant in variants:
                nav_df = tracker.get_nav_series(portfolio_id, variant, start_date, end_date)
                if nav_df.empty or "nav" not in nav_df.columns:
                    continue

                portfolio_nav = nav_df["nav"]
                portfolio_returns = calculate_returns(portfolio_nav)

                # Align with benchmark
                aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
                if len(aligned) < 30:
                    continue

                aligned_port = aligned.iloc[:, 0]
                aligned_bench = aligned.iloc[:, 1]

                # Calculate all institutional metrics
                inst_metrics = {
                    "Variant": variant.replace("_", " ").title(),
                    "VaR (95%)": f"{calculate_var(aligned_port, 0.95)*100:.2f}%",
                    "CVaR (95%)": f"{calculate_cvar(aligned_port, 0.95)*100:.2f}%",
                    "VaR (99%)": f"{calculate_var(aligned_port, 0.99)*100:.2f}%",
                    "CVaR (99%)": f"{calculate_cvar(aligned_port, 0.99)*100:.2f}%",
                    "Beta": f"{calculate_beta(aligned_port, aligned_bench):.2f}",
                    "Alpha (Ann.)": f"{calculate_alpha(portfolio_nav, benchmark_nav)*100:+.2f}%",
                    "Tracking Error": f"{calculate_tracking_error(aligned_port, aligned_bench)*100:.2f}%",
                    "Info Ratio": f"{calculate_information_ratio(portfolio_nav, benchmark_nav):.2f}",
                }
                institutional_data.append(inst_metrics)

            if institutional_data:
                inst_df = pd.DataFrame(institutional_data)

                # Style the dataframe
                def color_risk_metrics(val):
                    if isinstance(val, str) and "%" in val:
                        try:
                            num = float(val.replace("%", "").replace("+", ""))
                            # For VaR/CVaR (negative numbers are worse)
                            if num < -2:
                                return "color: #ef4444"  # Red for high risk
                            elif num > 0:
                                return "color: #22c55e"  # Green for positive
                        except:
                            pass
                    return ""

                styled_inst = inst_df.style.applymap(
                    color_risk_metrics,
                    subset=["VaR (95%)", "CVaR (95%)", "VaR (99%)", "CVaR (99%)", "Alpha (Ann.)"]
                ).set_properties(**{"text-align": "center"})

                st.dataframe(styled_inst, use_container_width=True, hide_index=True)
            else:
                st.caption("Insufficient data to calculate institutional metrics.")

    except Exception as e:
        st.caption(f"Could not calculate institutional metrics: {e}")

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


def _render_attribution_table(
    holdings: list,
    title: str,
    portfolio_tickers: list = None,
    is_universe_table: bool = False,
):
    """
    Render a styled attribution table for stock-level analysis.

    Args:
        holdings: List of holding dicts with ticker, weight, return, contribution
        title: Table title (for info messages)
        portfolio_tickers: List of tickers in the portfolio (for highlighting in universe table)
        is_universe_table: If True, sort portfolio tickers first and highlight non-portfolio tickers
    """
    if not holdings:
        st.info(f"No {title.lower()} data available")
        return

    df = pd.DataFrame(holdings)

    if is_universe_table and portfolio_tickers:
        # For universe table: sort portfolio tickers first (alphabetically), then others alphabetically
        portfolio_set = set(portfolio_tickers)
        df["in_portfolio"] = df["ticker"].isin(portfolio_set)

        # Sort: portfolio tickers first (alphabetically), then others alphabetically
        in_port = df[df["in_portfolio"]].copy().sort_values("ticker")
        not_in_port = df[~df["in_portfolio"]].copy().sort_values("ticker")
        df = pd.concat([in_port, not_in_port], ignore_index=True)
    else:
        # Sort alphabetically by ticker
        df = df.sort_values("ticker")

    # Store which tickers are not in portfolio (for highlighting)
    if is_universe_table and portfolio_tickers:
        not_in_portfolio_tickers = set(df["ticker"]) - set(portfolio_tickers)
    else:
        not_in_portfolio_tickers = set()

    # Add total row
    total_row = pd.DataFrame([{
        "ticker": "TOTAL",
        "weight": df["weight"].sum(),
        "return": None,
        "contribution": df["contribution"].sum(),
    }])

    # Clean up helper columns before concat
    if "in_portfolio" in df.columns:
        df = df.drop(columns=["in_portfolio"])

    df = pd.concat([df, total_row], ignore_index=True)

    # Format columns
    df["weight"] = df["weight"].apply(lambda x: f"{x*100:.1f}%" if x else "0.0%")
    df["return"] = df["return"].apply(
        lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "-"
    )
    df["contribution"] = df["contribution"].apply(
        lambda x: f"{x*100:+.3f}%"
    )

    df.columns = ["Ticker", "Weight", "Return", "Contribution"]

    # Color coding function for return/contribution values
    def color_values(val):
        if val == "-" or val == "TOTAL":
            return ""
        try:
            num = float(val.replace("%", "").replace("+", ""))
            if num > 0:
                return "color: #22c55e"
            elif num < 0:
                return "color: #ef4444"
        except:
            pass
        return ""

    # Row-level background styling for non-portfolio tickers
    def highlight_non_portfolio(row):
        ticker = row["Ticker"]
        if ticker in not_in_portfolio_tickers:
            return ["background-color: #e0f2fe"] * len(row)  # Light blue
        return [""] * len(row)

    styled = df.style.applymap(
        color_values,
        subset=["Return", "Contribution"]
    ).set_properties(
        **{"text-align": "right"},
        subset=["Weight", "Return", "Contribution"]
    )

    # Apply row highlighting for universe table
    if is_universe_table and not_in_portfolio_tickers:
        styled = styled.apply(highlight_non_portfolio, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(400, 35 * len(df) + 38)
    )


def _render_benchmark_comparison_tab(
    tracker, portfolio_id, portfolio, variants, start_date, end_date
):
    """Render the Benchmark Comparison tab - compare portfolio to EW universe."""
    from ..tracking import Variants
    from ..tracking.benchmark import (
        compare_portfolio_to_benchmark,
        calculate_stock_attribution,
        calculate_portfolio_monthly_returns_buyhold,
        calculate_benchmark_monthly_returns_buyhold,
        calculate_portfolio_monthly_returns_gips,
        calculate_benchmark_monthly_returns_gips,
    )
    from .styles import COLORS
    import plotly.graph_objects as go
    from datetime import date as date_type

    st.markdown("### Portfolio vs Equal-Weight Benchmark")
    st.markdown(
        "Compare your portfolio's performance against an equal-weight basket of all "
        "tickers in the source universe."
    )

    # Check if portfolio has a source
    if not portfolio or not portfolio.source:
        st.warning(
            "This portfolio doesn't have a source defined. "
            "Cannot calculate benchmark comparison."
        )
        return

    source = portfolio.source
    st.info(f"**Benchmark:** Equal-weight all tickers from source '{source}'")

    # Select which variant to compare
    variant_for_comparison = st.selectbox(
        "Select portfolio variant to compare",
        options=variants,
        index=0,  # Default to first variant (usually raw)
        format_func=lambda x: x.replace("_", " ").title(),
        key="benchmark_variant_select",
    )

    # Get portfolio NAV data
    nav_df = tracker.get_nav_series(
        portfolio_id, variant_for_comparison, start_date, end_date
    )

    if nav_df.empty:
        st.warning("No NAV data available for the selected date range.")
        return

    # Calculate comparison with loading indicator
    with st.spinner("Calculating benchmark returns..."):
        try:
            comparison = compare_portfolio_to_benchmark(
                portfolio_nav_df=nav_df,
                source=source,
                start_date=start_date,
                end_date=end_date,
                use_adjusted_close=True,
            )
        except Exception as e:
            st.error(f"Error calculating benchmark: {str(e)}")
            return

    if not comparison["benchmark_metrics"]:
        st.warning(
            f"No benchmark data available for source '{source}'. "
            "This may be because no ticker_period entries exist for this source."
        )
        return

    # ===== Summary Metrics Table =====
    st.markdown("---")
    st.markdown("### Performance Metrics")

    port_metrics = comparison["portfolio_metrics"]
    bench_metrics = comparison["benchmark_metrics"]

    # Build comparison table
    metrics_data = []
    metric_labels = {
        "total_return": ("Total Return", "{:.2%}"),
        "cagr": ("CAGR", "{:.2%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.2f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "calmar_ratio": ("Calmar Ratio", "{:.2f}"),
        "volatility": ("Volatility", "{:.2%}"),
        "win_rate": ("Win Rate", "{:.1%}"),
    }

    for metric_key, (label, fmt) in metric_labels.items():
        port_val = port_metrics.get(metric_key, 0)
        bench_val = bench_metrics.get(metric_key, 0)
        diff = port_val - bench_val if port_val is not None and bench_val is not None else None

        # Format for display
        port_str = fmt.format(port_val) if port_val is not None else "N/A"
        bench_str = fmt.format(bench_val) if bench_val is not None else "N/A"
        diff_str = fmt.format(diff) if diff is not None else "N/A"

        # Add sign prefix for difference
        if diff is not None and diff > 0:
            diff_str = "+" + diff_str

        metrics_data.append({
            "Metric": label,
            "Portfolio": port_str,
            "EW Benchmark": bench_str,
            "Difference": diff_str,
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Color the difference column
    def color_diff(val):
        if val == "N/A":
            return "color: #666"
        try:
            # Check if positive or negative
            if val.startswith("+"):
                return "color: #22c55e; font-weight: 600"
            elif val.startswith("-"):
                return "color: #ef4444; font-weight: 600"
        except:
            pass
        return ""

    styled_metrics = metrics_df.style.applymap(
        color_diff, subset=["Difference"]
    ).set_properties(**{"text-align": "center"}, subset=["Portfolio", "EW Benchmark", "Difference"])

    st.dataframe(styled_metrics, use_container_width=True, hide_index=True)

    # ===== Equity Curves Chart =====
    st.markdown("---")
    st.markdown("### Equity Curves")

    portfolio_nav = comparison["portfolio_nav"]
    benchmark_nav = comparison["benchmark_nav"]

    # Normalize to 100
    if not portfolio_nav.empty:
        portfolio_nav_normalized = (portfolio_nav / portfolio_nav.iloc[0]) * 100
    else:
        portfolio_nav_normalized = pd.Series(dtype=float)

    if not benchmark_nav.empty:
        benchmark_nav_normalized = (benchmark_nav / benchmark_nav.iloc[0]) * 100
    else:
        benchmark_nav_normalized = pd.Series(dtype=float)

    fig = go.Figure()

    # Portfolio line
    if not portfolio_nav_normalized.empty:
        fig.add_trace(go.Scatter(
            x=portfolio_nav_normalized.index,
            y=portfolio_nav_normalized.values,
            name=f"Portfolio ({variant_for_comparison.replace('_', ' ').title()})",
            mode="lines",
            line={"color": COLORS.get("primary", "#3b82f6"), "width": 2.5},
        ))

    # Benchmark line
    if not benchmark_nav_normalized.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_nav_normalized.index,
            y=benchmark_nav_normalized.values,
            name=f"EW Benchmark ({source})",
            mode="lines",
            line={"color": "#f97316", "width": 2, "dash": "dot"},
        ))

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="NAV (Normalized to 100)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===== Monthly Comparison Table =====
    st.markdown("---")
    st.markdown("### Monthly Returns Comparison")
    st.caption(
        "Returns calculated using GIPS-compliant methodology: compound daily returns. "
        "This matches the Scraper View and industry standards (GIPS/IBKR). "
        "Note: Stock attribution below uses buy-and-hold for per-stock analysis."
    )

    # Calculate monthly returns using GIPS methodology (compound daily returns)
    # This ensures consistency with the Scraper View and industry standards
    with st.spinner("Calculating monthly returns..."):
        portfolio_monthly_gips = calculate_portfolio_monthly_returns_gips(
            portfolio_id, start_date, end_date, tracker
        )
        benchmark_monthly_gips = calculate_benchmark_monthly_returns_gips(
            source, start_date, end_date, use_adjusted_close=True
        )

    # Build monthly_data from GIPS calculations
    all_months = sorted(set(portfolio_monthly_gips.keys()) | set(benchmark_monthly_gips.keys()))
    monthly_data = []
    for month in all_months:
        port_ret = portfolio_monthly_gips.get(month)
        bench_ret = benchmark_monthly_gips.get(month)
        diff = None
        if port_ret is not None and bench_ret is not None:
            diff = port_ret - bench_ret
        monthly_data.append({
            "month": month,
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "difference": diff,
        })

    if not monthly_data:
        st.info("No monthly comparison data available.")
    else:
        # Build monthly table sorted by month descending
        monthly_data_sorted = sorted(monthly_data, key=lambda x: x["month"], reverse=True)

        # Current month for MTD label
        current_month = date_type.today().strftime("%Y-%m")

        monthly_table = []
        cumulative_port = 1.0
        cumulative_bench = 1.0
        cumulative_diff = 0.0

        # Calculate cumulative from oldest to newest, then display newest first
        for record in sorted(monthly_data, key=lambda x: x["month"]):
            port_ret = record.get("portfolio_return")
            bench_ret = record.get("benchmark_return")
            diff = record.get("difference")

            if port_ret is not None:
                cumulative_port *= (1 + port_ret)
            if bench_ret is not None:
                cumulative_bench *= (1 + bench_ret)
            if diff is not None:
                cumulative_diff += diff

        # Now build display table
        for record in monthly_data_sorted:
            month = record["month"]
            port_ret = record.get("portfolio_return")
            bench_ret = record.get("benchmark_return")
            diff = record.get("difference")

            # Format
            month_label = f"{month} (MTD)" if month == current_month else month
            port_str = f"{port_ret*100:.2f}%" if port_ret is not None else "-"
            bench_str = f"{bench_ret*100:.2f}%" if bench_ret is not None else "-"
            diff_str = f"{diff*100:+.2f}%" if diff is not None else "-"

            monthly_table.append({
                "Month": month_label,
                "Portfolio": port_str,
                "EW Benchmark": bench_str,
                "Alpha": diff_str,
            })

        monthly_df = pd.DataFrame(monthly_table)

        # Color function
        def color_monthly_val(val):
            if val == "-":
                return "color: #666"
            try:
                num = float(val.replace("%", "").replace("+", ""))
                if "+" in val:
                    return "color: #22c55e; font-weight: 600"
                elif val.startswith("-"):
                    return "color: #ef4444; font-weight: 600"
                elif num > 0:
                    return "color: #22c55e"
                elif num < 0:
                    return "color: #ef4444"
            except:
                pass
            return ""

        styled_monthly = monthly_df.style.applymap(
            color_monthly_val, subset=["Portfolio", "EW Benchmark", "Alpha"]
        ).set_properties(**{"text-align": "center"}, subset=["Portfolio", "EW Benchmark", "Alpha"])

        st.dataframe(
            styled_monthly,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(monthly_df) + 38)
        )

        # Summary stats
        if monthly_data:
            total_alpha = (cumulative_port - 1) - (cumulative_bench - 1)
            months_with_alpha = sum(1 for r in monthly_data if r.get("difference") and r["difference"] > 0)
            total_months = sum(1 for r in monthly_data if r.get("difference") is not None)

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Cumulative Alpha",
                f"{total_alpha*100:+.2f}%",
                delta=None,
            )
            col2.metric(
                "Win Rate (months)",
                f"{months_with_alpha}/{total_months}",
                delta=f"{months_with_alpha/total_months*100:.0f}%" if total_months > 0 else "N/A",
            )
            col3.metric(
                "Period",
                f"{len(monthly_data)} months",
            )

    # ===== Stock-Level Attribution Analysis =====
    st.markdown("---")
    st.markdown("### Stock-Level Attribution (Buy-and-Hold)")
    st.markdown(
        "Analyze which stocks contributed most to portfolio and benchmark returns "
        "for a specific month. Uses simple buy-and-hold methodology: "
        "weight × (end_price / start_price - 1)."
    )

    # Month selector from available months
    if monthly_data:
        available_months = sorted(
            set(r["month"] for r in monthly_data),
            reverse=True
        )

        selected_month = st.selectbox(
            "Select month for attribution analysis",
            options=available_months,
            key="attribution_month_select"
        )

        if selected_month:
            with st.spinner("Calculating stock attribution..."):
                try:
                    attribution = calculate_stock_attribution(
                        portfolio_id=portfolio_id,
                        source=source,
                        month=selected_month,
                        tracker=tracker,
                    )
                except Exception as e:
                    st.error(f"Error calculating attribution: {str(e)}")
                    attribution = None

            if attribution:
                # Get portfolio ticker list for highlighting in universe table
                portfolio_tickers = [h["ticker"] for h in attribution["portfolio_holdings"]]
                n_portfolio = len(attribution["portfolio_holdings"])
                n_universe = len(attribution["universe_holdings"])

                # Display in two columns
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"#### Portfolio Holdings ({n_portfolio} Tickers)")
                    _render_attribution_table(attribution["portfolio_holdings"], "Portfolio")

                with col2:
                    st.markdown(f"#### Universe (EW Benchmark, {n_universe} Tickers)")
                    _render_attribution_table(
                        attribution["universe_holdings"],
                        "Benchmark",
                        portfolio_tickers=portfolio_tickers,
                        is_universe_table=True,
                    )

                # Summary metrics
                st.markdown("---")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric(
                    "Portfolio Return",
                    f"{attribution['portfolio_total']*100:+.2f}%"
                )
                mcol2.metric(
                    "Benchmark Return",
                    f"{attribution['benchmark_total']*100:+.2f}%"
                )
                mcol3.metric(
                    "Alpha",
                    f"{attribution['alpha']*100:+.2f}%"
                )
                mcol4.metric(
                    "Holdings / Universe",
                    f"{len(attribution['portfolio_holdings'])} / {len(attribution['universe_holdings'])}"
                )

                # Validate alignment with monthly table values
                month_record = next(
                    (r for r in monthly_data if r["month"] == selected_month),
                    None
                )
                if month_record:
                    monthly_port = month_record.get("portfolio_return")
                    monthly_bench = month_record.get("benchmark_return")

                    # Check alignment (both now use buy-and-hold methodology)
                    port_diff = abs((monthly_port or 0) - attribution["portfolio_total"]) if monthly_port else 0
                    bench_diff = abs((monthly_bench or 0) - attribution["benchmark_total"]) if monthly_bench else 0

                    # Small differences can occur due to rounding or missing prices
                    if port_diff > 0.005 or bench_diff > 0.005:
                        st.warning(
                            f"Calculation mismatch detected: Portfolio diff={port_diff*100:.2f}%, "
                            f"Benchmark diff={bench_diff*100:.2f}%. "
                            "This may indicate missing price data or calculation issues."
                        )
    else:
        st.info("No monthly data available for attribution analysis.")


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


def _render_risk_analytics_tab(
    tracker, portfolio_id, variants, start_date, end_date, nav_data
):
    """Render the Risk Analytics tab with VaR, Beta/Alpha, and rolling metrics."""
    from ..tracking.metrics import (
        calculate_returns,
        calculate_var,
        calculate_cvar,
        calculate_beta,
        calculate_alpha,
        calculate_tracking_error,
        calculate_information_ratio,
        calculate_correlation,
        calculate_rolling_sharpe,
        calculate_rolling_volatility,
        calculate_rolling_correlation,
        calculate_rolling_beta,
    )
    from ..tracking.benchmark_adapter import get_benchmark_adapter
    from .charts import (
        create_var_histogram,
        create_scatter_regression,
        create_rolling_metrics_chart,
    )
    from .styles import COLORS
    import numpy as np

    st.markdown("### Risk Analytics")
    st.markdown("Institutional-grade risk metrics including VaR, Beta, Alpha, and rolling analytics.")

    # Benchmark selector
    benchmark_adapter = get_benchmark_adapter()
    benchmark_options = benchmark_adapter.list_benchmarks()

    col1, col2 = st.columns([2, 3])
    with col1:
        selected_benchmark = st.selectbox(
            "Benchmark",
            options=list(benchmark_options.keys()),
            format_func=lambda x: f"{x} - {benchmark_options[x]}",
            key="risk_benchmark_select",
        )

    with col2:
        rolling_window = st.selectbox(
            "Rolling Window",
            options=[30, 60, 90, 252],
            index=1,  # Default to 60 days
            format_func=lambda x: f"{x} days" + (" (~3mo)" if x == 60 else " (~1yr)" if x == 252 else ""),
            key="risk_rolling_window",
        )

    # Get benchmark data
    benchmark_nav = benchmark_adapter.get_benchmark_nav(
        selected_benchmark, start_date, end_date, normalize=False
    )
    benchmark_returns = benchmark_adapter.get_benchmark_returns(
        selected_benchmark, start_date, end_date
    )

    if benchmark_nav.empty:
        st.warning(f"Could not load benchmark data for {selected_benchmark}. Make sure yfinance is installed.")
        return

    # Select variant to analyze
    variant_to_analyze = st.selectbox(
        "Analyze Variant",
        options=variants,
        format_func=lambda x: x.replace("_", " ").title(),
        key="risk_variant_select",
    )

    if variant_to_analyze not in nav_data:
        st.warning(f"No NAV data for {variant_to_analyze}.")
        return

    portfolio_nav = nav_data[variant_to_analyze]
    portfolio_returns = calculate_returns(portfolio_nav)

    # Align returns with benchmark
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) < 30:
        st.warning("Insufficient data for risk analysis. Need at least 30 days of aligned returns.")
        return

    aligned_port_returns = aligned_data.iloc[:, 0]
    aligned_bench_returns = aligned_data.iloc[:, 1]

    # ===== Risk Metrics Section =====
    st.markdown("---")
    st.markdown("#### Value at Risk & Risk Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # VaR metrics
        var_95 = calculate_var(aligned_port_returns, 0.95)
        var_99 = calculate_var(aligned_port_returns, 0.99)
        cvar_95 = calculate_cvar(aligned_port_returns, 0.95)
        cvar_99 = calculate_cvar(aligned_port_returns, 0.99)

        st.markdown("**Value at Risk (VaR)**")
        vcol1, vcol2 = st.columns(2)
        vcol1.metric("95% VaR (Daily)", f"{var_95*100:.2f}%")
        vcol2.metric("99% VaR (Daily)", f"{var_99*100:.2f}%")

        st.markdown("**Conditional VaR (Expected Shortfall)**")
        vcol1, vcol2 = st.columns(2)
        vcol1.metric("95% CVaR", f"{cvar_95*100:.2f}%")
        vcol2.metric("99% CVaR", f"{cvar_99*100:.2f}%")

        # VaR histogram
        var_chart = create_var_histogram(
            aligned_port_returns, var_95, var_99,
            title="Return Distribution with VaR",
            height=300,
        )
        st.plotly_chart(var_chart, use_container_width=True)

    with col2:
        # Beta & Alpha metrics
        beta = calculate_beta(aligned_port_returns, aligned_bench_returns)
        alpha = calculate_alpha(portfolio_nav, benchmark_nav)
        tracking_error = calculate_tracking_error(aligned_port_returns, aligned_bench_returns)
        info_ratio = calculate_information_ratio(portfolio_nav, benchmark_nav)
        correlation = calculate_correlation(aligned_port_returns, aligned_bench_returns)

        st.markdown(f"**Beta & Alpha (vs {selected_benchmark})**")
        bcol1, bcol2 = st.columns(2)
        bcol1.metric("Beta", f"{beta:.2f}")
        bcol2.metric("Alpha (Ann.)", f"{alpha*100:+.2f}%")

        st.markdown("**Active Risk Metrics**")
        bcol1, bcol2 = st.columns(2)
        bcol1.metric("Tracking Error", f"{tracking_error*100:.2f}%")
        bcol2.metric("Information Ratio", f"{info_ratio:.2f}")

        st.metric("Correlation", f"{correlation:.2f}")

        # Scatter plot
        scatter_chart = create_scatter_regression(
            aligned_port_returns, aligned_bench_returns, beta, alpha,
            title=f"Portfolio vs {selected_benchmark}",
            height=300,
        )
        st.plotly_chart(scatter_chart, use_container_width=True)

    # ===== Rolling Metrics Section =====
    st.markdown("---")
    st.markdown(f"#### Rolling Metrics ({rolling_window}-Day Window)")

    # Calculate rolling metrics
    rolling_sharpe = calculate_rolling_sharpe(aligned_port_returns, rolling_window)
    rolling_vol = calculate_rolling_volatility(aligned_port_returns, rolling_window)
    rolling_corr = calculate_rolling_correlation(aligned_port_returns, aligned_bench_returns, rolling_window)
    rolling_beta = calculate_rolling_beta(aligned_port_returns, aligned_bench_returns, rolling_window)

    # Rolling Sharpe chart
    col1, col2 = st.columns(2)

    with col1:
        sharpe_chart = create_rolling_metrics_chart(
            {"Rolling Sharpe": rolling_sharpe},
            title="Rolling Sharpe Ratio",
            height=280,
        )
        st.plotly_chart(sharpe_chart, use_container_width=True)

    with col2:
        vol_chart = create_rolling_metrics_chart(
            {"Rolling Volatility": rolling_vol},
            title="Rolling Volatility (Annualized)",
            height=280,
        )
        st.plotly_chart(vol_chart, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        corr_chart = create_rolling_metrics_chart(
            {"Rolling Correlation": rolling_corr},
            title=f"Rolling Correlation (vs {selected_benchmark})",
            height=280,
        )
        st.plotly_chart(corr_chart, use_container_width=True)

    with col2:
        beta_chart = create_rolling_metrics_chart(
            {"Rolling Beta": rolling_beta},
            title=f"Rolling Beta (vs {selected_benchmark})",
            height=280,
        )
        st.plotly_chart(beta_chart, use_container_width=True)


def _render_drawdown_analysis_tab(
    tracker, portfolio_id, variants, start_date, end_date, nav_data
):
    """Render the Drawdown Analysis tab with detailed drawdown metrics and tables."""
    from ..tracking.metrics import (
        analyze_drawdowns,
        get_worst_drawdowns,
        calculate_drawdown_series,
    )
    from .charts import create_drawdown_highlight_chart
    from .styles import COLORS

    st.markdown("### Drawdown Analysis")
    st.markdown("Comprehensive analysis of portfolio drawdowns including duration, recovery, and time underwater.")

    # Select variant to analyze
    variant_to_analyze = st.selectbox(
        "Analyze Variant",
        options=variants,
        format_func=lambda x: x.replace("_", " ").title(),
        key="dd_variant_select",
    )

    if variant_to_analyze not in nav_data:
        st.warning(f"No NAV data for {variant_to_analyze}.")
        return

    portfolio_nav = nav_data[variant_to_analyze]

    if len(portfolio_nav) < 10:
        st.warning("Insufficient data for drawdown analysis.")
        return

    # Get drawdown analysis
    dd_analysis = analyze_drawdowns(portfolio_nav)
    drawdown_series = calculate_drawdown_series(portfolio_nav)

    # ===== Summary Metrics =====
    st.markdown("---")
    st.markdown("#### Drawdown Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Drawdown",
            f"{dd_analysis['current_drawdown']*100:.2f}%",
            delta=f"{dd_analysis['current_duration_days']} days" if dd_analysis['current_duration_days'] > 0 else "At high",
        )

    with col2:
        st.metric(
            "Max Drawdown",
            f"{dd_analysis['max_drawdown']*100:.2f}%",
        )

    with col3:
        st.metric(
            "Avg Drawdown",
            f"{dd_analysis['avg_drawdown']*100:.2f}%",
        )

    with col4:
        st.metric(
            "Time Underwater",
            f"{dd_analysis['time_underwater_pct']*100:.1f}%",
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Duration", f"{dd_analysis['max_duration_days']} days")

    with col2:
        st.metric("Avg Duration", f"{dd_analysis['avg_duration_days']} days")

    with col3:
        st.metric("# of Drawdowns", f"{dd_analysis['num_drawdowns']}")

    with col4:
        pass  # Placeholder for alignment

    # ===== Drawdown Chart with Highlights =====
    st.markdown("---")
    st.markdown("#### Underwater Plot")

    # Get worst drawdowns for highlighting
    n_worst = st.slider("Highlight top N worst drawdowns", 1, 10, 5, key="dd_highlight_n")
    worst_dds = get_worst_drawdowns(portfolio_nav, n=n_worst)

    dd_chart = create_drawdown_highlight_chart(
        drawdown_series,
        worst_dds,
        title="Drawdown Analysis (Worst Periods Highlighted)",
        height=350,
    )
    st.plotly_chart(dd_chart, use_container_width=True)

    # ===== Worst Drawdowns Table =====
    st.markdown("---")
    st.markdown("#### Worst Drawdowns")

    if worst_dds.empty:
        st.info("No drawdown periods found in the selected date range.")
    else:
        # Format for display
        display_df = worst_dds.copy()
        display_df["Rank"] = range(1, len(display_df) + 1)

        # Format dates
        display_df["Peak Date"] = display_df["peak_date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "-"
        )
        display_df["Trough Date"] = display_df["trough_date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "-"
        )
        display_df["Recovery Date"] = display_df["recovery_date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "Not Recovered"
        )

        # Format percentages
        display_df["Drawdown"] = display_df["drawdown_pct"].apply(lambda x: f"{x*100:.2f}%")
        display_df["Duration"] = display_df["duration_days"].apply(lambda x: f"{x} days")
        display_df["Recovery Time"] = display_df["recovery_days"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else "N/A"
        )

        # Select columns for display
        display_cols = [
            "Rank", "Peak Date", "Trough Date", "Recovery Date",
            "Drawdown", "Duration", "Recovery Time"
        ]
        final_df = display_df[display_cols]

        # Style the table
        def color_drawdown(val):
            if "%" in str(val):
                try:
                    num = float(val.replace("%", ""))
                    if num < -10:
                        return "color: #dc3545; font-weight: 600"
                    elif num < -5:
                        return "color: #ef4444"
                    else:
                        return "color: #f97316"
                except:
                    pass
            return ""

        styled_df = final_df.style.applymap(
            color_drawdown, subset=["Drawdown"]
        ).set_properties(
            **{"text-align": "center"}
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ===== Compare Variants Drawdowns =====
    if len(variants) > 1:
        st.markdown("---")
        st.markdown("#### Variant Comparison")

        comparison_data = []
        for variant in variants:
            if variant in nav_data:
                var_nav = nav_data[variant]
                var_dd = analyze_drawdowns(var_nav)
                comparison_data.append({
                    "Variant": variant.replace("_", " ").title(),
                    "Max DD": f"{var_dd['max_drawdown']*100:.2f}%",
                    "Avg DD": f"{var_dd['avg_drawdown']*100:.2f}%",
                    "Max Duration": f"{var_dd['max_duration_days']} days",
                    "Avg Duration": f"{var_dd['avg_duration_days']} days",
                    "Time Underwater": f"{var_dd['time_underwater_pct']*100:.1f}%",
                    "# Drawdowns": var_dd['num_drawdowns'],
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def _render_multi_portfolio_comparison_tab(tracker, sidebar_start_date, sidebar_end_date):
    """Render the Multi-Portfolio Comparison tab with expandable month/day returns."""
    from ..tracking import Variants
    from .styles import format_percentage, get_variant_display_name, VARIANT_COLORS, COLORS
    from datetime import date, timedelta
    import pandas as pd
    import numpy as np

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

    # Filter to only portfolios with NAV data
    portfolios_with_data = []
    for p in all_portfolios_sorted:
        nav_df = tracker.get_nav_series(p.id, Variants.RAW)
        if not nav_df.empty:
            portfolios_with_data.append(p.name)

    # Function to clean portfolio names for display (remove _EqualWeight suffix)
    def clean_portfolio_name(name: str) -> str:
        """Remove _EqualWeight suffix for cleaner dropdown display."""
        return name.replace("_EqualWeight", "")

    # Default selection: ALL portfolios with NAV data
    default_selection = portfolios_with_data.copy()

    # Use session state to persist selection across visits
    session_key = "multi_portfolio_selected_portfolios"
    if session_key not in st.session_state:
        st.session_state[session_key] = default_selection if default_selection else portfolios_with_data[:3]
    # Filter out any portfolios that no longer have data
    st.session_state[session_key] = [p for p in st.session_state[session_key] if p in portfolios_with_data]
    if not st.session_state[session_key]:
        st.session_state[session_key] = default_selection if default_selection else portfolios_with_data[:3]

    # Multi-select for portfolios and variants side by side
    col1, col2 = st.columns(2)

    with col1:
        selected_portfolio_names = st.multiselect(
            "Select Portfolios",
            options=portfolios_with_data,
            default=st.session_state[session_key],
            key="multi_portfolio_selector",
            format_func=clean_portfolio_name,
            help="Select portfolios to compare (only portfolios with NAV data shown)",
        )
        # Save selection to session state
        st.session_state[session_key] = selected_portfolio_names

    with col2:
        variant_options = Variants.all()
        selected_variants = st.multiselect(
            "Select Variants",
            options=variant_options,
            default=[Variants.RAW],  # Default to raw only
            format_func=get_variant_display_name,
            key="multi_portfolio_variants",
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

    # Controls row: month selector and allocation toggle
    col_month, col_alloc = st.columns([3, 2])
    with col_month:
        selected_month = st.selectbox(
            "Select month:",
            options=all_months,
            format_func=lambda x: f"{x} (MTD)" if x == current_month else x,
            key="daily_details_month",
        )
    with col_alloc:
        st.markdown("")  # Spacer to align with selectbox
        show_allocations = st.checkbox(
            "Show Model Allocations (from S3)",
            key="show_allocations_toggle",
            help="Show target and actual allocations for Conservative and Trend Regime models",
        )

    # Load allocation history from S3 if toggle is enabled
    allocation_data = {}
    if show_allocations:
        try:
            from ..tracking.s3_adapter import S3DataLoader
            s3_loader = S3DataLoader()
            for model in ["conservative", "trend_regime_v2"]:
                try:
                    alloc_df = s3_loader.load_allocation_history(model)
                    if not alloc_df.empty:
                        # Convert date to string for matching
                        alloc_df["date_str"] = alloc_df["date"].dt.strftime("%Y-%m-%d")
                        allocation_data[model] = alloc_df.set_index("date_str")
                except Exception as e:
                    logger.warning(f"Could not load allocation history for {model}: {e}")
        except Exception as e:
            st.warning(f"Could not load allocation data from S3: {e}")

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

                        # Add allocation columns for overlay variants if enabled
                        if show_allocations and variant in ["conservative", "trend_regime_v2"]:
                            abbrev = variant_abbrev.get(variant, variant)
                            if variant in allocation_data:
                                alloc_df = allocation_data[variant]
                                if day in alloc_df.index:
                                    try:
                                        target_alloc = float(alloc_df.loc[day, "target_allocation"])
                                        actual_alloc = float(alloc_df.loc[day, "allocation"])
                                        row[f"{abbrev} Target"] = f"{target_alloc*100:.0f}%"
                                        row[f"{abbrev} Actual"] = f"{actual_alloc*100:.0f}%"
                                    except Exception:
                                        row[f"{abbrev} Target"] = "-"
                                        row[f"{abbrev} Actual"] = "-"
                                else:
                                    row[f"{abbrev} Target"] = "-"
                                    row[f"{abbrev} Actual"] = "-"
                            else:
                                # S3 data not loaded for this model
                                row[f"{abbrev} Target"] = "N/A"
                                row[f"{abbrev} Actual"] = "N/A"

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

    # ===== Correlation Matrix Section =====
    # Senior quant approach: correlate daily returns, not prices
    # Use inner join for date alignment (only days where all portfolios have data)
    st.markdown("---")
    st.markdown("#### Portfolio Correlation Matrix")
    st.markdown("Correlation of daily returns between selected portfolios. Higher correlation = less diversification benefit.")

    # Build returns DataFrame for correlation calculation
    from ..tracking.metrics import calculate_returns
    from .charts import create_correlation_heatmap, create_efficient_frontier_scatter

    # Collect daily returns for each portfolio/variant combination
    returns_data = {}
    performance_data = []  # For risk-return scatter

    # Diagnostic: Track why portfolios are excluded from correlation
    corr_diagnostics = []

    for portfolio_name in portfolios_with_data:
        portfolio_id = portfolio_options[portfolio_name]
        clean_pname = clean_name(portfolio_name)

        for variant in selected_variants:
            col_name = f"{clean_pname} {variant_abbrev.get(variant, variant)}"
            nav_df = tracker.get_nav_series(portfolio_id, variant, start_date, end_date)

            if nav_df.empty:
                corr_diagnostics.append({
                    "portfolio": col_name,
                    "status": "❌ Empty NAV",
                    "reason": "get_nav_series returned empty DataFrame",
                    "rows": 0,
                })
                continue

            if "nav" not in nav_df.columns:
                corr_diagnostics.append({
                    "portfolio": col_name,
                    "status": "❌ No NAV col",
                    "reason": f"Columns: {list(nav_df.columns)}",
                    "rows": len(nav_df),
                })
                continue

            portfolio_nav = nav_df["nav"]
            nav_len = len(portfolio_nav)

            if nav_len < 30:
                corr_diagnostics.append({
                    "portfolio": col_name,
                    "status": "❌ < 30 days",
                    "reason": f"Only {nav_len} data points (need 30+)",
                    "rows": nav_len,
                })
                continue

            # Use pre-computed daily returns if available, else calculate
            if "daily_return" in nav_df.columns:
                returns = nav_df["daily_return"].dropna()
            else:
                returns = calculate_returns(portfolio_nav)

            returns_data[col_name] = returns
            corr_diagnostics.append({
                "portfolio": col_name,
                "status": "✅ Included",
                "reason": f"{len(returns)} returns, dates: {returns.index.min().date()} to {returns.index.max().date()}",
                "rows": nav_len,
            })

            # Collect metrics for risk-return scatter
            from ..tracking.metrics import calculate_cagr, calculate_volatility
            try:
                cagr = calculate_cagr(portfolio_nav)
                vol = calculate_volatility(returns)
                if cagr != 0 and vol != 0:
                    performance_data.append({
                        'name': col_name,
                        'return': cagr,
                        'volatility': vol,
                        'sharpe': cagr / vol if vol > 0 else 0,
                    })
            except Exception:
                pass

    if len(returns_data) >= 2:
        # Align returns by date (inner join - only dates present in all series)
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) >= 30:
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()

            col1, col2 = st.columns([3, 2])

            with col1:
                # Correlation heatmap
                heatmap = create_correlation_heatmap(
                    corr_matrix,
                    title="Returns Correlation Matrix",
                    height=max(350, 40 * len(corr_matrix)),
                )
                st.plotly_chart(heatmap, use_container_width=True)

            with col2:
                # Summary statistics
                st.markdown("**Correlation Summary**")

                # Get off-diagonal correlations
                mask = ~np.eye(len(corr_matrix), dtype=bool)
                off_diag = corr_matrix.values[mask]

                st.metric("Average Correlation", f"{np.mean(off_diag):.3f}")
                st.metric("Max Correlation", f"{np.max(off_diag):.3f}")
                st.metric("Min Correlation", f"{np.min(off_diag):.3f}")
                st.metric("Obs. Days (Aligned)", f"{len(returns_df)}")

                st.markdown("---")
                st.markdown("**Interpretation:**")
                st.caption(
                    "**< 0.3**: Low correlation (good diversification)  \n"
                    "**0.3 - 0.7**: Moderate correlation  \n"
                    "**> 0.7**: High correlation (limited diversification)"
                )

            # Show raw correlation values in expandable section
            with st.expander("View Raw Correlation Values"):
                # Format correlation matrix for display
                styled_corr = corr_matrix.style.background_gradient(
                    cmap='RdYlGn_r', vmin=-1, vmax=1
                ).format("{:.3f}")
                st.dataframe(styled_corr, use_container_width=True)

        else:
            st.caption(f"Need at least 30 aligned trading days for correlation. Found: {len(returns_df)}")
    else:
        st.caption("Select at least 2 portfolios to view correlation matrix.")

    # Diagnostic expander - shows why portfolios were included/excluded
    if corr_diagnostics:
        with st.expander("🔍 Portfolio Data Summary", expanded=True):
            st.markdown(f"**Input:** {len(portfolios_with_data)} portfolios × {len(selected_variants)} variants = {len(portfolios_with_data) * len(selected_variants)} combinations")
            st.markdown(f"**Output:** {len(returns_data)} series included in correlation matrix")

            # Summary counts
            included = sum(1 for d in corr_diagnostics if d["status"].startswith("✅"))
            excluded = len(corr_diagnostics) - included
            st.markdown(f"**Included:** {included} | **Excluded:** {excluded}")

            # Show diagnostic table
            diag_df = pd.DataFrame(corr_diagnostics)
            st.dataframe(diag_df, use_container_width=True, hide_index=True)

            # Additional debug: show date ranges of included series
            if returns_data:
                st.markdown("**Date Alignment Check:**")
                date_ranges = []
                for name, returns in returns_data.items():
                    date_ranges.append({
                        "Series": name,
                        "Start": returns.index.min().date(),
                        "End": returns.index.max().date(),
                        "Count": len(returns),
                    })
                st.dataframe(pd.DataFrame(date_ranges), use_container_width=True, hide_index=True)

    # ===== Risk-Return Scatter Plot =====
    if len(performance_data) >= 2:
        st.markdown("---")
        st.markdown("#### Risk-Return Profile")
        st.markdown("Annualized return vs volatility. Higher and to the left is better (higher return per unit risk).")

        scatter = create_efficient_frontier_scatter(
            performance_data,
            title="Portfolio Risk-Return Profile",
            height=400,
        )
        st.plotly_chart(scatter, use_container_width=True)

    # ===== Sector Exposure by Month =====
    st.markdown("---")
    st.markdown("#### Sector Exposure by Month")

    # Option to combine all portfolios
    combine_all_sectors = st.checkbox(
        "Combine All Portfolios",
        value=False,
        key="sector_combine_all",
        help="Aggregate sector exposure across all selected portfolios (average weights)"
    )

    # Get sector data for all tickers
    from ..data_manager import StockDataManager
    from ..models import TickerInfo
    from ..db import get_session
    from sqlmodel import select

    # Build sector lookup from TickerInfo table
    sector_lookup = {}
    with get_session() as session:
        ticker_infos = session.exec(select(TickerInfo)).all()
        for ti in ticker_infos:
            if ti.sector:
                sector_lookup[ti.ticker] = ti.sector

    # Get months in date range (respecting exact start_date and end_date)
    sector_months = []  # List of (month_label, holdings_date) tuples
    current_month = start_date.replace(day=1)
    while current_month <= end_date:
        # Use the later of (first day of month) and (start_date) for holdings lookup
        holdings_date = max(current_month, start_date)
        # But cap at end_date
        holdings_date = min(holdings_date, end_date)
        month_label = current_month.strftime("%b %Y")
        sector_months.append((month_label, holdings_date))

        if current_month.month == 12:
            current_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            current_month = current_month.replace(month=current_month.month + 1)

    # Build sector exposure data for each portfolio and month
    sector_data = []
    missing_sector_tickers = {}  # {ticker: [portfolios]}

    for portfolio_name in selected_portfolio_names:
        portfolio_id = portfolio_options[portfolio_name]
        clean_pf_name = clean_name(portfolio_name)

        for month_label, holdings_date in sector_months:
            # Get holdings for this month using a date within the selected range
            holdings = tracker.get_holdings(portfolio_id, holdings_date)

            if holdings:
                # Calculate sector weights
                sector_weights = {}
                total_weight = 0

                for h in holdings:
                    ticker = h.ticker
                    weight = float(h.weight) if h.weight else 0
                    sector = sector_lookup.get(ticker)

                    # Track tickers with missing sector
                    if not sector:
                        sector = "Unknown"
                        if ticker not in missing_sector_tickers:
                            missing_sector_tickers[ticker] = set()
                        missing_sector_tickers[ticker].add(clean_pf_name)

                    if sector not in sector_weights:
                        sector_weights[sector] = 0
                    sector_weights[sector] += weight
                    total_weight += weight

                # Normalize weights if needed
                if total_weight > 0:
                    for sector in sector_weights:
                        sector_weights[sector] = sector_weights[sector] / total_weight

                # Add row for each sector
                for sector, weight in sector_weights.items():
                    sector_data.append({
                        "Portfolio": clean_pf_name,
                        "Month": month_label,
                        "Sector": sector,
                        "Weight": weight,
                    })

    if sector_data:
        sector_df = pd.DataFrame(sector_data)
        month_order = [m[0] for m in sector_months]  # m[0] is month_label

        if combine_all_sectors:
            # Aggregate across all portfolios: average sector weight per month
            # Only average across portfolios that have data for each month
            # (don't include portfolios that didn't exist yet)

            all_sectors = sector_df["Sector"].unique()

            # Find which portfolios have data for each month
            portfolios_per_month = sector_df.groupby("Month")["Portfolio"].apply(set).to_dict()

            # For each month, create complete sector grid only for portfolios that exist
            combined_data = []
            for month in sector_df["Month"].unique():
                month_portfolios = portfolios_per_month.get(month, set())
                month_df = sector_df[sector_df["Month"] == month]

                # Create complete grid for this month's portfolios × all sectors
                from itertools import product
                month_grid = pd.DataFrame(
                    list(product(month_portfolios, [month], all_sectors)),
                    columns=["Portfolio", "Month", "Sector"]
                )
                month_complete = month_grid.merge(
                    month_df, on=["Portfolio", "Month", "Sector"], how="left"
                ).fillna(0)
                combined_data.append(month_complete)

            sector_df_complete = pd.concat(combined_data, ignore_index=True)

            # Now average across portfolios - only portfolios that existed in each month
            pivot_df = sector_df_complete.groupby(["Month", "Sector"])["Weight"].mean().unstack(fill_value=0)
            pivot_df = pivot_df.reindex(index=[c for c in month_order if c in pivot_df.index])
            # Transpose so sectors are rows, months are columns
            pivot_df = pivot_df.T

            # Count portfolios per month for caption
            portfolios_counts = {m: len(p) for m, p in portfolios_per_month.items()}
            count_str = ", ".join(f"{m}: {c}" for m, c in sorted(portfolios_counts.items()))
            st.caption(f"Average sector weights (portfolios per month: {count_str})")

            # Format as percentages
            styled_pivot = pivot_df.style.format("{:.1%}").background_gradient(
                cmap="Blues", axis=None, vmin=0, vmax=0.5
            )
            st.dataframe(styled_pivot, use_container_width=True)
        else:
            # Show by portfolio: rows = Portfolio + Sector, columns = Month
            pivot_df = sector_df.pivot_table(
                index=["Portfolio", "Sector"],
                columns="Month",
                values="Weight",
                aggfunc="sum",
                fill_value=0
            )

            # Reorder columns chronologically
            pivot_df = pivot_df.reindex(columns=[c for c in month_order if c in pivot_df.columns])

            # Format as percentages
            styled_pivot = pivot_df.style.format("{:.1%}").background_gradient(
                cmap="Blues", axis=None, vmin=0, vmax=0.5
            )
            st.dataframe(styled_pivot, use_container_width=True)

            # Also show a summary by sector across all portfolios
            with st.expander("Sector Summary (All Portfolios Combined)"):
                summary_df = sector_df.groupby(["Month", "Sector"])["Weight"].mean().unstack(fill_value=0)
                summary_df = summary_df.reindex(index=[c for c in month_order if c in summary_df.index])
                styled_summary = summary_df.style.format("{:.1%}").background_gradient(
                    cmap="Greens", axis=None, vmin=0, vmax=0.5
                )
                st.dataframe(styled_summary, use_container_width=True)
    else:
        st.info("No sector data available for the selected portfolios and date range.")

    # Show tickers with missing sector data
    if missing_sector_tickers:
        with st.expander(f"⚠️ Tickers Missing Sector Data ({len(missing_sector_tickers)})"):
            st.markdown("The following tickers are categorized as 'Unknown' because sector data is missing:")
            missing_data = []
            for ticker, portfolios in sorted(missing_sector_tickers.items()):
                missing_data.append({
                    "Ticker": ticker,
                    "Portfolios": ", ".join(sorted(portfolios))
                })
            st.dataframe(pd.DataFrame(missing_data), use_container_width=True, hide_index=True)
            st.caption("To fix: Add sector data to the TickerInfo table for these tickers.")


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

    # Filter to only portfolios with NAV data
    portfolios_with_data = []
    for p in all_portfolios_sorted:
        nav_df = tracker.get_nav_series(p.id, Variants.RAW)
        if not nav_df.empty:
            portfolios_with_data.append(p.name)

    if not portfolios_with_data:
        st.warning("No portfolios with NAV data available.")
        return

    # Function to clean portfolio names for display (remove _EqualWeight suffix)
    def clean_portfolio_name(name: str) -> str:
        """Remove _EqualWeight suffix for cleaner dropdown display."""
        return name.replace("_EqualWeight", "")

    # Default selection for Scraper View: ALL portfolios with NAV data
    default_selected = portfolios_with_data.copy()

    # Use session state to persist selection across visits
    session_key = "scraper_view_selected_portfolios"
    if session_key not in st.session_state:
        st.session_state[session_key] = default_selected
    # Filter out any portfolios that no longer have data
    st.session_state[session_key] = [p for p in st.session_state[session_key] if p in portfolios_with_data]
    if not st.session_state[session_key]:
        st.session_state[session_key] = default_selected if default_selected else portfolios_with_data[:3]

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
        # Default to Oct 1, 2025 until today
        default_start = date(2025, 10, 1)
        date_range = st.date_input(
            "Date Range",
            value=(default_start, sidebar_end_date),
            key="scraper_view_dates",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_start, sidebar_end_date

    # Portfolio filter - only show portfolios with NAV data
    selected_portfolios = st.multiselect(
        "Select Portfolios",
        options=portfolios_with_data,
        default=st.session_state[session_key],
        key="scraper_view_portfolios_selector",
        format_func=clean_portfolio_name,
        help="Only portfolios with NAV data are shown",
    )
    # Save selection to session state
    st.session_state[session_key] = selected_portfolios

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
        cleaned_name = clean_name(portfolio.name)
        returns_data[cleaned_name] = returns

    if not returns_data:
        st.warning("No data available for the selected date range and variant.")
        return

    # Create DataFrame with portfolios as rows, dates as columns
    df = pd.DataFrame(returns_data).T

    # Sort columns (dates) chronologically
    df = df.reindex(sorted(df.columns), axis=1)

    # Group dates by month and calculate compounded returns
    date_cols = list(df.columns)
    month_groups = {}  # month_key -> list of date columns
    month_totals = {}  # month_key -> total column name

    if date_cols:
        # Group by year-month
        for d in date_cols:
            if hasattr(d, 'strftime'):
                month_key = d.strftime("%Y-%m")
                if month_key not in month_groups:
                    month_groups[month_key] = []
                month_groups[month_key].append(d)

        # Build new column order with monthly totals inserted
        new_columns = []
        for month_key in sorted(month_groups.keys()):
            month_dates = month_groups[month_key]
            # Calculate monthly total for each portfolio (compounded return)
            month_name = pd.Timestamp(month_dates[0]).strftime("%b %Y")
            total_col_name = f"{month_name} Total"
            month_totals[month_key] = total_col_name

            # Calculate compounded monthly return for each portfolio
            # Industry standard (GIPS/IBKR): first day's return belongs to current month
            # Dec MTD = compound of all daily returns on December trading days
            monthly_returns = []
            for portfolio in df.index:
                month_data = df.loc[portfolio, month_dates].dropna()
                if len(month_data) > 0:
                    compounded = (1 + month_data).prod() - 1
                    monthly_returns.append(compounded)
                else:
                    monthly_returns.append(np.nan)

            df[total_col_name] = monthly_returns
            # Add total first, then daily columns (total always visible)
            new_columns.append(total_col_name)
            new_columns.extend(month_dates)

        # Reorder columns
        df = df[new_columns]

    # Format column headers - use unique names to avoid duplicates
    # Daily columns get "YYYY-MM-DD" format internally, display as just day number
    date_to_field = {}  # Maps original date to field name for AgGrid
    for col in df.columns:
        if hasattr(col, 'strftime'):
            # Use full date as field name to ensure uniqueness
            field_name = col.strftime("%Y-%m-%d")
            date_to_field[col] = field_name
        else:
            date_to_field[col] = str(col)

    df.columns = [date_to_field.get(c, str(c)) for c in df.columns]

    # Convert to percentages for display
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""
        )

    # Reset index to make portfolio name a column
    display_df = display_df.reset_index()
    display_df.rename(columns={"index": "Portfolio"}, inplace=True)

    # Use AgGrid with column grouping for collapsible months
    from st_aggrid import AgGrid, JsCode

    # JavaScript function for cell styling based on return value
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.value === '' || params.value === null || params.value === undefined) {
            return {};
        }
        try {
            var num = parseFloat(params.value.replace('%', ''));
            var isTotal = params.colDef.field.includes('Total');

            if (isTotal) {
                if (num > 5) return {'backgroundColor': '#145214', 'color': 'white', 'fontWeight': 'bold'};
                else if (num > 2) return {'backgroundColor': '#1e7b1e', 'color': 'white', 'fontWeight': 'bold'};
                else if (num > 0) return {'backgroundColor': '#2d8f2d', 'color': 'white', 'fontWeight': 'bold'};
                else if (num === 0) return {'backgroundColor': '#6c757d', 'color': 'white', 'fontWeight': 'bold'};
                else if (num > -2) return {'backgroundColor': '#c82333', 'color': 'white', 'fontWeight': 'bold'};
                else if (num > -5) return {'backgroundColor': '#a71d2a', 'color': 'white', 'fontWeight': 'bold'};
                else return {'backgroundColor': '#6b0f18', 'color': 'white', 'fontWeight': 'bold'};
            } else {
                if (num > 5) return {'backgroundColor': '#1e7b1e', 'color': 'white'};
                else if (num > 2) return {'backgroundColor': '#28a745', 'color': 'white'};
                else if (num > 0) return {'backgroundColor': '#90EE90', 'color': 'black'};
                else if (num === 0) return {'backgroundColor': '#f8f9fa', 'color': 'black'};
                else if (num > -2) return {'backgroundColor': '#ffcccb', 'color': 'black'};
                else if (num > -5) return {'backgroundColor': '#dc3545', 'color': 'white'};
                else return {'backgroundColor': '#8b0000', 'color': 'white'};
            }
        } catch(e) {
            return {};
        }
    }
    """)

    # Build column definitions with grouping
    # Current month for determining which groups are open
    current_month = date.today().strftime("%Y-%m")

    column_defs = [
        {
            "field": "Portfolio",
            "pinned": "left",
            "minWidth": 200,
            "cellStyle": {"fontWeight": "bold"},
        }
    ]

    # Build column groups for each month
    for month_key in sorted(month_groups.keys()):
        month_dates = month_groups[month_key]
        total_col = month_totals[month_key]
        month_display = pd.Timestamp(f"{month_key}-01").strftime("%b %Y")

        # Is this the current month? Expand current month by default
        is_current_month = month_key == current_month

        # Build children columns for this month group
        children = []

        # Daily columns - only shown when group is open
        for d in month_dates:
            field_name = d.strftime("%Y-%m-%d")  # Unique field name
            day_label = d.strftime("%d %b")  # Display as "01 Dec"
            children.append({
                "field": field_name,
                "headerName": day_label,
                "width": 65,
                "columnGroupShow": "open",  # Only show when expanded
                "cellStyle": cell_style_jscode,
            })

        # Total column - always visible (shown when closed), placed at the end
        children.append({
            "field": total_col,
            "headerName": "Total",
            "width": 70,
            "cellStyle": cell_style_jscode,
        })

        # Add the month column group
        column_defs.append({
            "headerName": month_display,
            "children": children,
            "openByDefault": is_current_month,
            "marryChildren": True,
        })

    # Build grid options
    grid_options = {
        "columnDefs": column_defs,
        "defaultColDef": {
            "resizable": True,
            "sortable": True,
        },
        "suppressColumnVirtualisation": True,
        "groupHeaderHeight": 25,
    }

    # Calculate height based on number of rows
    height = 35 * (len(display_df) + 1) + 60

    # Display AgGrid
    AgGrid(
        display_df,
        gridOptions=grid_options,
        height=height,
        allow_unsafe_jscode=True,
        theme="streamlit",
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

    # ===== Ticker-Level Returns Section =====
    st.markdown("---")
    st.markdown("### Ticker-Level Returns")
    st.markdown("View daily returns for individual holdings within a portfolio.")

    # Portfolio selector for ticker view
    ticker_portfolio = st.selectbox(
        "Select Portfolio for Ticker View",
        options=portfolios_with_data,
        format_func=clean_portfolio_name,
        key="scraper_view_ticker_portfolio",
    )

    if ticker_portfolio:
        # Get portfolio object
        portfolio_obj = next((p for p in all_portfolios_sorted if p.name == ticker_portfolio), None)
        if portfolio_obj:
            # Get holdings across all months in the date range
            # This ensures we show the correct tickers for each period (handles rebalances)
            all_tickers = set()
            holdings_by_date = {}  # date -> set of tickers active on that date

            # Get first day of each month in the range
            current = start_date.replace(day=1)
            month_dates = []
            while current <= end_date:
                month_dates.append(current)
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

            # Get holdings for each month-start date
            for month_start in month_dates:
                month_holdings = tracker.get_holdings(portfolio_obj.id, month_start)
                if month_holdings:
                    month_tickers = set(h.ticker for h in month_holdings)
                    all_tickers.update(month_tickers)
                    # Get effective_date to know when these holdings are valid
                    eff_date = month_holdings[0].effective_date
                    holdings_by_date[eff_date] = month_tickers

            # Also get end_date holdings in case of recent rebalance
            end_holdings = tracker.get_holdings(portfolio_obj.id, end_date)
            if end_holdings:
                all_tickers.update(h.ticker for h in end_holdings)
                eff_date = end_holdings[0].effective_date
                holdings_by_date[eff_date] = set(h.ticker for h in end_holdings)

            # Convert to sorted lists
            tickers = sorted(all_tickers)
            weights = {}  # weights will vary by period, use end_date weights for display
            if end_holdings:
                weights = {h.ticker: float(h.weight) if h.weight else 0 for h in end_holdings}

            # Build date-to-tickers mapping for masking returns
            # For each trading date, determine which tickers were in the portfolio
            effective_dates_sorted = sorted(holdings_by_date.keys())

            def get_active_tickers_for_date(trade_date):
                """Get which tickers were in portfolio on a given date."""
                # Find the most recent effective_date <= trade_date
                active_eff = None
                for eff in effective_dates_sorted:
                    if eff <= trade_date:
                        active_eff = eff
                    else:
                        break
                return holdings_by_date.get(active_eff, set()) if active_eff else set()

            if tickers:

                # Get price data for these tickers
                # Fetch extra days BEFORE start_date to calculate first day's return
                # (pct_change needs previous day's price)
                from ..data_manager import StockDataManager
                dm = StockDataManager()
                price_fetch_start = start_date - timedelta(days=10)  # Buffer for weekends/holidays

                try:
                    price_dicts = dm.get_price_data(
                        tickers,
                        price_fetch_start.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )

                    if price_dicts:
                        price_df = pd.DataFrame(price_dicts)
                        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

                        # Calculate daily returns for each ticker
                        # We fetched extra days before start_date to calculate first day's return
                        ticker_returns = {}
                        for ticker in tickers:
                            ticker_data = price_df[price_df["ticker"] == ticker].copy()
                            ticker_data = ticker_data.sort_values("trade_date")

                            if len(ticker_data) > 1:
                                # Use adjusted close if available
                                if "adjusted_close" in ticker_data.columns:
                                    prices = ticker_data["adjusted_close"].fillna(ticker_data["close"])
                                else:
                                    prices = ticker_data["close"]

                                # Calculate returns (first day will now have return from previous day)
                                ticker_data["return"] = prices.pct_change()

                                # Filter to only include dates within user's selected range
                                ticker_data = ticker_data[ticker_data["trade_date"] >= start_date]

                                returns_series = pd.Series(
                                    ticker_data["return"].values,
                                    index=ticker_data["trade_date"].values
                                )

                                # Mask returns for dates when ticker wasn't in portfolio
                                # (handles rebalances where tickers are added/removed)
                                for trade_date in returns_series.index:
                                    active_tickers = get_active_tickers_for_date(trade_date)
                                    if ticker not in active_tickers:
                                        returns_series[trade_date] = np.nan

                                ticker_returns[ticker] = returns_series

                        if ticker_returns:
                            # Create DataFrame (tickers as rows, dates as columns)
                            ticker_df = pd.DataFrame(ticker_returns).T
                            ticker_df = ticker_df.reindex(sorted(ticker_df.columns), axis=1)

                            # Group by month like the portfolio table
                            ticker_date_cols = list(ticker_df.columns)
                            ticker_month_groups = {}
                            ticker_month_totals = {}

                            for d in ticker_date_cols:
                                if hasattr(d, 'strftime'):
                                    month_key = d.strftime("%Y-%m")
                                elif isinstance(d, date):
                                    month_key = d.strftime("%Y-%m")
                                else:
                                    continue
                                if month_key not in ticker_month_groups:
                                    ticker_month_groups[month_key] = []
                                ticker_month_groups[month_key].append(d)

                            # Add monthly totals
                            ticker_new_columns = []
                            for month_key in sorted(ticker_month_groups.keys()):
                                month_dates = ticker_month_groups[month_key]
                                month_name = pd.Timestamp(month_dates[0]).strftime("%b %Y")
                                total_col_name = f"{month_name} Total"
                                ticker_month_totals[month_key] = total_col_name

                                # Calculate compounded monthly return (GIPS standard)
                                monthly_returns = []
                                for ticker in ticker_df.index:
                                    month_data = ticker_df.loc[ticker, month_dates].dropna()
                                    if len(month_data) > 0:
                                        compounded = (1 + month_data).prod() - 1
                                        monthly_returns.append(compounded)
                                    else:
                                        monthly_returns.append(np.nan)

                                ticker_df[total_col_name] = monthly_returns
                                ticker_new_columns.append(total_col_name)
                                ticker_new_columns.extend(month_dates)

                            ticker_df = ticker_df[ticker_new_columns]

                            # Format column headers
                            ticker_date_to_field = {}
                            for col in ticker_df.columns:
                                if hasattr(col, 'strftime'):
                                    ticker_date_to_field[col] = col.strftime("%Y-%m-%d")
                                elif isinstance(col, date):
                                    ticker_date_to_field[col] = col.strftime("%Y-%m-%d")
                                else:
                                    ticker_date_to_field[col] = str(col)

                            ticker_df.columns = [ticker_date_to_field.get(c, str(c)) for c in ticker_df.columns]

                            # Add Portfolio Total row (weighted sum of returns)
                            portfolio_total = {}
                            for col in ticker_df.columns:
                                weighted_sum = 0.0
                                for ticker in ticker_df.index:
                                    ret = ticker_df.loc[ticker, col]
                                    if pd.notna(ret):
                                        weighted_sum += weights.get(ticker, 0) * ret
                                portfolio_total[col] = weighted_sum

                            # Add total row to dataframe
                            ticker_df.loc["Portfolio Total"] = portfolio_total

                            # Convert to percentages
                            ticker_display_df = ticker_df.copy()
                            for col in ticker_display_df.columns:
                                ticker_display_df[col] = ticker_display_df[col].apply(
                                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""
                                )

                            # Add weight column and reset index
                            ticker_display_df = ticker_display_df.reset_index()
                            ticker_display_df.rename(columns={"index": "Ticker"}, inplace=True)
                            ticker_display_df.insert(1, "Weight", ticker_display_df["Ticker"].map(
                                lambda t: "100.0%" if t == "Portfolio Total" else f"{weights.get(t, 0)*100:.1f}%"
                            ))

                            # JavaScript for highlighting the Portfolio Total row
                            ticker_cell_style = JsCode("""
                            function(params) {
                                if (params.data && params.data.Ticker === 'Portfolio Total') {
                                    return {'fontWeight': 'bold', 'backgroundColor': '#e3f2fd'};
                                }
                                return {'fontWeight': 'bold'};
                            }
                            """)

                            weight_cell_style = JsCode("""
                            function(params) {
                                if (params.data && params.data.Ticker === 'Portfolio Total') {
                                    return {'fontWeight': 'bold', 'backgroundColor': '#bbdefb'};
                                }
                                return {'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'};
                            }
                            """)

                            # Cell style for returns that also highlights total row
                            ticker_return_style = JsCode("""
                            function(params) {
                                var isPortfolioTotal = params.data && params.data.Ticker === 'Portfolio Total';
                                var baseStyle = {};

                                if (params.value === '' || params.value === null || params.value === undefined) {
                                    return isPortfolioTotal ? {'backgroundColor': '#e3f2fd'} : {};
                                }
                                try {
                                    var num = parseFloat(params.value.replace('%', ''));
                                    var isTotal = params.colDef.field.includes('Total');

                                    if (isPortfolioTotal || isTotal) {
                                        if (num > 5) baseStyle = {'backgroundColor': '#145214', 'color': 'white', 'fontWeight': 'bold'};
                                        else if (num > 2) baseStyle = {'backgroundColor': '#1e7b1e', 'color': 'white', 'fontWeight': 'bold'};
                                        else if (num > 0) baseStyle = {'backgroundColor': '#2d8f2d', 'color': 'white', 'fontWeight': 'bold'};
                                        else if (num === 0) baseStyle = {'backgroundColor': '#6c757d', 'color': 'white', 'fontWeight': 'bold'};
                                        else if (num > -2) baseStyle = {'backgroundColor': '#c82333', 'color': 'white', 'fontWeight': 'bold'};
                                        else if (num > -5) baseStyle = {'backgroundColor': '#a71d2a', 'color': 'white', 'fontWeight': 'bold'};
                                        else baseStyle = {'backgroundColor': '#6b0f18', 'color': 'white', 'fontWeight': 'bold'};
                                    } else {
                                        if (num > 5) baseStyle = {'backgroundColor': '#1e7b1e', 'color': 'white'};
                                        else if (num > 2) baseStyle = {'backgroundColor': '#28a745', 'color': 'white'};
                                        else if (num > 0) baseStyle = {'backgroundColor': '#90EE90', 'color': 'black'};
                                        else if (num === 0) baseStyle = {'backgroundColor': '#f8f9fa', 'color': 'black'};
                                        else if (num > -2) baseStyle = {'backgroundColor': '#ffcccb', 'color': 'black'};
                                        else if (num > -5) baseStyle = {'backgroundColor': '#dc3545', 'color': 'white'};
                                        else baseStyle = {'backgroundColor': '#8b0000', 'color': 'white'};
                                    }
                                } catch(e) {
                                    return isPortfolioTotal ? {'backgroundColor': '#e3f2fd'} : {};
                                }
                                return baseStyle;
                            }
                            """)

                            # Build column defs for ticker grid
                            ticker_column_defs = [
                                {
                                    "field": "Ticker",
                                    "pinned": "left",
                                    "minWidth": 80,
                                    "cellStyle": ticker_cell_style,
                                },
                                {
                                    "field": "Weight",
                                    "pinned": "left",
                                    "width": 70,
                                    "cellStyle": weight_cell_style,
                                }
                            ]

                            # Build column groups for each month
                            for month_key in sorted(ticker_month_groups.keys()):
                                month_dates = ticker_month_groups[month_key]
                                total_col = ticker_month_totals[month_key]
                                month_display = pd.Timestamp(f"{month_key}-01").strftime("%b %Y")
                                is_current_month = month_key == current_month

                                children = []
                                for d in month_dates:
                                    if hasattr(d, 'strftime'):
                                        field_name = d.strftime("%Y-%m-%d")
                                        day_label = d.strftime("%d %b")
                                    elif isinstance(d, date):
                                        field_name = d.strftime("%Y-%m-%d")
                                        day_label = d.strftime("%d %b")
                                    else:
                                        continue
                                    children.append({
                                        "field": field_name,
                                        "headerName": day_label,
                                        "width": 65,
                                        "columnGroupShow": "open",
                                        "cellStyle": ticker_return_style,
                                    })

                                children.append({
                                    "field": total_col,
                                    "headerName": "Total",
                                    "width": 70,
                                    "cellStyle": ticker_return_style,
                                })

                                ticker_column_defs.append({
                                    "headerName": month_display,
                                    "children": children,
                                    "openByDefault": is_current_month,
                                    "marryChildren": True,
                                })

                            ticker_grid_options = {
                                "columnDefs": ticker_column_defs,
                                "defaultColDef": {
                                    "resizable": True,
                                    "sortable": True,
                                },
                                "suppressColumnVirtualisation": True,
                                "groupHeaderHeight": 25,
                            }

                            ticker_height = 35 * (len(ticker_display_df) + 1) + 60

                            AgGrid(
                                ticker_display_df,
                                gridOptions=ticker_grid_options,
                                height=min(ticker_height, 600),  # Cap height
                                allow_unsafe_jscode=True,
                                theme="streamlit",
                                key="ticker_level_grid",
                            )

                            # Ticker summary
                            st.markdown("#### Ticker Performance Summary")
                            ticker_summary = []
                            for ticker in tickers:
                                if ticker in ticker_returns:
                                    valid_returns = ticker_returns[ticker].dropna()
                                    if len(valid_returns) > 0:
                                        total_return = (1 + valid_returns).prod() - 1
                                        ticker_summary.append({
                                            "Ticker": ticker,
                                            "Weight": f"{weights.get(ticker, 0)*100:.1f}%",
                                            "Total Return": f"{total_return*100:.2f}%",
                                            "Trading Days": len(valid_returns),
                                        })

                            if ticker_summary:
                                ticker_summary_df = pd.DataFrame(ticker_summary)
                                ticker_summary_df = ticker_summary_df.sort_values(
                                    "Total Return",
                                    ascending=False,
                                    key=lambda x: x.str.replace("%", "").astype(float)
                                )
                                st.dataframe(ticker_summary_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Could not calculate returns for tickers.")
                    else:
                        st.info("No price data available for the selected date range.")
                except Exception as e:
                    st.warning(f"Could not load ticker data: {e}")
            else:
                st.info("No holdings found for this portfolio.")


def _render_etoro_compare_tab():
    """Render the eToro Compare tab - compare portfolio against top popular investors."""
    import pandas as pd

    st.markdown("### eToro Portfolio Comparison")
    st.markdown("Compare your eToro portfolio against top popular investors.")

    # Configuration
    MY_ETORO_USERNAME = "alphawizzard"

    # Use caching to avoid excessive scraping
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_etoro_data(username: str, top_count: int):
        """Fetch eToro data with caching."""
        from ..data_sources.etoro_scraper import get_etoro_comparison_data
        return get_etoro_comparison_data(username, top_count)

    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        top_count = st.slider("Number of top investors to compare", 3, 10, 5)
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Fetch data
    with st.spinner("Fetching eToro data..."):
        try:
            data = fetch_etoro_data(MY_ETORO_USERNAME, top_count)
        except Exception as e:
            st.error(f"Failed to fetch eToro data: {e}")
            return

    my_stats = data.get('my_stats')
    top_investors = data.get('top_investors', [])
    fetched_at = data.get('fetched_at', 'Unknown')

    st.caption(f"Data fetched at: {fetched_at}")

    if not my_stats:
        st.warning(f"Could not fetch stats for {MY_ETORO_USERNAME}")
        return

    # Section 1: My Portfolio Stats
    st.markdown("---")
    st.markdown("### 📊 Your Portfolio")

    my_profile_url = f"https://www.etoro.com/people/{my_stats.username}"
    st.markdown(f"**[{my_stats.full_name} (@{my_stats.username})]({my_profile_url})**")
    st.markdown(f"Risk Score: **{my_stats.risk_score}/10** | Copiers: **{my_stats.copiers:,}**")

    # KPIs for my portfolio
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("1Y Return", f"{my_stats.gain_1y:.1f}%")
    with kpi_cols[1]:
        st.metric("2Y Return", f"{my_stats.gain_2y:.1f}%")
    with kpi_cols[2]:
        st.metric("YTD Return", f"{my_stats.gain_ytd:.1f}%")
    with kpi_cols[3]:
        st.metric("Win Ratio", f"{my_stats.win_ratio:.0f}%")
    with kpi_cols[4]:
        st.metric("Profitable Months", f"{my_stats.profitable_months_pct:.0f}%")

    # Section 2: Top Investors Comparison
    if top_investors:
        st.markdown("---")
        st.markdown(f"### 🏆 Top {len(top_investors)} Popular Investors")

        # Build comparison table
        comparison_data = []
        all_investors = [my_stats] + top_investors

        from datetime import datetime
        current_month_key = datetime.now().strftime('%Y-%m')

        for inv in all_investors:
            is_me = inv.username.lower() == MY_ETORO_USERNAME.lower()
            profile_url = f"https://www.etoro.com/people/{inv.username}"
            # Get MTD from current month's return in monthly_returns
            mtd = inv.monthly_returns.get(current_month_key, 0.0) if inv.monthly_returns else 0.0
            comparison_data.append({
                "": "⭐" if is_me else "",
                "Username": inv.username,
                "ProfileURL": profile_url,
                "Investor": f"{inv.full_name} (@{inv.username})",
                "Risk": inv.risk_score,
                "Copiers": inv.copiers,
                "MTD": mtd,
                "1Y Return": inv.gain_1y,
                "2Y Return": inv.gain_2y,
                "YTD": inv.gain_ytd,
                "Win %": inv.win_ratio,
                "Profitable Mo.": inv.profitable_months_pct,
            })

        df = pd.DataFrame(comparison_data)

        # Display comparison table
        st.markdown("#### Performance Comparison")

        # Header row
        header_cols = st.columns([0.5, 3.5, 1, 1.5, 1.2, 1.2, 1.2, 1.2, 1, 1.2])
        with header_cols[0]:
            st.write("")
        with header_cols[1]:
            st.write("**Investor**")
        with header_cols[2]:
            st.write("**Risk**")
        with header_cols[3]:
            st.write("**Copiers**")
        with header_cols[4]:
            st.write("**MTD**")
        with header_cols[5]:
            st.write("**1Y**")
        with header_cols[6]:
            st.write("**2Y**")
        with header_cols[7]:
            st.write("**YTD**")
        with header_cols[8]:
            st.write("**Win%**")
        with header_cols[9]:
            st.write("**Prof.Mo**")

        # Data rows
        for i, row in df.iterrows():
            is_me = row[""] == "⭐"

            cols = st.columns([0.5, 3.5, 1, 1.5, 1.2, 1.2, 1.2, 1.2, 1, 1.2])

            with cols[0]:
                st.write(row[""])
            with cols[1]:
                if is_me:
                    st.markdown(f"**[{row['Investor']}]({row['ProfileURL']})** (You)")
                else:
                    st.markdown(f"[{row['Investor']}]({row['ProfileURL']})")
            with cols[2]:
                st.write(f"{row['Risk']}/10")
            with cols[3]:
                st.write(f"{row['Copiers']:,}")
            with cols[4]:
                color = "green" if row["MTD"] > 0 else "red"
                st.markdown(f":{color}[{row['MTD']:.1f}%]")
            with cols[5]:
                color = "green" if row["1Y Return"] > 0 else "red"
                st.markdown(f":{color}[{row['1Y Return']:.1f}%]")
            with cols[6]:
                color = "green" if row["2Y Return"] > 0 else "red"
                st.markdown(f":{color}[{row['2Y Return']:.1f}%]")
            with cols[7]:
                color = "green" if row["YTD"] > 0 else "red"
                st.markdown(f":{color}[{row['YTD']:.1f}%]")
            with cols[8]:
                st.write(f"{row['Win %']:.0f}%")
            with cols[9]:
                st.write(f"{row['Profitable Mo.']:.0f}%")

        # Add header row explanation
        st.caption("Risk: 1-10 scale (lower = less risky) | Win %: Percentage of profitable weeks")

        # Section 3: Monthly Returns Chart
        st.markdown("---")
        st.markdown("### 📈 Monthly Returns Comparison")

        # Get monthly returns for all investors
        monthly_data = {}
        for inv in all_investors:
            if inv.monthly_returns:
                is_me = inv.username.lower() == MY_ETORO_USERNAME.lower()
                label = f"{inv.username} (You)" if is_me else inv.username
                monthly_data[label] = inv.monthly_returns

        if monthly_data:
            # Create DataFrame for monthly returns
            monthly_df = pd.DataFrame(monthly_data)
            monthly_df = monthly_df.sort_index()

            # Limit to last 12 months
            monthly_df = monthly_df.tail(12)

            if not monthly_df.empty:
                st.line_chart(monthly_df)

                # Also show as table
                with st.expander("View Monthly Returns Table"):
                    styled_monthly = monthly_df.style.format("{:.2f}%").background_gradient(
                        cmap="RdYlGn", axis=None, vmin=-10, vmax=10
                    )
                    st.dataframe(styled_monthly, use_container_width=True)
        else:
            st.info("Monthly returns data not available.")

        # Section 4: Ranking
        st.markdown("---")
        st.markdown("### 🎯 Your Ranking")

        # Calculate rankings
        all_1y_returns = [(inv.username, inv.gain_1y) for inv in all_investors]
        all_1y_returns.sort(key=lambda x: x[1], reverse=True)
        my_rank_1y = next(i+1 for i, (u, _) in enumerate(all_1y_returns) if u.lower() == MY_ETORO_USERNAME.lower())

        all_2y_returns = [(inv.username, inv.gain_2y) for inv in all_investors]
        all_2y_returns.sort(key=lambda x: x[1], reverse=True)
        my_rank_2y = next(i+1 for i, (u, _) in enumerate(all_2y_returns) if u.lower() == MY_ETORO_USERNAME.lower())

        rank_cols = st.columns(2)
        with rank_cols[0]:
            st.metric(
                "1Y Return Rank",
                f"#{my_rank_1y} of {len(all_investors)}",
                delta=f"Top {my_rank_1y}/{len(all_investors)}"
            )
        with rank_cols[1]:
            st.metric(
                "2Y Return Rank",
                f"#{my_rank_2y} of {len(all_investors)}",
                delta=f"Top {my_rank_2y}/{len(all_investors)}"
            )

    else:
        st.info("Could not fetch top investors data.")


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
