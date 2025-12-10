"""
Chart Components for Portfolio Tracking Dashboard.

Provides Plotly charts matching the asm-models comprehensive dashboard design.
"""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .styles import COLORS, VARIANT_COLORS, get_variant_display_name


def get_chart_layout(
    title: Optional[str] = None,
    height: int = 400,
    show_legend: bool = True,
) -> dict:
    """
    Get standardized chart layout configuration.

    Args:
        title: Chart title
        height: Chart height in pixels
        show_legend: Whether to show legend

    Returns:
        Plotly layout dict
    """
    return {
        "title": {
            "text": title,
            "font": {"size": 16, "color": COLORS["text_primary"]},
            "x": 0.02,
        } if title else None,
        "height": height,
        "margin": {"l": 60, "r": 40, "t": 80 if title else 40, "b": 50},
        "paper_bgcolor": COLORS["bg_card"],
        "plot_bgcolor": COLORS["bg_card"],
        "font": {"family": "Inter, sans-serif", "color": COLORS["text_primary"]},
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        "xaxis": {
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "tickfont": {"size": 11},
        },
        "yaxis": {
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "tickfont": {"size": 11},
        },
        "hovermode": "x unified",
    }


def create_nav_chart(
    nav_data: dict[str, pd.Series],
    title: str = "Portfolio NAV",
    normalize: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Create a NAV line chart comparing multiple variants.

    Args:
        nav_data: Dict mapping variant names to NAV Series
        title: Chart title
        normalize: If True, normalize all series to start at 100
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for variant, nav_series in nav_data.items():
        if nav_series.empty:
            continue

        # Normalize if requested
        if normalize:
            display_series = (nav_series / nav_series.iloc[0]) * 100
            y_label = "Normalized NAV (Base = 100)"
        else:
            display_series = nav_series
            y_label = "NAV"

        color = VARIANT_COLORS.get(variant, COLORS["chart_1"])
        display_name = get_variant_display_name(variant)

        fig.add_trace(go.Scatter(
            x=display_series.index,
            y=display_series.values,
            name=display_name,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate=f"{display_name}: %{{y:.2f}}<extra></extra>",
        ))

    layout = get_chart_layout(title, height)
    layout["yaxis"]["title"] = y_label
    layout["xaxis"]["title"] = "Date"
    fig.update_layout(**layout)

    return fig


def create_drawdown_chart(
    drawdown_data: dict[str, pd.Series],
    title: str = "Drawdown",
    height: int = 300,
) -> go.Figure:
    """
    Create a drawdown area chart.

    Args:
        drawdown_data: Dict mapping variant names to drawdown Series (negative values)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for variant, dd_series in drawdown_data.items():
        if dd_series.empty:
            continue

        color = VARIANT_COLORS.get(variant, COLORS["chart_1"])
        display_name = get_variant_display_name(variant)

        # Convert to percentage
        dd_pct = dd_series * 100

        fig.add_trace(go.Scatter(
            x=dd_pct.index,
            y=dd_pct.values,
            name=display_name,
            mode="lines",
            fill="tozeroy",
            line={"color": color, "width": 1.5},
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
            hovertemplate=f"{display_name}: %{{y:.2f}}%<extra></extra>",
        ))

    layout = get_chart_layout(title, height)
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["title"] = "Date"
    fig.update_layout(**layout)

    return fig


def create_allocation_chart(
    allocation_series: pd.Series,
    title: str = "Equity Allocation",
    height: int = 250,
) -> go.Figure:
    """
    Create an allocation area chart.

    Args:
        allocation_series: Series with allocation values (0 to 1)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Convert to percentage
    alloc_pct = allocation_series * 100

    fig.add_trace(go.Scatter(
        x=alloc_pct.index,
        y=alloc_pct.values,
        name="Equity",
        mode="lines",
        fill="tozeroy",
        line={"color": COLORS["primary"], "width": 2},
        fillcolor=f"rgba(37, 99, 235, 0.2)",
        hovertemplate="Equity: %{y:.1f}%<extra></extra>",
    ))

    # Add cash as complement
    cash_pct = 100 - alloc_pct
    fig.add_trace(go.Scatter(
        x=cash_pct.index,
        y=cash_pct.values,
        name="Cash",
        mode="lines",
        line={"color": COLORS["neutral"], "width": 1, "dash": "dot"},
        hovertemplate="Cash: %{y:.1f}%<extra></extra>",
    ))

    layout = get_chart_layout(title, height)
    layout["yaxis"]["title"] = "Allocation (%)"
    layout["yaxis"]["range"] = [0, 105]
    layout["yaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["title"] = "Date"
    fig.update_layout(**layout)

    return fig


def create_returns_bar_chart(
    returns_data: dict[str, float],
    title: str = "Returns Comparison",
    height: int = 300,
) -> go.Figure:
    """
    Create a bar chart comparing returns across variants.

    Args:
        returns_data: Dict mapping variant names to return values
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    variants = list(returns_data.keys())
    values = [returns_data[v] * 100 for v in variants]  # Convert to percentage
    colors = [VARIANT_COLORS.get(v, COLORS["primary"]) for v in variants]
    display_names = [get_variant_display_name(v) for v in variants]

    fig.add_trace(go.Bar(
        x=display_names,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))

    layout = get_chart_layout(title, height, show_legend=False)
    layout["yaxis"]["title"] = "Return (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return fig


def create_monthly_returns_heatmap(
    nav_series: pd.Series,
    title: str = "Monthly Returns",
    height: int = 400,
) -> go.Figure:
    """
    Create a monthly returns heatmap.

    Args:
        nav_series: NAV time series
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    # Calculate monthly returns within each month (first to last value)
    # This ensures the first month also shows a return
    monthly_returns_list = []

    for period, group in nav_series.groupby(pd.Grouper(freq="ME")):
        if len(group) >= 1:
            first_val = group.iloc[0]
            last_val = group.iloc[-1]
            if first_val > 0:
                monthly_ret = ((last_val / first_val) - 1) * 100
                monthly_returns_list.append({
                    "year": period.year,
                    "month": period.month,
                    "return": monthly_ret,
                })

    if not monthly_returns_list:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No monthly data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    df = pd.DataFrame(monthly_returns_list)

    # Pivot to year x month - ensure all 12 months are included
    pivot = df.pivot(index="year", columns="month", values="return")

    # Reindex columns to ensure all 12 months are present
    pivot = pivot.reindex(columns=range(1, 13))

    # Month names
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Ensure year labels are strings (not floats)
    year_labels = [str(int(y)) for y in pivot.index]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names,
        y=year_labels,
        colorscale=[
            [0, COLORS["negative"]],
            [0.5, COLORS["bg_secondary"]],
            [1, COLORS["positive"]],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar={
            "title": "Return (%)",
            "ticksuffix": "%",
        },
    ))

    layout = get_chart_layout(title, height, show_legend=False)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = ""
    layout["yaxis"]["autorange"] = "reversed"
    layout["yaxis"]["type"] = "category"  # Force categorical axis for years
    fig.update_layout(**layout)

    return fig


def create_signals_chart(
    signals_df: pd.DataFrame,
    signal_columns: list[str],
    title: str = "Overlay Signals",
    height: int = 400,
) -> go.Figure:
    """
    Create a multi-line chart for overlay signals.

    Args:
        signals_df: DataFrame with signal data
        signal_columns: List of column names to plot
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=len(signal_columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=signal_columns,
    )

    colors = [COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
              COLORS["chart_4"], COLORS["chart_5"], COLORS["chart_6"]]

    for i, col in enumerate(signal_columns):
        if col not in signals_df.columns:
            continue

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df[col],
                name=col,
                mode="lines",
                line={"color": color, "width": 1.5},
            ),
            row=i + 1,
            col=1,
        )

    layout = get_chart_layout(title, height, show_legend=False)
    fig.update_layout(**layout)

    return fig


def create_comparison_chart(
    data: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    height: int = 300,
) -> go.Figure:
    """
    Create a grouped bar chart comparing a metric across variants and periods.

    Args:
        data: DataFrame with variants as columns, periods as index
        metric: Metric being compared (for labeling)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for col in data.columns:
        color = VARIANT_COLORS.get(col, COLORS["primary"])
        display_name = get_variant_display_name(col)

        values = data[col] * 100  # Convert to percentage

        fig.add_trace(go.Bar(
            x=data.index,
            y=values,
            name=display_name,
            marker_color=color,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))

    if title is None:
        title = f"{metric.replace('_', ' ').title()} by Period"

    layout = get_chart_layout(title, height)
    layout["barmode"] = "group"
    layout["yaxis"]["title"] = f"{metric.replace('_', ' ').title()} (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return fig


# =============================================================================
# INSTITUTIONAL RISK CHARTS
# =============================================================================


def create_var_histogram(
    returns: pd.Series,
    var_95: float,
    var_99: float,
    title: str = "Return Distribution with VaR",
    height: int = 350,
) -> go.Figure:
    """
    Create a return distribution histogram with VaR lines.

    Args:
        returns: Daily returns series
        var_95: 95% VaR value
        var_99: 99% VaR value
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    import numpy as np

    fig = go.Figure()

    # Histogram of returns
    returns_pct = returns * 100  # Convert to percentage

    fig.add_trace(go.Histogram(
        x=returns_pct,
        nbinsx=50,
        name="Daily Returns",
        marker_color=COLORS["primary"],
        opacity=0.7,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))

    # VaR lines
    fig.add_vline(
        x=var_95 * 100,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"95% VaR: {var_95*100:.2f}%",
        annotation_position="top left",
    )

    fig.add_vline(
        x=var_99 * 100,
        line_dash="dash",
        line_color=COLORS["negative"],
        annotation_text=f"99% VaR: {var_99*100:.2f}%",
        annotation_position="top left",
    )

    layout = get_chart_layout(title, height, show_legend=False)
    layout["xaxis"]["title"] = "Daily Return (%)"
    layout["xaxis"]["ticksuffix"] = "%"
    layout["yaxis"]["title"] = "Frequency"
    fig.update_layout(**layout)

    return fig


def create_scatter_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    beta: float,
    alpha: float,
    title: str = "Portfolio vs Benchmark Returns",
    height: int = 400,
) -> go.Figure:
    """
    Create a scatter plot with regression line for beta/alpha visualization.

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        beta: Calculated beta value
        alpha: Calculated alpha value (annualized)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    import numpy as np

    # Align returns
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['portfolio', 'benchmark']

    port_pct = aligned['portfolio'] * 100
    bench_pct = aligned['benchmark'] * 100

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=bench_pct,
        y=port_pct,
        mode="markers",
        name="Daily Returns",
        marker=dict(
            color=COLORS["primary"],
            size=4,
            opacity=0.5,
        ),
        hovertemplate="Benchmark: %{x:.2f}%<br>Portfolio: %{y:.2f}%<extra></extra>",
    ))

    # Regression line
    x_range = np.linspace(bench_pct.min(), bench_pct.max(), 100)
    daily_alpha = alpha / 252 * 100  # Convert annualized alpha to daily %
    y_line = daily_alpha + beta * x_range

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_line,
        mode="lines",
        name=f"β={beta:.2f}, α={alpha*100:.1f}%/yr",
        line=dict(color=COLORS["negative"], width=2),
    ))

    # Zero lines
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["border"])
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["border"])

    layout = get_chart_layout(title, height)
    layout["xaxis"]["title"] = "Benchmark Return (%)"
    layout["yaxis"]["title"] = "Portfolio Return (%)"
    layout["xaxis"]["ticksuffix"] = "%"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return fig


def create_rolling_metrics_chart(
    metrics_dict: dict[str, pd.Series],
    title: str = "Rolling Metrics",
    height: int = 350,
) -> go.Figure:
    """
    Create a multi-line chart for rolling metrics.

    Args:
        metrics_dict: Dict mapping metric names to series
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    colors = [COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
              COLORS["chart_4"], COLORS["chart_5"]]

    for i, (name, series) in enumerate(metrics_dict.items()):
        if series.empty:
            continue

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=name,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        ))

    layout = get_chart_layout(title, height)
    layout["xaxis"]["title"] = "Date"
    fig.update_layout(**layout)

    return fig


def create_drawdown_highlight_chart(
    drawdown_series: pd.Series,
    worst_drawdowns: pd.DataFrame,
    title: str = "Drawdown Analysis",
    height: int = 350,
) -> go.Figure:
    """
    Create a drawdown chart with worst periods highlighted.

    Args:
        drawdown_series: Drawdown series (negative values)
        worst_drawdowns: DataFrame with worst drawdowns info
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Main drawdown area
    dd_pct = drawdown_series * 100

    fig.add_trace(go.Scatter(
        x=dd_pct.index,
        y=dd_pct.values,
        name="Drawdown",
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["negative"], width=1.5),
        fillcolor="rgba(239, 68, 68, 0.2)",
        hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
    ))

    # Highlight worst drawdown periods
    if not worst_drawdowns.empty:
        for i, row in worst_drawdowns.head(3).iterrows():
            peak_date = row['peak_date']
            recovery_date = row.get('recovery_date', None) or dd_pct.index[-1]

            fig.add_vrect(
                x0=peak_date,
                x1=recovery_date,
                fillcolor="rgba(239, 68, 68, 0.1)",
                line_width=0,
                annotation_text=f"#{i+1}: {row['drawdown_pct']*100:.1f}%",
                annotation_position="top left",
            )

    layout = get_chart_layout(title, height, show_legend=False)
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["title"] = "Date"
    fig.update_layout(**layout)

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Portfolio Correlation Matrix",
    height: int = 400,
) -> go.Figure:
    """
    Create a correlation heatmap for multiple portfolios.

    Args:
        correlation_matrix: DataFrame with correlation values
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    # Clean names for display
    clean_names = [name.replace("_EqualWeight", "").replace("_", " ")
                   for name in correlation_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=clean_names,
        y=clean_names,
        colorscale=[
            [0, COLORS["negative"]],
            [0.5, COLORS["bg_secondary"]],
            [1, COLORS["positive"]],
        ],
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in correlation_matrix.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>",
        colorbar={"title": "Correlation"},
    ))

    layout = get_chart_layout(title, height, show_legend=False)
    layout["xaxis"]["title"] = ""
    layout["yaxis"]["title"] = ""
    fig.update_layout(**layout)

    return fig


def create_efficient_frontier_scatter(
    portfolios: list[dict],
    title: str = "Risk-Return Profile",
    height: int = 400,
) -> go.Figure:
    """
    Create a scatter plot of portfolios on risk-return space.

    Args:
        portfolios: List of dicts with 'name', 'volatility', 'return', optionally 'sharpe'
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for portfolio in portfolios:
        name = portfolio.get('name', 'Portfolio')
        vol = portfolio.get('volatility', 0) * 100
        ret = portfolio.get('return', 0) * 100
        sharpe = portfolio.get('sharpe', None)

        hover_text = f"{name}<br>Return: {ret:.1f}%<br>Volatility: {vol:.1f}%"
        if sharpe is not None:
            hover_text += f"<br>Sharpe: {sharpe:.2f}"

        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode="markers+text",
            name=name.replace("_EqualWeight", "").replace("_", " "),
            marker=dict(size=12),
            text=[name.replace("_EqualWeight", "").replace("_", " ")[:15]],
            textposition="top center",
            hovertemplate=hover_text + "<extra></extra>",
        ))

    layout = get_chart_layout(title, height)
    layout["xaxis"]["title"] = "Annualized Volatility (%)"
    layout["yaxis"]["title"] = "Annualized Return (%)"
    layout["xaxis"]["ticksuffix"] = "%"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return fig
