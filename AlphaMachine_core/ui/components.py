"""
Reusable UI Components for Portfolio Tracking Dashboard.

Provides HTML/Streamlit components that match the asm-models dashboard design.
"""

import pandas as pd
from typing import Optional

from .styles import (
    COLORS,
    VARIANT_COLORS,
    format_percentage,
    format_number,
    format_ratio,
    get_value_class,
    get_variant_display_name,
    get_period_display_name,
)


def render_kpi_card(
    label: str,
    value: str,
    subtitle: Optional[str] = None,
    value_class: str = "",
) -> str:
    """
    Render a single KPI card as HTML.

    Args:
        label: KPI label/title
        value: Formatted value string
        subtitle: Optional subtitle text
        value_class: CSS class for value (positive/negative)

    Returns:
        HTML string for the KPI card
    """
    subtitle_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""

    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {value_class}">{value}</div>
        {subtitle_html}
    </div>
    """


def render_kpi_grid(kpis: list[dict]) -> str:
    """
    Render a grid of KPI cards.

    Args:
        kpis: List of dicts with keys: label, value, subtitle (optional), value_class (optional)

    Returns:
        HTML string for the KPI grid
    """
    cards = [
        render_kpi_card(
            label=kpi["label"],
            value=kpi["value"],
            subtitle=kpi.get("subtitle"),
            value_class=kpi.get("value_class", ""),
        )
        for kpi in kpis
    ]

    return f"""
    <div class="kpi-grid">
        {"".join(cards)}
    </div>
    """


def render_metrics_table(
    metrics: dict,
    title: Optional[str] = None,
) -> str:
    """
    Render a metrics table with formatted values.

    Args:
        metrics: Dict mapping metric names to values
        title: Optional table title

    Returns:
        HTML string for the table
    """
    title_html = f"<h3>{title}</h3>" if title else ""

    rows = []
    for name, value in metrics.items():
        # Determine formatting based on metric name
        if "return" in name.lower() or "cagr" in name.lower() or "volatility" in name.lower():
            formatted = format_percentage(value)
            value_class = get_value_class(value)
        elif "drawdown" in name.lower():
            formatted = format_percentage(value)
            value_class = get_value_class(value, invert=True)
        elif "ratio" in name.lower() or "rate" in name.lower():
            formatted = format_ratio(value)
            value_class = get_value_class(value)
        else:
            formatted = format_number(value)
            value_class = ""

        rows.append(f"""
            <tr>
                <td>{name}</td>
                <td class="value-{value_class}" style="text-align: right; font-weight: 600;">
                    {formatted}
                </td>
            </tr>
        """)

    return f"""
    {title_html}
    <table class="metrics-table">
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def render_comparison_table(
    data: pd.DataFrame,
    metric_names: Optional[dict] = None,
    highlight_best: bool = True,
) -> str:
    """
    Render a comparison table with variants as columns and metrics as rows.

    Args:
        data: DataFrame with variants as columns and metrics as index
        metric_names: Optional dict mapping metric keys to display names
        highlight_best: Whether to highlight best values

    Returns:
        HTML string for the comparison table
    """
    if metric_names is None:
        metric_names = {
            "total_return": "Total Return",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "max_drawdown": "Max Drawdown",
            "calmar_ratio": "Calmar Ratio",
            "volatility": "Volatility",
            "win_rate": "Win Rate",
        }

    # Header row
    headers = ['<th class="row-header">Metric</th>']
    for col in data.columns:
        display_name = get_variant_display_name(col)
        color = VARIANT_COLORS.get(col, COLORS["primary"])
        headers.append(
            f'<th style="background: {color};">{display_name}</th>'
        )

    # Data rows
    rows = []
    for metric in data.index:
        row_values = data.loc[metric]
        display_name = metric_names.get(metric, metric.replace("_", " ").title())

        # Determine if higher or lower is better
        lower_is_better = metric in ["max_drawdown", "volatility"]

        # Find best value
        if lower_is_better:
            best_val = row_values.min()
        else:
            best_val = row_values.max()

        cells = [f'<td class="row-header">{display_name}</td>']
        for col in data.columns:
            val = row_values[col]

            # Format based on metric type
            if metric in ["total_return", "cagr", "max_drawdown", "volatility", "win_rate"]:
                formatted = format_percentage(val)
                value_class = get_value_class(val, invert=(metric == "max_drawdown"))
            else:
                formatted = format_ratio(val)
                value_class = get_value_class(val)

            # Highlight best
            best_class = "best-value" if highlight_best and val == best_val else ""

            cells.append(
                f'<td class="{best_class}" style="color: {COLORS["positive"] if value_class == "positive" else COLORS["negative"] if value_class == "negative" else "inherit"};">'
                f'{formatted}</td>'
            )

        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <table class="comparison-table">
        <thead>
            <tr>{"".join(headers)}</tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def render_section_header(title: str, icon: str = "") -> str:
    """
    Render a section header with optional icon.

    Args:
        title: Section title
        icon: Optional emoji/icon

    Returns:
        HTML string for the section header
    """
    icon_html = f'<span class="icon">{icon}</span>' if icon else ""
    return f"""
    <div class="section-header">
        {icon_html}
        <h2>{title}</h2>
    </div>
    """


def render_variant_badge(variant: str) -> str:
    """
    Render a styled badge for a variant.

    Args:
        variant: Variant name

    Returns:
        HTML string for the badge
    """
    display_name = get_variant_display_name(variant)
    return f'<span class="variant-badge {variant}">{display_name}</span>'


def render_info_card(
    message: str,
    card_type: str = "info",
) -> str:
    """
    Render an info/warning/error card.

    Args:
        message: Card message
        card_type: "info", "warning", "success", or "error"

    Returns:
        HTML string for the card
    """
    return f"""
    <div class="info-card {card_type}">
        {message}
    </div>
    """


def render_legend(items: list[dict]) -> str:
    """
    Render a chart legend.

    Args:
        items: List of dicts with keys: label, color

    Returns:
        HTML string for the legend
    """
    legend_items = [
        f"""
        <div class="legend-item">
            <div class="legend-color" style="background: {item['color']};"></div>
            <span>{item['label']}</span>
        </div>
        """
        for item in items
    ]

    return f"""
    <div class="chart-legend">
        {"".join(legend_items)}
    </div>
    """


def format_metrics_for_display(metrics: dict) -> list[dict]:
    """
    Convert raw metrics dict to list of KPI card data.

    Args:
        metrics: Dict with metric values

    Returns:
        List of dicts ready for render_kpi_grid
    """
    kpis = []

    if "total_return" in metrics:
        val = metrics["total_return"]
        kpis.append({
            "label": "Total Return",
            "value": format_percentage(val),
            "value_class": get_value_class(val),
        })

    if "cagr" in metrics:
        val = metrics["cagr"]
        kpis.append({
            "label": "CAGR",
            "value": format_percentage(val),
            "value_class": get_value_class(val),
        })

    if "sharpe_ratio" in metrics:
        val = metrics["sharpe_ratio"]
        kpis.append({
            "label": "Sharpe Ratio",
            "value": format_ratio(val),
            "value_class": get_value_class(val),
        })

    if "sortino_ratio" in metrics:
        val = metrics["sortino_ratio"]
        kpis.append({
            "label": "Sortino Ratio",
            "value": format_ratio(val),
            "value_class": get_value_class(val),
        })

    if "max_drawdown" in metrics:
        val = metrics["max_drawdown"]
        kpis.append({
            "label": "Max Drawdown",
            "value": format_percentage(val),
            "value_class": get_value_class(val, invert=True),
        })

    if "calmar_ratio" in metrics:
        val = metrics["calmar_ratio"]
        kpis.append({
            "label": "Calmar Ratio",
            "value": format_ratio(val),
            "value_class": get_value_class(val),
        })

    if "information_ratio" in metrics:
        val = metrics["information_ratio"]
        kpis.append({
            "label": "Info Ratio",
            "value": format_ratio(val),
            "value_class": get_value_class(val),
            "subtitle": "vs SPY",
        })

    if "volatility" in metrics:
        val = metrics["volatility"]
        kpis.append({
            "label": "Volatility",
            "value": format_percentage(val),
            "subtitle": "Annualized",
        })

    if "win_rate" in metrics:
        val = metrics["win_rate"]
        kpis.append({
            "label": "Win Rate",
            "value": format_percentage(val),
            "subtitle": "Positive days",
        })

    return kpis
