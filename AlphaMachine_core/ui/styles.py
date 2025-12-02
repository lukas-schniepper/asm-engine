"""
Dashboard Styles for Portfolio Tracking.

CSS and styling utilities matching the asm-models comprehensive dashboard design.
"""

from typing import Optional


# Color palette from comprehensive dashboard
COLORS = {
    # Primary colors
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "primary_light": "#60a5fa",

    # Status colors
    "positive": "#10b981",
    "positive_dark": "#059669",
    "positive_light": "#34d399",
    "negative": "#ef4444",
    "negative_dark": "#dc2626",
    "negative_light": "#f87171",
    "warning": "#f59e0b",
    "neutral": "#6b7280",

    # Background colors
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8fafc",
    "bg_dark": "#1e293b",
    "bg_card": "#ffffff",

    # Text colors
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "text_light": "#ffffff",

    # Border colors
    "border": "#e2e8f0",
    "border_dark": "#cbd5e1",

    # Chart colors (ordered for consistency)
    "chart_1": "#2563eb",  # Raw
    "chart_2": "#10b981",  # Conservative
    "chart_3": "#8b5cf6",  # TrendRegimeV2
    "chart_4": "#f59e0b",  # SPY
    "chart_5": "#ef4444",
    "chart_6": "#06b6d4",
}

# Variant-specific colors
VARIANT_COLORS = {
    "raw": "#2563eb",
    "conservative": "#10b981",
    "trend_regime_v2": "#8b5cf6",
    "spy": "#f59e0b",
}


def get_dashboard_css() -> str:
    """
    Return CSS for the Performance Tracker dashboard.

    Matches the styling from the asm-models comprehensive dashboard.
    """
    return f"""
    <style>
    /* ===== Base Styles ===== */
    .dashboard-container {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {COLORS["text_primary"]};
    }}

    /* ===== KPI Cards ===== */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}

    .kpi-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }}

    .kpi-card:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    .kpi-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {COLORS["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}

    .kpi-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        line-height: 1.2;
    }}

    .kpi-value.positive {{
        color: {COLORS["positive"]};
    }}

    .kpi-value.negative {{
        color: {COLORS["negative"]};
    }}

    .kpi-subtitle {{
        font-size: 0.75rem;
        color: {COLORS["text_muted"]};
        margin-top: 0.25rem;
    }}

    /* ===== Section Headers ===== */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid {COLORS["primary"]};
    }}

    .section-header h2 {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {COLORS["text_primary"]};
        margin: 0;
    }}

    .section-header .icon {{
        font-size: 1.5rem;
    }}

    /* ===== Data Tables ===== */
    .metrics-table {{
        width: 100%;
        border-collapse: collapse;
        background: {COLORS["bg_card"]};
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    .metrics-table th {{
        background: {COLORS["bg_secondary"]};
        padding: 0.75rem 1rem;
        text-align: left;
        font-size: 0.75rem;
        font-weight: 600;
        color: {COLORS["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid {COLORS["border"]};
    }}

    .metrics-table td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid {COLORS["border"]};
        font-size: 0.875rem;
    }}

    .metrics-table tr:last-child td {{
        border-bottom: none;
    }}

    .metrics-table tr:hover {{
        background: {COLORS["bg_secondary"]};
    }}

    .metrics-table .value-positive {{
        color: {COLORS["positive"]};
        font-weight: 600;
    }}

    .metrics-table .value-negative {{
        color: {COLORS["negative"]};
        font-weight: 600;
    }}

    /* ===== Comparison Table ===== */
    .comparison-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }}

    .comparison-table th {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
        color: white;
        padding: 1rem;
        text-align: center;
        font-weight: 600;
    }}

    .comparison-table th.row-header {{
        background: {COLORS["bg_secondary"]};
        color: {COLORS["text_primary"]};
        text-align: left;
    }}

    .comparison-table td {{
        padding: 0.75rem 1rem;
        text-align: center;
        border-bottom: 1px solid {COLORS["border"]};
    }}

    .comparison-table td.row-header {{
        text-align: left;
        font-weight: 500;
        background: {COLORS["bg_secondary"]};
    }}

    .comparison-table .best-value {{
        background: rgba(16, 185, 129, 0.1);
        font-weight: 700;
    }}

    /* ===== Variant Badges ===== */
    .variant-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    .variant-badge.raw {{
        background: rgba(37, 99, 235, 0.1);
        color: {COLORS["chart_1"]};
    }}

    .variant-badge.conservative {{
        background: rgba(16, 185, 129, 0.1);
        color: {COLORS["chart_2"]};
    }}

    .variant-badge.trend_regime_v2 {{
        background: rgba(139, 92, 246, 0.1);
        color: {COLORS["chart_3"]};
    }}

    /* ===== Period Selector Styles ===== */
    .period-selector {{
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }}

    .period-btn {{
        padding: 0.5rem 1rem;
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        background: {COLORS["bg_card"]};
        color: {COLORS["text_secondary"]};
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s;
    }}

    .period-btn:hover {{
        border-color: {COLORS["primary"]};
        color: {COLORS["primary"]};
    }}

    .period-btn.active {{
        background: {COLORS["primary"]};
        border-color: {COLORS["primary"]};
        color: white;
    }}

    /* ===== Info Cards ===== */
    .info-card {{
        background: {COLORS["bg_secondary"]};
        border-left: 4px solid {COLORS["primary"]};
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }}

    .info-card.warning {{
        border-left-color: {COLORS["warning"]};
    }}

    .info-card.success {{
        border-left-color: {COLORS["positive"]};
    }}

    .info-card.error {{
        border-left-color: {COLORS["negative"]};
    }}

    /* ===== Legend ===== */
    .chart-legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
        padding: 0.75rem;
        background: {COLORS["bg_secondary"]};
        border-radius: 6px;
    }}

    .legend-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
    }}

    .legend-color {{
        width: 12px;
        height: 12px;
        border-radius: 3px;
    }}

    /* ===== Responsive adjustments ===== */
    @media (max-width: 768px) {{
        .kpi-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}

        .kpi-value {{
            font-size: 1.5rem;
        }}
    }}
    </style>
    """


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as percentage string."""
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    if value is None:
        return "-"
    return f"{value:,.{decimals}f}"


def format_ratio(value: float, decimals: int = 2) -> str:
    """Format a ratio value."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def get_value_class(value: float, invert: bool = False) -> str:
    """
    Get CSS class based on value sign.

    Args:
        value: The numeric value
        invert: If True, negative is positive (e.g., for drawdown)
    """
    if value is None:
        return ""
    if invert:
        return "positive" if value < 0 else "negative" if value > 0 else ""
    return "positive" if value > 0 else "negative" if value < 0 else ""


def get_variant_display_name(variant: str) -> str:
    """Get display name for a variant."""
    names = {
        "raw": "Raw (100% Equity)",
        "conservative": "Conservative Model",
        "trend_regime_v2": "Trend Regime V2.0",
        "spy": "SPY Benchmark",
    }
    return names.get(variant, variant.replace("_", " ").title())


def get_period_display_name(period: str) -> str:
    """Get display name for a period."""
    names = {
        "week": "1 Week",
        "month": "1 Month",
        "quarter": "3 Months",
        "year": "1 Year",
        "ytd": "YTD",
        "all": "All Time",
    }
    return names.get(period, period.title())
