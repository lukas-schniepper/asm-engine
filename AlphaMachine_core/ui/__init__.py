"""
UI Components for Portfolio Tracking Dashboard.

This module provides reusable UI components matching the styling
from the asm-models comprehensive dashboard.
"""

from .styles import get_dashboard_css, COLORS, format_percentage, format_number
from .components import (
    render_kpi_card,
    render_kpi_grid,
    render_metrics_table,
    render_comparison_table,
)
from .charts import (
    create_nav_chart,
    create_drawdown_chart,
    create_allocation_chart,
    create_returns_bar_chart,
)

__all__ = [
    # Styles
    "get_dashboard_css",
    "COLORS",
    "format_percentage",
    "format_number",
    # Components
    "render_kpi_card",
    "render_kpi_grid",
    "render_metrics_table",
    "render_comparison_table",
    # Charts
    "create_nav_chart",
    "create_drawdown_chart",
    "create_allocation_chart",
    "create_returns_bar_chart",
]
