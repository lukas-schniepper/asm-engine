"""
Portfolio Selection Module

Provides optimization algorithms to select the best combination of portfolios
for trading, based on risk-adjusted metrics and diversification analysis.
"""

from .optimizer import (
    find_optimal_portfolio_combination,
    simulate_combined_portfolio,
    calculate_candidate_metrics,
    calculate_ulcer_index,
    calculate_upi,
    normalize_metrics,
    optimize_portfolio_weights,
    walk_forward_validation,
    OPTIMIZATION_PRESETS,
    DEFAULT_WEIGHTS,
    WEIGHT_METHODS,
)

__all__ = [
    "find_optimal_portfolio_combination",
    "simulate_combined_portfolio",
    "calculate_candidate_metrics",
    "calculate_ulcer_index",
    "calculate_upi",
    "normalize_metrics",
    "optimize_portfolio_weights",
    "walk_forward_validation",
    "OPTIMIZATION_PRESETS",
    "DEFAULT_WEIGHTS",
    "WEIGHT_METHODS",
]
