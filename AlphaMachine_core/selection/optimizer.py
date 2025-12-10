"""
Portfolio Selection Optimizer

Provides algorithms to find optimal combinations of portfolios based on
risk-adjusted metrics and diversification analysis.

Key functions:
- find_optimal_portfolio_combination(): Main optimizer
- simulate_combined_portfolio(): Simulate combined portfolio performance
- calculate_candidate_metrics(): Calculate all metrics for a portfolio
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..tracking.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_returns,
    calculate_rolling_sharpe,
    TRADING_DAYS_PER_YEAR,
)


# =============================================================================
# OPTIMIZATION PRESETS
# =============================================================================

OPTIMIZATION_PRESETS = {
    "Risk-Adjusted Focus": {
        "sharpe": 3, "sortino": 2, "calmar": 1, "upi": 2,
        "cagr": 1, "max_dd": 2, "volatility": 1, "correlation": 2,
    },
    "Absolute Returns": {
        "sharpe": 1, "sortino": 1, "calmar": 1, "upi": 1,
        "cagr": 3, "max_dd": 1, "volatility": 0, "correlation": 2,
    },
    "Capital Preservation": {
        "sharpe": 1, "sortino": 2, "calmar": 2, "upi": 3,
        "cagr": 0, "max_dd": 3, "volatility": 3, "correlation": 1,
    },
    "Max Diversification": {
        "sharpe": 1, "sortino": 0, "calmar": 0, "upi": 1,
        "cagr": 1, "max_dd": 1, "volatility": 1, "correlation": 3,
    },
}

DEFAULT_WEIGHTS = {
    "sharpe": 2, "sortino": 2, "calmar": 1, "upi": 2,
    "cagr": 2, "max_dd": 2, "volatility": 1, "correlation": 2,
}


# =============================================================================
# ULCER INDEX & UPI
# =============================================================================


def calculate_ulcer_index(nav: pd.Series) -> float:
    """
    Calculate Ulcer Index - measures depth and duration of drawdowns.

    Unlike volatility which penalizes both up and down moves equally,
    the Ulcer Index only measures downside pain.

    Formula: UI = sqrt(mean(drawdown_pct²))

    Args:
        nav: NAV time series

    Returns:
        Ulcer Index (higher = more drawdown pain)
    """
    if len(nav) < 2:
        return 0.0

    cummax = nav.cummax()
    # Percentage drawdown (negative values become positive in squared)
    drawdown_pct = ((nav - cummax) / cummax) * 100

    ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))

    return float(ulcer_index) if not np.isnan(ulcer_index) else 0.0


def calculate_upi(nav: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Ulcer Performance Index.

    UPI = (CAGR - Risk Free Rate) / Ulcer Index

    Similar to Sharpe but uses Ulcer Index instead of volatility,
    making it more suitable for investors who care about drawdowns.

    Args:
        nav: NAV time series
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Ulcer Performance Index (higher = better risk-adjusted return)
    """
    if len(nav) < 2:
        return 0.0

    cagr = calculate_cagr(nav)
    ulcer = calculate_ulcer_index(nav)

    if ulcer == 0 or np.isnan(ulcer):
        # No drawdowns - return high value if positive CAGR
        return float(cagr * 100) if cagr > 0 else 0.0

    return float((cagr - risk_free_rate) / (ulcer / 100))  # Ulcer is in %, convert back


# =============================================================================
# CANDIDATE METRICS
# =============================================================================


def calculate_candidate_metrics(
    nav: pd.Series,
    returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate all metrics for a portfolio candidate.

    Args:
        nav: NAV time series
        returns: Pre-calculated returns (optional, will calculate if not provided)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with all metrics
    """
    if len(nav) < 2:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "upi": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "volatility": 0.0,
        }

    if returns is None:
        returns = calculate_returns(nav)

    return {
        "sharpe": calculate_sharpe_ratio(returns, risk_free_rate),
        "sortino": calculate_sortino_ratio(returns, risk_free_rate),
        "calmar": calculate_calmar_ratio(nav),
        "upi": calculate_upi(nav, risk_free_rate),
        "cagr": calculate_cagr(nav),
        "max_dd": calculate_max_drawdown(nav),  # Returns negative value
        "volatility": calculate_volatility(returns),
    }


# =============================================================================
# NORMALIZATION
# =============================================================================


def normalize_metrics(
    all_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Normalize metrics to 0-1 scale for fair comparison.

    Higher normalized value = better.
    For inverted metrics (max_dd, volatility), lower raw value = higher normalized value.

    Args:
        all_metrics: Dictionary of {candidate_name: {metric_name: value}}

    Returns:
        Dictionary with normalized metrics (0-1 scale)
    """
    if not all_metrics:
        return {}

    # Get all metric names
    metric_names = list(next(iter(all_metrics.values())).keys())

    # Metrics where LOWER is better (need to invert)
    inverted_metrics = {"max_dd", "volatility"}

    normalized = {name: {} for name in all_metrics}

    for metric in metric_names:
        values = [m[metric] for m in all_metrics.values()]

        # Handle max_dd which is negative (more negative = worse)
        if metric == "max_dd":
            # Convert to positive (absolute value) for normalization
            values = [abs(v) for v in values]

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        for name, metrics in all_metrics.items():
            raw_value = metrics[metric]

            # Handle max_dd conversion
            if metric == "max_dd":
                raw_value = abs(raw_value)

            if range_val == 0:
                # All values are the same
                norm_value = 0.5
            else:
                norm_value = (raw_value - min_val) / range_val

            # Invert for metrics where lower is better
            if metric in inverted_metrics:
                norm_value = 1 - norm_value

            normalized[name][metric] = norm_value

    return normalized


# =============================================================================
# COMBINATION SCORING
# =============================================================================


def score_combination(
    combo: Tuple[str, ...],
    normalized_metrics: Dict[str, Dict[str, float]],
    corr_matrix: pd.DataFrame,
    weights: Dict[str, float],
) -> Dict:
    """
    Score a portfolio combination.

    Args:
        combo: Tuple of portfolio names in the combination
        normalized_metrics: Normalized metrics for all candidates
        corr_matrix: Correlation matrix of all candidates
        weights: Metric weights (0-3 scale)

    Returns:
        Dictionary with scoring breakdown
    """
    n = len(combo)

    # Individual metric scores (excluding correlation)
    metric_weights = {k: v for k, v in weights.items() if k != "correlation"}
    total_weight = sum(metric_weights.values())

    if total_weight == 0:
        total_weight = 1  # Avoid division by zero

    individual_scores = []
    for name in combo:
        metrics = normalized_metrics[name]
        score = sum(
            weights.get(m, 0) * metrics.get(m, 0)
            for m in metric_weights
        )
        individual_scores.append(score / total_weight)

    avg_individual_score = np.mean(individual_scores)

    # Average pairwise correlation
    corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = corr_matrix.loc[combo[i], combo[j]]
            corr_pairs.append(corr_val)

    avg_correlation = np.mean(corr_pairs) if corr_pairs else 0

    # Correlation penalty (higher correlation = lower score)
    corr_weight = weights.get("correlation", 0)
    # Normalize correlation score: 0 correlation = full bonus, 1 correlation = no bonus
    corr_bonus = (1 - avg_correlation) * (corr_weight / 3)  # Scale by weight/max_weight

    # Total score
    total_score = avg_individual_score + corr_bonus

    # Estimate combined Sharpe using diversification formula
    # SR_combined ~ SR_avg * sqrt(n) / sqrt(1 + (n-1) * avg_corr)
    avg_sharpe = np.mean([
        normalized_metrics[name].get("sharpe", 0) for name in combo
    ])
    if avg_correlation < 0.99:
        diversification_factor = np.sqrt(n) / np.sqrt(1 + (n - 1) * avg_correlation)
    else:
        diversification_factor = 1.0

    return {
        "combination": combo,
        "total_score": total_score,
        "avg_individual_score": avg_individual_score,
        "avg_correlation": avg_correlation,
        "corr_bonus": corr_bonus,
        "diversification_factor": diversification_factor,
        "estimated_combined_sharpe_improvement": diversification_factor,
    }


# =============================================================================
# MAIN OPTIMIZER
# =============================================================================


def find_optimal_portfolio_combination(
    portfolio_returns: Dict[str, pd.Series],
    n_select: int = 3,
    weights: Optional[Dict[str, float]] = None,
    min_data_points: int = 30,
) -> Dict:
    """
    Find optimal combination of N portfolios.

    This is the main optimization function that:
    1. Calculates metrics for each candidate
    2. Normalizes metrics for fair comparison
    3. Builds correlation matrix
    4. Evaluates all C(n, n_select) combinations
    5. Returns top recommendations

    Args:
        portfolio_returns: Dictionary mapping candidate names to daily returns Series
        n_select: Number of portfolios to select (default 3)
        weights: Metric weights (default: DEFAULT_WEIGHTS)
        min_data_points: Minimum data points required (default 30)

    Returns:
        Dictionary with:
        - recommended: Top combination with details
        - alternatives: Next 4 best combinations
        - all_scores: All combination scores (sorted)
        - portfolio_metrics: Raw metrics for each portfolio
        - normalized_metrics: Normalized metrics
        - correlation_matrix: Full correlation matrix
        - candidates_count: Number of valid candidates
        - combinations_evaluated: Number of combinations scored
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Filter candidates with sufficient data
    valid_candidates = {
        name: returns for name, returns in portfolio_returns.items()
        if len(returns.dropna()) >= min_data_points
    }

    if len(valid_candidates) < n_select:
        return {
            "error": f"Not enough candidates with {min_data_points}+ data points. "
                     f"Found {len(valid_candidates)}, need {n_select}.",
            "candidates_count": len(valid_candidates),
            "combinations_evaluated": 0,
        }

    # Step 1: Calculate metrics for each candidate
    portfolio_metrics = {}
    portfolio_navs = {}

    for name, returns in valid_candidates.items():
        # Build NAV from returns
        nav = (1 + returns).cumprod() * 100
        nav = nav.dropna()
        portfolio_navs[name] = nav
        portfolio_metrics[name] = calculate_candidate_metrics(nav, returns)

    # Step 2: Normalize metrics
    normalized_metrics = normalize_metrics(portfolio_metrics)

    # Step 3: Build correlation matrix
    returns_df = pd.DataFrame(valid_candidates)
    returns_df = returns_df.dropna()
    corr_matrix = returns_df.corr()

    # Step 4: Generate and score all combinations
    all_combos = list(combinations(valid_candidates.keys(), n_select))

    scores = []
    for combo in all_combos:
        score = score_combination(combo, normalized_metrics, corr_matrix, weights)
        scores.append(score)

    # Step 5: Sort by total score descending
    scores.sort(key=lambda x: x["total_score"], reverse=True)

    # Enrich top results with raw metrics
    for score in scores[:5]:
        combo = score["combination"]
        score["portfolio_details"] = {
            name: portfolio_metrics[name] for name in combo
        }

    return {
        "recommended": scores[0] if scores else None,
        "alternatives": scores[1:5] if len(scores) > 1 else [],
        "all_scores": scores,
        "portfolio_metrics": portfolio_metrics,
        "normalized_metrics": normalized_metrics,
        "correlation_matrix": corr_matrix,
        "candidates_count": len(valid_candidates),
        "combinations_evaluated": len(all_combos),
    }


# =============================================================================
# COMBINED PORTFOLIO SIMULATION
# =============================================================================


def simulate_combined_portfolio(
    portfolio_returns: Dict[str, pd.Series],
    selected: List[str],
    weights: Optional[List[float]] = None,
    base_nav: float = 100.0,
) -> Dict:
    """
    Simulate combined portfolio performance.

    Args:
        portfolio_returns: Daily returns by portfolio name
        selected: List of selected portfolio names
        weights: Portfolio weights (default: equal weight)
        base_nav: Starting NAV value (default 100)

    Returns:
        Dictionary with:
        - nav: Combined NAV series
        - returns: Combined daily returns series
        - metrics: Performance metrics of combined portfolio
        - weights: Applied weights
    """
    if not selected:
        return {"error": "No portfolios selected"}

    if weights is None:
        weights = [1.0 / len(selected)] * len(selected)

    # Validate weights sum to 1
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.001:
        weights = [w / weight_sum for w in weights]

    # Align returns to common dates
    returns_df = pd.DataFrame({
        name: portfolio_returns[name] for name in selected
        if name in portfolio_returns
    })
    returns_df = returns_df.dropna()

    if len(returns_df) < 2:
        return {"error": "Insufficient aligned data points"}

    # Calculate weighted returns
    combined_returns = pd.Series(0.0, index=returns_df.index)
    for name, weight in zip(selected, weights):
        combined_returns += returns_df[name] * weight

    # Build NAV series
    combined_nav = (1 + combined_returns).cumprod() * base_nav

    # Calculate metrics
    metrics = calculate_candidate_metrics(combined_nav, combined_returns)

    # Calculate rolling Sharpe for robustness analysis
    rolling_sharpe = calculate_rolling_sharpe(combined_returns, window=60)
    sharpe_stability = rolling_sharpe.std() if len(rolling_sharpe) > 0 else 0.0

    return {
        "nav": combined_nav,
        "returns": combined_returns,
        "metrics": metrics,
        "weights": dict(zip(selected, weights)),
        "rolling_sharpe": rolling_sharpe,
        "sharpe_stability": sharpe_stability,
        "aligned_days": len(returns_df),
    }


def get_low_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.4,
) -> List[Dict]:
    """
    Find pairs of portfolios with low correlation (good diversification).

    Args:
        corr_matrix: Correlation matrix
        threshold: Maximum correlation to consider "low" (default 0.4)

    Returns:
        List of dictionaries with pair info, sorted by correlation ascending
    """
    pairs = []
    candidates = corr_matrix.columns.tolist()

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            corr = corr_matrix.iloc[i, j]
            if corr <= threshold:
                pairs.append({
                    "portfolio_1": candidates[i],
                    "portfolio_2": candidates[j],
                    "correlation": corr,
                })

    # Sort by correlation (lowest first)
    pairs.sort(key=lambda x: x["correlation"])

    return pairs
