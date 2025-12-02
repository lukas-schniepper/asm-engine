"""
Performance Metrics Calculation Module.

Provides functions for calculating standard performance metrics:
- Returns (daily, cumulative, annualized)
- Sharpe Ratio
- Sortino Ratio
- CAGR
- Maximum Drawdown
- Calmar Ratio
- Volatility
- Win Rate

All calculations follow industry-standard formulas and are designed
to work with NAV time series data.
"""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading days per year (US markets)
TRADING_DAYS_PER_YEAR = 252


def calculate_returns(nav_series: pd.Series) -> pd.Series:
    """
    Calculate daily returns from NAV series.

    Args:
        nav_series: NAV time series (indexed by date)

    Returns:
        Daily returns series
    """
    return nav_series.pct_change().dropna()


def calculate_cumulative_returns(nav_series: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from NAV series.

    Args:
        nav_series: NAV time series (indexed by date)

    Returns:
        Cumulative returns series (starting from 0)
    """
    if len(nav_series) == 0:
        return pd.Series(dtype=float)
    return (nav_series / nav_series.iloc[0]) - 1


def calculate_total_return(nav_series: pd.Series) -> float:
    """
    Calculate total return over the period.

    Args:
        nav_series: NAV time series

    Returns:
        Total return as decimal (e.g., 0.15 for 15%)
    """
    if len(nav_series) < 2:
        return 0.0
    return float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)


def calculate_cagr(
    nav_series: pd.Series,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        nav_series: NAV time series
        start_date: Override start date (default: first date in series)
        end_date: Override end date (default: last date in series)

    Returns:
        CAGR as decimal
    """
    if len(nav_series) < 2:
        return 0.0

    start_nav = nav_series.iloc[0]
    end_nav = nav_series.iloc[-1]

    # Calculate years
    if start_date is None:
        start_date = nav_series.index[0]
    if end_date is None:
        end_date = nav_series.index[-1]

    # Handle pandas Timestamp vs date
    if hasattr(start_date, "date"):
        start_date = start_date.date()
    if hasattr(end_date, "date"):
        end_date = end_date.date()

    days = (end_date - start_date).days
    if days <= 0:
        return 0.0

    years = days / 365.25

    if start_nav <= 0:
        return 0.0

    return float((end_nav / start_nav) ** (1 / years) - 1)


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: Returns series
        annualize: Whether to annualize (default True)

    Returns:
        Volatility as decimal
    """
    if len(returns) < 2:
        return 0.0

    vol = returns.std()
    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(vol)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate (default 0)
        annualize: Whether to annualize (default True)

    Returns:
        Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Adjust for daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_rf

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    sharpe = mean_excess / std_excess

    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Calculate Sortino Ratio (only considers downside volatility).

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate (default 0)
        annualize: Whether to annualize (default True)

    Returns:
        Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Adjust for daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_rf

    mean_excess = excess_returns.mean()

    # Downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) < 1:
        # No downside - return large positive ratio
        return float(mean_excess * TRADING_DAYS_PER_YEAR * 100) if annualize else float(mean_excess * 100)

    downside_std = np.sqrt((negative_returns ** 2).mean())

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    sortino = mean_excess / downside_std

    if annualize:
        sortino *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(sortino)


def calculate_max_drawdown(nav_series: pd.Series) -> float:
    """
    Calculate Maximum Drawdown.

    Args:
        nav_series: NAV time series

    Returns:
        Maximum drawdown as negative decimal (e.g., -0.15 for -15%)
    """
    if len(nav_series) < 2:
        return 0.0

    # Running maximum
    running_max = nav_series.expanding().max()

    # Drawdown at each point
    drawdowns = (nav_series - running_max) / running_max

    return float(drawdowns.min())


def calculate_drawdown_series(nav_series: pd.Series) -> pd.Series:
    """
    Calculate drawdown at each point in time.

    Args:
        nav_series: NAV time series

    Returns:
        Drawdown series (negative values)
    """
    if len(nav_series) == 0:
        return pd.Series(dtype=float)

    running_max = nav_series.expanding().max()
    return (nav_series - running_max) / running_max


def calculate_calmar_ratio(
    nav_series: pd.Series,
    returns: Optional[pd.Series] = None,
) -> float:
    """
    Calculate Calmar Ratio (CAGR / |Max Drawdown|).

    Args:
        nav_series: NAV time series
        returns: Pre-calculated returns (optional)

    Returns:
        Calmar Ratio
    """
    if len(nav_series) < 2:
        return 0.0

    cagr = calculate_cagr(nav_series)
    max_dd = calculate_max_drawdown(nav_series)

    if max_dd >= 0:
        # No drawdown - return high value
        return float(cagr * 100) if cagr > 0 else 0.0

    return float(cagr / abs(max_dd))


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive return days).

    Args:
        returns: Returns series

    Returns:
        Win rate as decimal (e.g., 0.55 for 55%)
    """
    if len(returns) == 0:
        return 0.0

    positive_days = (returns > 0).sum()
    total_days = len(returns)

    return float(positive_days / total_days)


def calculate_all_metrics(
    nav_series: pd.Series,
    risk_free_rate: float = 0.0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    """
    Calculate all performance metrics for a NAV series.

    Args:
        nav_series: NAV time series
        risk_free_rate: Annual risk-free rate
        start_date: Override start date
        end_date: Override end date

    Returns:
        Dictionary with all metrics
    """
    if len(nav_series) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
        }

    returns = calculate_returns(nav_series)

    return {
        "total_return": calculate_total_return(nav_series),
        "cagr": calculate_cagr(nav_series, start_date, end_date),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(nav_series),
        "calmar_ratio": calculate_calmar_ratio(nav_series),
        "volatility": calculate_volatility(returns),
        "win_rate": calculate_win_rate(returns),
    }


def calculate_period_metrics(
    nav_df: pd.DataFrame,
    period_type: str,
    period_start: date,
    period_end: date,
    nav_column: str = "nav",
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Calculate metrics for a specific period.

    Args:
        nav_df: DataFrame with NAV data (indexed by date)
        period_type: Period identifier ('week', 'month', 'quarter', 'year', 'ytd', 'all')
        period_start: Start date of period
        period_end: End date of period
        nav_column: Name of NAV column
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with period info and metrics
    """
    # Filter to period
    mask = (nav_df.index >= pd.to_datetime(period_start)) & (
        nav_df.index <= pd.to_datetime(period_end)
    )
    period_df = nav_df[mask]

    if len(period_df) < 2:
        return {
            "period_type": period_type,
            "period_start": period_start,
            "period_end": period_end,
            **calculate_all_metrics(pd.Series(dtype=float), risk_free_rate),
        }

    nav_series = period_df[nav_column]
    metrics = calculate_all_metrics(nav_series, risk_free_rate, period_start, period_end)

    return {
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        **metrics,
    }


def get_period_boundaries(
    as_of_date: date,
) -> dict[str, tuple[date, date]]:
    """
    Get start and end dates for standard periods as of a given date.

    Args:
        as_of_date: Reference date

    Returns:
        Dictionary mapping period names to (start_date, end_date) tuples
    """
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta

    periods = {}

    # Week (last 5 trading days, approximately)
    periods["week"] = (as_of_date - timedelta(days=7), as_of_date)

    # Month (last ~21 trading days)
    periods["month"] = (as_of_date - relativedelta(months=1), as_of_date)

    # Quarter
    periods["quarter"] = (as_of_date - relativedelta(months=3), as_of_date)

    # Year
    periods["year"] = (as_of_date - relativedelta(years=1), as_of_date)

    # YTD (year to date)
    year_start = date(as_of_date.year, 1, 1)
    periods["ytd"] = (year_start, as_of_date)

    return periods


def compare_variants(
    variant_nav_dict: dict[str, pd.Series],
    period_start: Optional[date] = None,
    period_end: Optional[date] = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Compare metrics across multiple portfolio variants.

    Args:
        variant_nav_dict: Dictionary mapping variant names to NAV series
        period_start: Optional start date filter
        period_end: Optional end date filter
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame with variants as columns and metrics as rows
    """
    results = {}

    for variant, nav_series in variant_nav_dict.items():
        # Filter to period if specified
        if period_start is not None or period_end is not None:
            mask = pd.Series(True, index=nav_series.index)
            if period_start is not None:
                mask &= nav_series.index >= pd.to_datetime(period_start)
            if period_end is not None:
                mask &= nav_series.index <= pd.to_datetime(period_end)
            nav_series = nav_series[mask]

        metrics = calculate_all_metrics(nav_series, risk_free_rate, period_start, period_end)
        results[variant] = metrics

    return pd.DataFrame(results)
