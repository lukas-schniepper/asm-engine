"""
Performance Metrics Calculation Module.

Provides functions for calculating standard and institutional-grade performance metrics:

Basic Metrics:
- Returns (daily, cumulative, annualized)
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- CAGR, Maximum Drawdown, Volatility, Win Rate

Institutional Risk Metrics:
- Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
- Beta and Alpha (Jensen's Alpha)
- Information Ratio and Tracking Error
- Portfolio-Benchmark Correlation

Rolling Metrics:
- Rolling Sharpe Ratio, Volatility, Correlation, Beta

Advanced Drawdown Analysis:
- Drawdown duration, recovery time, time underwater
- Worst N drawdowns with full details

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
    Calculate total return over the period using GIPS-compliant daily compounding.

    This is the industry-standard method (GIPS/IBKR) that compounds daily returns.
    The first day's return belongs to the current period.

    Args:
        nav_series: NAV time series

    Returns:
        Total return as decimal (e.g., 0.15 for 15%)
    """
    if len(nav_series) < 2:
        return 0.0
    daily_returns = nav_series.pct_change().dropna()
    if len(daily_returns) == 0:
        return 0.0
    return float((1 + daily_returns).prod() - 1)


def calculate_total_return_point_to_point(nav_series: pd.Series) -> float:
    """
    Calculate total return using simple point-to-point NAV ratio.

    This is the legacy method: (end_nav / start_nav) - 1
    Kept for backwards compatibility.

    Args:
        nav_series: NAV time series

    Returns:
        Total return as decimal (e.g., 0.15 for 15%)
    """
    if len(nav_series) < 2:
        return 0.0
    return float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)


def calculate_period_return_gips(daily_returns: pd.Series) -> float:
    """
    Calculate return for a period using GIPS-compliant daily compounding.

    This compounds daily returns for any period (day, month, quarter, year, etc.)
    Industry standard per GIPS and Interactive Brokers.

    Formula: (1 + r1) × (1 + r2) × ... × (1 + rn) - 1

    Args:
        daily_returns: Series of daily returns for the period

    Returns:
        Compounded return as decimal (e.g., 0.15 for 15%)
    """
    if len(daily_returns) == 0:
        return 0.0
    return float((1 + daily_returns).prod() - 1)


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
    daily_returns: Optional[pd.Series] = None,
) -> dict:
    """
    Calculate all performance metrics for a NAV series.

    Args:
        nav_series: NAV time series
        risk_free_rate: Annual risk-free rate
        start_date: Override start date
        end_date: Override end date
        daily_returns: Pre-calculated daily returns (optional, recommended for GIPS compliance).
                       When provided, these are used for total_return calculation to ensure
                       the first day's return is included (GIPS standard).

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

    # Use pre-calculated returns if provided (GIPS-compliant)
    # Otherwise calculate from NAV (loses first day's return)
    if daily_returns is not None and len(daily_returns) > 0:
        returns = daily_returns.dropna()
        total_return = calculate_period_return_gips(returns)
    else:
        returns = calculate_returns(nav_series)
        total_return = calculate_total_return(nav_series)

    return {
        "total_return": total_return,
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


# =============================================================================
# INSTITUTIONAL-LEVEL RISK METRICS
# =============================================================================


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    VaR measures the worst expected loss at a given confidence level.

    Args:
        returns: Daily returns series
        confidence: Confidence level (0.95 = 95% VaR, 0.99 = 99% VaR)
        method: "historical" (percentile-based) or "parametric" (assumes normal distribution)

    Returns:
        VaR as negative decimal (e.g., -0.02 means 2% daily loss not exceeded 95% of time)
    """
    if len(returns) < 10:
        return 0.0

    if method == "historical":
        return float(returns.quantile(1 - confidence))
    else:
        # Parametric (assumes normal distribution)
        from scipy import stats
        z = stats.norm.ppf(1 - confidence)
        return float(returns.mean() + z * returns.std())


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (CVaR), also known as Expected Shortfall.

    CVaR is the average loss beyond the VaR threshold - captures tail risk.

    Args:
        returns: Daily returns series
        confidence: Confidence level (0.95 = 95% CVaR)

    Returns:
        CVaR as negative decimal (e.g., -0.03 means average loss of 3% in worst 5% of days)
    """
    if len(returns) < 10:
        return 0.0

    var = calculate_var(returns, confidence, method="historical")
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    return float(tail_returns.mean())


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate portfolio beta relative to benchmark.

    Beta measures systematic risk exposure:
    - Beta > 1: More volatile than benchmark
    - Beta < 1: Less volatile than benchmark
    - Beta = 1: Same volatility as benchmark

    Formula: Beta = Cov(Portfolio, Benchmark) / Var(Benchmark)

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns (e.g., SPY)

    Returns:
        Beta coefficient
    """
    # Align series by index
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 10:
        return 0.0

    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    covariance = port_ret.cov(bench_ret)
    benchmark_var = bench_ret.var()

    if benchmark_var == 0 or np.isnan(benchmark_var):
        return 0.0

    return float(covariance / benchmark_var)


def calculate_alpha(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Jensen's Alpha (annualized).

    Alpha measures risk-adjusted outperformance - the excess return not explained
    by market exposure (beta).

    Formula: Alpha = Portfolio Return - [Rf + Beta × (Benchmark Return - Rf)]

    Args:
        portfolio_nav: Portfolio NAV series
        benchmark_nav: Benchmark NAV series
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Alpha as annualized decimal (e.g., 0.05 = 5% annual alpha)
    """
    if len(portfolio_nav) < 10 or len(benchmark_nav) < 10:
        return 0.0

    port_returns = calculate_returns(portfolio_nav)
    bench_returns = calculate_returns(benchmark_nav)

    port_cagr = calculate_cagr(portfolio_nav)
    bench_cagr = calculate_cagr(benchmark_nav)
    beta = calculate_beta(port_returns, bench_returns)

    # Jensen's Alpha formula
    alpha = port_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

    return float(alpha)


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate annualized Tracking Error (active risk).

    Tracking Error measures how closely portfolio follows the benchmark.
    Lower TE = more passive/index-like, Higher TE = more active management.

    Formula: TE = StdDev(Portfolio Returns - Benchmark Returns) × √252

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        Annualized tracking error as decimal (e.g., 0.08 = 8%)
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return 0.0

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_information_ratio(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
) -> float:
    """
    Calculate Information Ratio (IR).

    IR measures active management skill - excess return per unit of active risk.

    Formula: IR = Annualized Active Return / Tracking Error

    Interpretation:
    - IR > 0.5: Good active manager
    - IR > 1.0: Excellent active manager
    - IR < 0: Underperforming benchmark after adjusting for tracking error

    Args:
        portfolio_nav: Portfolio NAV series
        benchmark_nav: Benchmark NAV series

    Returns:
        Information Ratio
    """
    if len(portfolio_nav) < 10 or len(benchmark_nav) < 10:
        return 0.0

    port_returns = calculate_returns(portfolio_nav)
    bench_returns = calculate_returns(benchmark_nav)

    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()

    if len(aligned) < 10:
        return 0.0

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    active_mean_annual = active_returns.mean() * TRADING_DAYS_PER_YEAR

    te = calculate_tracking_error(port_returns, bench_returns)

    if te == 0 or np.isnan(te):
        return 0.0

    return float(active_mean_annual / te)


def calculate_correlation(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate correlation between portfolio and benchmark returns.

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        Correlation coefficient (-1 to 1)
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 10:
        return 0.0

    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


# =============================================================================
# ROLLING METRICS
# =============================================================================


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Daily returns series
        window: Rolling window in days (default 60 ~3 months)
        risk_free_rate: Annual risk-free rate

    Returns:
        Series of rolling Sharpe ratios
    """
    if len(returns) < window:
        return pd.Series(dtype=float)

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - daily_rf

    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()

    # Avoid division by zero
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)
    rolling_sharpe = rolling_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)

    return rolling_sharpe


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Calculate rolling annualized volatility.

    Args:
        returns: Daily returns series
        window: Rolling window in days (default 60 ~3 months)

    Returns:
        Series of rolling annualized volatility
    """
    if len(returns) < window:
        return pd.Series(dtype=float)

    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_rolling_correlation(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Calculate rolling correlation to benchmark.

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        window: Rolling window in days (default 60 ~3 months)

    Returns:
        Series of rolling correlations
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < window:
        return pd.Series(dtype=float)

    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def calculate_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Calculate rolling beta to benchmark.

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        window: Rolling window in days (default 60 ~3 months)

    Returns:
        Series of rolling betas
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['portfolio', 'benchmark']

    if len(aligned) < window:
        return pd.Series(dtype=float)

    def calc_beta(x):
        if len(x) < 10:
            return np.nan
        port = x['portfolio']
        bench = x['benchmark']
        cov = port.cov(bench)
        var = bench.var()
        return cov / var if var > 0 else np.nan

    return aligned.rolling(window).apply(
        lambda x: calc_beta(pd.DataFrame({'portfolio': x.iloc[:len(x)//2], 'benchmark': x.iloc[len(x)//2:]})),
        raw=False
    ).iloc[:, 0] if False else aligned.iloc[:, 0].rolling(window).cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].rolling(window).var()


# =============================================================================
# ADVANCED DRAWDOWN ANALYSIS
# =============================================================================


def analyze_drawdowns(nav_series: pd.Series) -> dict:
    """
    Comprehensive drawdown analysis.

    Args:
        nav_series: NAV time series

    Returns:
        Dictionary with:
        - current_drawdown: Current drawdown level
        - current_duration_days: Days in current drawdown
        - max_drawdown: Maximum historical drawdown
        - max_duration_days: Longest drawdown duration
        - avg_drawdown: Average drawdown depth
        - avg_duration_days: Average drawdown duration
        - num_drawdowns: Number of distinct drawdown periods
        - time_underwater_pct: Percentage of time in drawdown
    """
    if len(nav_series) < 2:
        return {
            'current_drawdown': 0.0,
            'current_duration_days': 0,
            'max_drawdown': 0.0,
            'max_duration_days': 0,
            'avg_drawdown': 0.0,
            'avg_duration_days': 0,
            'num_drawdowns': 0,
            'time_underwater_pct': 0.0,
        }

    running_max = nav_series.expanding().max()
    drawdowns = (nav_series - running_max) / running_max
    in_drawdown = drawdowns < 0

    # Find all drawdown periods
    drawdown_periods = []
    in_dd = False
    dd_start = None
    dd_worst = 0.0

    for idx, (date_idx, dd_val) in enumerate(drawdowns.items()):
        if dd_val < 0 and not in_dd:
            # Start of drawdown
            in_dd = True
            dd_start = date_idx
            dd_worst = dd_val
        elif dd_val < 0 and in_dd:
            # Continue drawdown, track worst
            if dd_val < dd_worst:
                dd_worst = dd_val
        elif dd_val >= 0 and in_dd:
            # End of drawdown
            in_dd = False
            duration = (date_idx - dd_start).days if hasattr(date_idx, '__sub__') else 0
            drawdown_periods.append({
                'start': dd_start,
                'end': date_idx,
                'duration_days': duration,
                'depth': dd_worst,
            })
            dd_worst = 0.0

    # Handle if we're still in a drawdown
    current_dd = float(drawdowns.iloc[-1]) if len(drawdowns) > 0 else 0.0
    current_duration = 0
    if in_dd and dd_start is not None:
        last_date = nav_series.index[-1]
        current_duration = (last_date - dd_start).days if hasattr(last_date, '__sub__') else 0

    # Calculate statistics
    max_duration = max([d['duration_days'] for d in drawdown_periods]) if drawdown_periods else 0
    avg_dd = np.mean([d['depth'] for d in drawdown_periods]) if drawdown_periods else 0.0
    avg_duration = int(np.mean([d['duration_days'] for d in drawdown_periods])) if drawdown_periods else 0
    time_underwater = float(in_drawdown.sum() / len(in_drawdown)) if len(in_drawdown) > 0 else 0.0

    return {
        'current_drawdown': current_dd,
        'current_duration_days': current_duration,
        'max_drawdown': float(drawdowns.min()) if len(drawdowns) > 0 else 0.0,
        'max_duration_days': max_duration,
        'avg_drawdown': float(avg_dd),
        'avg_duration_days': avg_duration,
        'num_drawdowns': len(drawdown_periods),
        'time_underwater_pct': time_underwater,
    }


def get_worst_drawdowns(nav_series: pd.Series, n: int = 5) -> pd.DataFrame:
    """
    Get the N worst drawdowns with full details.

    Args:
        nav_series: NAV time series
        n: Number of worst drawdowns to return (default 5)

    Returns:
        DataFrame with columns:
        - peak_date: Date of peak before drawdown
        - trough_date: Date of maximum drawdown
        - recovery_date: Date when NAV recovered to peak (None if not recovered)
        - peak_nav: NAV at peak
        - trough_nav: NAV at trough
        - drawdown_pct: Drawdown percentage (negative)
        - duration_days: Days from peak to trough
        - recovery_days: Days from trough to recovery (None if not recovered)
    """
    if len(nav_series) < 2:
        return pd.DataFrame()

    running_max = nav_series.expanding().max()
    drawdowns = (nav_series - running_max) / running_max

    # Find all drawdown events (local minima in drawdown series)
    drawdown_events = []
    in_dd = False
    dd_start = None
    dd_worst = 0.0
    dd_worst_idx = None

    for idx, (date_idx, dd_val) in enumerate(drawdowns.items()):
        if dd_val < 0 and not in_dd:
            in_dd = True
            dd_start = date_idx
            dd_worst = dd_val
            dd_worst_idx = date_idx
        elif dd_val < 0 and in_dd:
            if dd_val < dd_worst:
                dd_worst = dd_val
                dd_worst_idx = date_idx
        elif dd_val >= 0 and in_dd:
            in_dd = False
            drawdown_events.append({
                'trough_idx': dd_worst_idx,
                'trough_dd': dd_worst,
                'recovery_idx': date_idx,
            })
            dd_worst = 0.0

    # Handle current drawdown (not yet recovered)
    if in_dd and dd_worst_idx is not None:
        drawdown_events.append({
            'trough_idx': dd_worst_idx,
            'trough_dd': dd_worst,
            'recovery_idx': None,
        })

    # Sort by magnitude (most negative first)
    drawdown_events.sort(key=lambda x: x['trough_dd'])
    worst_n = drawdown_events[:n]

    results = []
    for event in worst_n:
        trough_idx = event['trough_idx']
        recovery_idx = event['recovery_idx']

        # Find peak (last time at high water mark before trough)
        pre_trough_max = running_max.loc[:trough_idx]
        peak_mask = nav_series.loc[:trough_idx] == pre_trough_max
        peak_dates = nav_series.loc[:trough_idx][peak_mask].index
        peak_idx = peak_dates[-1] if len(peak_dates) > 0 else trough_idx

        # Calculate durations
        duration_days = (trough_idx - peak_idx).days if hasattr(trough_idx, '__sub__') else 0
        recovery_days = (recovery_idx - trough_idx).days if recovery_idx is not None and hasattr(recovery_idx, '__sub__') else None

        results.append({
            'peak_date': peak_idx,
            'trough_date': trough_idx,
            'recovery_date': recovery_idx,
            'peak_nav': float(nav_series.loc[peak_idx]),
            'trough_nav': float(nav_series.loc[trough_idx]),
            'drawdown_pct': float(event['trough_dd']),
            'duration_days': duration_days,
            'recovery_days': recovery_days,
        })

    return pd.DataFrame(results)


# =============================================================================
# COMPREHENSIVE INSTITUTIONAL METRICS
# =============================================================================


def calculate_institutional_metrics(
    portfolio_nav: pd.Series,
    benchmark_nav: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Calculate all institutional-grade performance metrics in one call.

    This is the main function for institutional reporting, calculating:
    - Basic performance metrics (return, CAGR, Sharpe, Sortino, etc.)
    - Risk metrics (VaR, CVaR at 95% and 99%)
    - Benchmark-relative metrics (Beta, Alpha, IR, TE, Correlation)
    - Drawdown analytics (current, max, duration, time underwater)

    Args:
        portfolio_nav: Portfolio NAV series
        benchmark_nav: Optional benchmark NAV series (e.g., SPY)
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Dictionary with all metrics
    """
    if len(portfolio_nav) < 2:
        return {
            # Basic metrics
            'total_return': 0.0,
            'cagr': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            # Risk metrics
            'var_95': 0.0,
            'cvar_95': 0.0,
            'var_99': 0.0,
            'cvar_99': 0.0,
            # Drawdown metrics
            'current_drawdown': 0.0,
            'current_dd_duration': 0,
            'max_dd_duration': 0,
            'time_underwater_pct': 0.0,
        }

    returns = calculate_returns(portfolio_nav)

    # Basic metrics (existing)
    metrics = {
        'total_return': calculate_total_return(portfolio_nav),
        'cagr': calculate_cagr(portfolio_nav),
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(portfolio_nav),
        'calmar_ratio': calculate_calmar_ratio(portfolio_nav),
        'win_rate': calculate_win_rate(returns),
    }

    # Risk metrics (new)
    metrics.update({
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        'var_99': calculate_var(returns, 0.99),
        'cvar_99': calculate_cvar(returns, 0.99),
    })

    # Benchmark-relative metrics (if benchmark provided)
    if benchmark_nav is not None and len(benchmark_nav) >= 10:
        bench_returns = calculate_returns(benchmark_nav)
        metrics.update({
            'beta': calculate_beta(returns, bench_returns),
            'alpha': calculate_alpha(portfolio_nav, benchmark_nav, risk_free_rate),
            'tracking_error': calculate_tracking_error(returns, bench_returns),
            'information_ratio': calculate_information_ratio(portfolio_nav, benchmark_nav),
            'correlation': calculate_correlation(returns, bench_returns),
        })

    # Drawdown metrics (enhanced)
    dd_analysis = analyze_drawdowns(portfolio_nav)
    metrics.update({
        'current_drawdown': dd_analysis['current_drawdown'],
        'current_dd_duration': dd_analysis['current_duration_days'],
        'max_dd_duration': dd_analysis['max_duration_days'],
        'avg_drawdown': dd_analysis['avg_drawdown'],
        'num_drawdowns': dd_analysis['num_drawdowns'],
        'time_underwater_pct': dd_analysis['time_underwater_pct'],
    })

    return metrics
