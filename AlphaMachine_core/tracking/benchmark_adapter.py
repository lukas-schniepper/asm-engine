"""
Benchmark Data Adapter.

Provides access to benchmark data (SPY, QQQ, etc.) for calculating
benchmark-relative metrics like Beta, Alpha, and Information Ratio.

Uses Yahoo Finance for data with caching to avoid repeated API calls.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkAdapter:
    """
    Load and cache benchmark data for performance analysis.

    Supported benchmarks:
    - SPY: S&P 500 ETF
    - QQQ: Nasdaq 100 ETF
    - IWM: Russell 2000 ETF
    - EFA: MSCI EAFE ETF
    - AGG: Bloomberg Aggregate Bond ETF
    """

    BENCHMARKS = {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "IWM": "Russell 2000 ETF",
        "EFA": "MSCI EAFE ETF",
        "AGG": "US Aggregate Bond ETF",
    }

    def __init__(self):
        """Initialize the benchmark adapter with cache."""
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_expiry: dict[str, date] = {}

    def get_benchmark_nav(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        normalize: bool = True,
    ) -> pd.Series:
        """
        Get benchmark NAV series for a date range.

        Args:
            symbol: Benchmark ticker (e.g., 'SPY')
            start_date: Start date
            end_date: End date
            normalize: If True, normalize to start at 100 (default True)

        Returns:
            NAV series indexed by date
        """
        df = self._get_benchmark_data(symbol, start_date, end_date)

        if df.empty:
            logger.warning(f"No benchmark data for {symbol}")
            return pd.Series(dtype=float)

        # Filter to date range
        mask = (df.index >= pd.to_datetime(start_date)) & (
            df.index <= pd.to_datetime(end_date)
        )
        df = df[mask]

        if df.empty:
            return pd.Series(dtype=float)

        nav = df['Close']

        if normalize and len(nav) > 0:
            nav = (nav / nav.iloc[0]) * 100

        return nav

    def get_benchmark_returns(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """
        Get benchmark daily returns for a date range.

        Args:
            symbol: Benchmark ticker (e.g., 'SPY')
            start_date: Start date
            end_date: End date

        Returns:
            Daily returns series indexed by date
        """
        nav = self.get_benchmark_nav(symbol, start_date, end_date, normalize=False)

        if len(nav) < 2:
            return pd.Series(dtype=float)

        return nav.pct_change().dropna()

    def _get_benchmark_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get benchmark OHLCV data, with caching.

        Args:
            symbol: Benchmark ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"

        # Check cache (valid for 1 day)
        if cache_key in self._cache:
            if self._cache_expiry.get(cache_key, date.min) >= date.today():
                return self._cache[cache_key]

        # Fetch from Yahoo Finance
        try:
            df = self._fetch_from_yfinance(symbol, start_date, end_date)
            if not df.empty:
                self._cache[cache_key] = df
                self._cache_expiry[cache_key] = date.today() + timedelta(days=1)
            return df
        except Exception as e:
            logger.error(f"Error fetching benchmark data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_yfinance(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf

            # Add buffer days for returns calculation
            buffer_start = start_date - timedelta(days=10)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=buffer_start, end=end_date + timedelta(days=1))

            if df.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return pd.DataFrame()

            # Ensure timezone-naive index
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            return df

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame()

    def list_benchmarks(self) -> dict[str, str]:
        """Return dict of available benchmarks."""
        return self.BENCHMARKS.copy()

    def get_default_benchmark(self) -> str:
        """Return the default benchmark symbol."""
        return "SPY"


# Singleton instance for convenience
_benchmark_adapter: Optional[BenchmarkAdapter] = None


def get_benchmark_adapter() -> BenchmarkAdapter:
    """Get the singleton benchmark adapter instance."""
    global _benchmark_adapter
    if _benchmark_adapter is None:
        _benchmark_adapter = BenchmarkAdapter()
    return _benchmark_adapter
