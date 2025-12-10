"""
Benchmark Data Adapter.

Provides access to benchmark data (SPY, QQQ, etc.) for calculating
benchmark-relative metrics like Beta, Alpha, and Information Ratio.

Uses EODHD API for data with caching to avoid repeated API calls.
"""

import logging
import os
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
        """Initialize the benchmark adapter with EODHD client and cache."""
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_expiry: dict[str, date] = {}
        self._eodhd_client = None

    def _get_eodhd_client(self):
        """Lazy initialization of EODHD client."""
        if self._eodhd_client is None:
            from AlphaMachine_core.data_sources.eodhd_http_client import EODHDHttpClient

            api_key = os.getenv('EODHD_API_KEY')
            if not api_key:
                # Try Streamlit secrets as fallback
                try:
                    import streamlit as st
                    api_key = st.secrets.get('EODHD_API_KEY')
                except:
                    pass

            if not api_key:
                raise ValueError(
                    "EODHD_API_KEY not found in environment or Streamlit secrets. "
                    "Please add to .streamlit/secrets.toml"
                )

            self._eodhd_client = EODHDHttpClient(api_key)

        return self._eodhd_client

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

        # Fetch from EODHD
        try:
            df = self._fetch_from_eodhd(symbol, start_date, end_date)
            if not df.empty:
                self._cache[cache_key] = df
                self._cache_expiry[cache_key] = date.today() + timedelta(days=1)
            return df
        except Exception as e:
            logger.error(f"Error fetching benchmark data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_eodhd(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch data from EODHD API.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            client = self._get_eodhd_client()

            # Add buffer days for returns calculation
            buffer_start = start_date - timedelta(days=10)

            df = client.get_eod_data(
                ticker=symbol,
                start_date=buffer_start.strftime('%Y-%m-%d'),
                end_date=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            )

            if df.empty:
                logger.warning(f"No data returned from EODHD for {symbol}")
                return pd.DataFrame()

            # Ensure timezone-naive index
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            return df

        except ValueError as e:
            # EODHD API key not configured
            logger.error(f"EODHD configuration error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"EODHD error for {symbol}: {e}")
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
