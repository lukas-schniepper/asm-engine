"""
S3 Data Adapter for asm-models data.

Loads data from the asm-models S3 bucket:
- features_latest.parquet: Technical indicators and signals
- spy.csv: SPY OHLCV prices
- allocation_history.csv: Per-model allocation history
- config files: Model parameters

The asm-data workflow updates S3 nightly, so we always load from S3
to get the freshest data.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from io import BytesIO
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import boto3, but make it optional for environments without S3 access
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. S3 functionality will be disabled.")


# S3 Configuration
DEFAULT_BUCKET = "alpha-state-machine2030"
DEFAULT_DATA_PREFIX = "alpha-state-machine/data"

# Local cache directory
CACHE_DIR = Path("data/cache")


class S3DataLoader:
    """
    Load data from asm-models S3 bucket.

    Features:
    - Automatic retry on transient failures
    - Local caching for fallback when S3 is unavailable
    - Graceful degradation when boto3 is not installed
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        data_prefix: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize S3 data loader.

        Args:
            bucket: S3 bucket name (default from env or DEFAULT_BUCKET)
            data_prefix: S3 key prefix for data files
            cache_enabled: Whether to cache downloaded files locally
            cache_ttl_hours: How long cached files are valid
        """
        self.bucket = bucket or os.environ.get("S3_BUCKET", DEFAULT_BUCKET)
        self.data_prefix = data_prefix or os.environ.get("S3_DATA_PREFIX", DEFAULT_DATA_PREFIX)
        self.cache_enabled = cache_enabled
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Initialize S3 client if boto3 is available
        self._s3_client = None
        if BOTO3_AVAILABLE:
            try:
                self._s3_client = boto3.client("s3")
                # Test connection
                self._s3_client.head_bucket(Bucket=self.bucket)
                logger.info(f"S3 connection established to bucket: {self.bucket}")
            except NoCredentialsError:
                logger.warning("AWS credentials not found. S3 will use cached data only.")
                self._s3_client = None
            except ClientError as e:
                logger.warning(f"S3 connection failed: {e}. Will use cached data only.")
                self._s3_client = None
            except Exception as e:
                logger.warning(f"S3 initialization error: {e}. Will use cached data only.")
                self._s3_client = None

        # Ensure cache directory exists
        if self.cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_connected(self) -> bool:
        """Check if S3 client is available and connected."""
        return self._s3_client is not None

    def _get_s3_key(self, filename: str) -> str:
        """Build full S3 key from filename."""
        return f"{self.data_prefix}/{filename}"

    def _get_cache_path(self, filename: str) -> Path:
        """Get local cache path for a file."""
        return CACHE_DIR / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file exists and is not expired."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < self.cache_ttl

    def _load_from_s3(self, key: str) -> bytes:
        """Load raw bytes from S3."""
        if not self._s3_client:
            raise RuntimeError("S3 client not available")

        response = self._s3_client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def _load_with_cache(self, filename: str, loader_func) -> any:
        """
        Load data with caching strategy.

        1. Try S3 first
        2. On success, update cache
        3. On failure, try cache
        4. If cache invalid/missing, raise error
        """
        cache_path = self._get_cache_path(filename)
        s3_key = self._get_s3_key(filename)

        # Try S3 first
        if self._s3_client:
            try:
                data = self._load_from_s3(s3_key)

                # Update cache
                if self.cache_enabled:
                    cache_path.write_bytes(data)
                    logger.debug(f"Updated cache: {cache_path}")

                return loader_func(BytesIO(data))

            except Exception as e:
                logger.warning(f"S3 load failed for {filename}: {e}")

        # Fall back to cache
        if self.cache_enabled and cache_path.exists():
            if self._is_cache_valid(cache_path):
                logger.info(f"Using cached data: {cache_path}")
                return loader_func(cache_path)
            else:
                logger.warning(f"Cache expired for {filename}, but using anyway as fallback")
                return loader_func(cache_path)

        raise RuntimeError(
            f"Cannot load {filename}: S3 unavailable and no valid cache. "
            f"Checked S3 key: s3://{self.bucket}/{s3_key}"
        )

    def load_features_latest(self) -> pd.DataFrame:
        """
        Load features_latest.parquet containing all technical indicators.

        Returns:
            DataFrame with columns like:
            - date (index)
            - rsi_14, rsi_21
            - volatility_regime
            - stress_level
            - momentum_strength
            - slope
            - vix
            - etc.
        """
        def loader(source):
            df = pd.read_parquet(source)
            if "date" in df.columns:
                df = df.set_index("date")
            return df

        return self._load_with_cache("features_latest.parquet", loader)

    def load_spy_prices(self) -> pd.DataFrame:
        """
        Load SPY price data.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: date
        """
        def loader(source):
            df = pd.read_csv(source, index_col=0, parse_dates=True)
            # Standardize column names
            df.columns = [c.title() if c.islower() else c for c in df.columns]
            return df

        return self._load_with_cache("spy.csv", loader)

    def load_cash_rates(self) -> pd.DataFrame:
        """
        Load cash interest rates.

        Returns:
            DataFrame with various broker cash rates
            Index: date
        """
        def loader(source):
            return pd.read_csv(source, index_col=0, parse_dates=True)

        try:
            return self._load_with_cache("cash_interest_rates.csv", loader)
        except Exception as e:
            logger.warning(f"Could not load cash rates: {e}. Using default 0% rate.")
            # Return empty DataFrame with same structure
            return pd.DataFrame(columns=["ibkr_rate_100k"])

    def load_allocation_history(self, model: str) -> pd.DataFrame:
        """
        Load allocation history for a specific model.

        Args:
            model: Model name ('conservative' or 'trend_regime_v2')

        Returns:
            DataFrame with columns:
            - date
            - allocation
            - target_allocation
            - trade_executed
            - strategy_return
            - spy_return
            - signals (various)
        """
        # Map model names to S3 paths
        model_paths = {
            "conservative": "models/conservative/allocation_history.csv",
            "trend_regime_v2": "models/v2_regime/allocation_history.csv",
        }

        if model not in model_paths:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_paths.keys())}")

        # Build full S3 key (these are outside the data prefix)
        s3_key = f"alpha-state-machine/{model_paths[model]}"

        def loader(source):
            df = pd.read_csv(source, parse_dates=["date"])
            return df

        # Custom load without prefix
        if self._s3_client:
            try:
                data = self._s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                return loader(BytesIO(data["Body"].read()))
            except Exception as e:
                logger.warning(f"Could not load allocation history for {model}: {e}")

        raise RuntimeError(f"Could not load allocation history for {model}")

    def load_model_config(self, model: str) -> dict:
        """
        Load frozen configuration for a specific model.

        Args:
            model: Model name ('conservative' or 'trend_regime_v2')

        Returns:
            Dict with 'frozen_parameters' and other config.
            Returns empty dict if config not found (uses defaults).
        """
        import json

        model_configs = {
            "conservative": "models/conservative/config.json",
            "trend_regime_v2": "models/v2_regime/config.json",
        }

        if model not in model_configs:
            raise ValueError(f"Unknown model: {model}")

        s3_key = f"alpha-state-machine/{model_configs[model]}"

        if self._s3_client:
            try:
                data = self._s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                return json.loads(data["Body"].read().decode("utf-8"))
            except Exception as e:
                logger.info(f"Config not found for {model}, using defaults: {e}")
                return {}

        # Return empty dict - caller will use defaults
        logger.info(f"S3 not available, using default config for {model}")
        return {}


# Singleton instance for convenience
_default_loader: Optional[S3DataLoader] = None


def get_s3_loader() -> S3DataLoader:
    """Get or create the default S3 data loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = S3DataLoader()
    return _default_loader


def load_features() -> pd.DataFrame:
    """Convenience function to load features_latest.parquet."""
    return get_s3_loader().load_features_latest()


def load_spy() -> pd.DataFrame:
    """Convenience function to load SPY prices."""
    return get_s3_loader().load_spy_prices()
