"""
Overlay Adapter for applying risk overlay models to portfolio NAV.

This module provides a unified interface for applying different risk overlay
models (Conservative OCT16, TrendRegimeV2, and future overlays) to a portfolio's
raw NAV to calculate risk-adjusted NAV.

The overlay models are implemented following the exact logic from asm-models,
ensuring consistency with production allocation calculations.

Design Principles:
1. Registry pattern for easy addition of new overlays
2. Configuration loaded from S3 (or cache) to match production
3. All calculations are pure functions for testability
"""

import logging
from datetime import date
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .s3_adapter import S3DataLoader, get_s3_loader

logger = logging.getLogger(__name__)


# =============================================================================
# Overlay Calculation Functions
# =============================================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) indicator.

    Args:
        series: Price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100 range)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_enhanced_features(features: pd.DataFrame, spy_prices: pd.Series) -> pd.DataFrame:
    """
    Calculate enhanced features for allocation models.

    This is the common feature set used by both Conservative and TrendRegimeV2 models.

    Features calculated:
    - Volatility regime (low/high vol)
    - Stress score (composite)
    - Momentum (12m/6m/3m consistency, strength)
    - RSI (14/21 period)
    - Drawdown (current, in_drawdown, severe)
    - Market breadth (high/low correlation)
    - Contrarian signals (COT, put/call)

    Args:
        features: Base features DataFrame (from features_latest.parquet)
        spy_prices: SPY price series

    Returns:
        Enhanced features DataFrame
    """
    enhanced = pd.DataFrame(index=features.index)

    # Calculate SPY returns
    returns = spy_prices.pct_change()

    # Volatility Regime
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    vol_percentile = rolling_vol.rolling(252).rank(pct=True)
    enhanced["low_vol_regime"] = (vol_percentile < 0.33).astype(float)
    enhanced["high_vol_regime"] = (vol_percentile > 0.67).astype(float)

    # Stress Score
    stress_components = []
    if "vix_term" in features.columns:
        vix_stress = features["vix_term"].rolling(252).rank(pct=True)
        stress_components.append(vix_stress)
    if "hy_spread" in features.columns:
        credit_stress = features["hy_spread"].rolling(252).rank(pct=True)
        stress_components.append(credit_stress)

    vol_stress = rolling_vol.rolling(252).rank(pct=True)
    stress_components.append(vol_stress)

    if stress_components:
        composite_stress = pd.concat(stress_components, axis=1).mean(axis=1)
        enhanced["stress_score"] = composite_stress
        enhanced["low_stress"] = (composite_stress < 0.33).astype(float)
        enhanced["extreme_stress"] = (composite_stress > 0.90).astype(float)

    # Momentum
    momentum_12m = returns.rolling(252).sum()
    momentum_6m = returns.rolling(126).sum()
    momentum_3m = returns.rolling(63).sum()

    enhanced["momentum_consistent"] = (
        (momentum_12m > 0) & (momentum_6m > 0) & (momentum_3m > 0)
    ).astype(float)
    enhanced["momentum_strength"] = momentum_12m

    # RSI
    enhanced["rsi_14"] = calculate_rsi(spy_prices, 14)
    enhanced["rsi_21"] = calculate_rsi(spy_prices, 21)

    # Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    enhanced["drawdown"] = drawdown
    enhanced["in_drawdown"] = (drawdown < -0.01).astype(float)
    enhanced["severe_drawdown"] = (drawdown < -0.10).astype(float)

    # Market breadth
    if "pct_above_200" in features.columns:
        breadth = features["pct_above_200"]
        enhanced["high_correlation"] = (breadth < 0.30).astype(float)
        enhanced["low_correlation"] = (breadth > 0.70).astype(float)
        enhanced["pct_above_200"] = breadth

    # Contrarian signals
    if "cot_net" in features.columns:
        enhanced["cot_net"] = features["cot_net"].fillna(0.5)
    if "put_call" in features.columns:
        enhanced["put_call"] = features["put_call"].fillna(0)

    # Slope feature
    if "slope" in features.columns:
        enhanced["slope"] = features["slope"].fillna(0.5)

    # VIX for regime-adaptive features
    if "vix" in features.columns:
        enhanced["vix"] = features["vix"]

    return enhanced


def calculate_allocation_conservative(
    trade_date: date,
    params: dict,
    enhanced_features: pd.DataFrame,
) -> tuple[float, dict, dict]:
    """
    Calculate allocation using Conservative (OCT16) model.

    7-component multiplicative allocation system:
    1. Momentum filter (3-state)
    2. Volatility regime
    3. Stress protection
    4. Market breadth
    5. Slope 5-state logic
    6. Contrarian signals (OR-gated)
    7. Drawdown protection

    Args:
        trade_date: Trading date
        params: Frozen parameters from config
        enhanced_features: Enhanced features DataFrame

    Returns:
        Tuple of (allocation, signals_dict, impacts_dict)
    """
    if trade_date < pd.to_datetime("2008-06-01").date():
        return params["base_allocation"], {}, {"base_allocation": params["base_allocation"]}

    # Get current signals
    dt = pd.to_datetime(trade_date)
    if dt not in enhanced_features.index:
        # Find nearest date
        idx = enhanced_features.index.get_indexer([dt], method="ffill")[0]
        if idx < 0:
            return params["base_allocation"], {}, {"base_allocation": params["base_allocation"]}
        dt = enhanced_features.index[idx]

    current = enhanced_features.loc[dt]
    allocation = params["base_allocation"]

    # Track signals and impacts
    signals = {
        "momentum_strength": float(current.get("momentum_strength", 0)),
        "momentum_consistent": bool(current.get("momentum_consistent", 0)),
        "volatility_regime": "HIGH" if current.get("high_vol_regime", 0) else ("LOW" if current.get("low_vol_regime", 0) else "NORMAL"),
        "stress_category": "EXTREME" if current.get("extreme_stress", 0) else ("LOW" if current.get("low_stress", 0) else "NORMAL"),
        "rsi_14": float(current.get("rsi_14", 50)),
        "rsi_21": float(current.get("rsi_21", 50)),
        "rsi_avg": float((current.get("rsi_14", 50) + current.get("rsi_21", 50)) / 2),
        "slope": float(current.get("slope", 0.5)),
        "drawdown": float(current.get("drawdown", 0)),
        "in_drawdown": bool(current.get("in_drawdown", 0)),
        "cot_net": float(current.get("cot_net", 0.5)),
        "put_call": float(current.get("put_call", 0)),
        "high_correlation": int(current.get("high_correlation", 0)),
        "stress_level": float(current.get("stress_score", 0.5)),
    }

    impacts = {"base_allocation": params["base_allocation"]}

    # 1. Momentum filter
    before = allocation
    momentum_strength = current.get("momentum_strength", 0)
    if momentum_strength < -0.05:
        allocation *= 0.30
    elif momentum_strength < 0:
        allocation *= 0.60
    elif current.get("momentum_consistent", 0) == 1:
        allocation *= 1.15
    impacts["momentum_impact"] = allocation - before

    # 2. Volatility regime
    before = allocation
    if current.get("low_vol_regime", 0) == 1:
        allocation *= params.get("low_vol_multiplier", 1.0)
    elif current.get("high_vol_regime", 0) == 1:
        allocation *= params.get("high_vol_multiplier", 0.55)
    impacts["volatility_impact"] = allocation - before

    # 3. Stress protection
    before = allocation
    if current.get("extreme_stress", 0) == 1:
        allocation *= params.get("extreme_stress_factor", 0.35)
    elif current.get("low_stress", 0) == 1:
        allocation *= 1.10
    impacts["stress_impact"] = allocation - before

    # 4. Market breadth
    before = allocation
    if current.get("high_correlation", 0) == 1:
        allocation *= 0.80
    impacts["breadth_impact"] = allocation - before

    # 5. Slope 5-state logic
    before = allocation
    slope_value = current.get("slope", 0.5)
    if not pd.isna(slope_value):
        if slope_value < -0.25:
            allocation *= 0.65
        elif slope_value < 0.5:
            allocation *= 0.80
        elif slope_value > 2.5:
            allocation *= params.get("slope_very_steep_boost", 1.55)
        elif slope_value > 2.0:
            allocation *= params.get("slope_steep_boost", 1.2)
    impacts["slope_impact"] = allocation - before

    # 6. Contrarian signals (OR-gated)
    before = allocation
    rsi_avg = (current.get("rsi_14", 50) + current.get("rsi_21", 50)) / 2
    is_oversold = False

    if rsi_avg < params.get("rsi_oversold", 32):
        is_oversold = True
    if current.get("cot_net", 0.5) > params.get("cot_threshold", 1.5):
        is_oversold = True
    if current.get("put_call", 0) > params.get("put_call_threshold", 2.25):
        is_oversold = True

    if is_oversold:
        allocation *= params.get("oversold_boost", 1.2)
    elif rsi_avg > 70:
        allocation *= 0.85
    impacts["rsi_impact"] = allocation - before

    # 7. Drawdown protection
    before = allocation
    if current.get("severe_drawdown", 0) == 1:
        allocation *= 0.50
    elif current.get("in_drawdown", 0) == 1:
        drawdown_severity = abs(current.get("drawdown", 0))
        protection_factor = max(0.6, 1 - drawdown_severity)
        allocation *= protection_factor
    impacts["drawdown_impact"] = allocation - before

    # Final clipping
    allocation = float(np.clip(allocation, 0.05, params.get("max_allocation", 1.0)))

    return allocation, signals, impacts


def calculate_allocation_trend_regime_v2(
    trade_date: date,
    params: dict,
    enhanced_features: pd.DataFrame,
    spy_prices: pd.Series,
) -> tuple[float, dict, dict]:
    """
    Calculate allocation using TrendRegimeV2 model.

    Extends Conservative (OCT16) with 3 additional features:
    1. Regime-adaptive trend boost (VIX-based)
    2. Efficient momentum (drawdown-adjusted momentum)
    3. Alignment quality score (multi-factor confirmation)

    Args:
        trade_date: Trading date
        params: Frozen parameters from config
        enhanced_features: Enhanced features DataFrame
        spy_prices: SPY price series (for efficient momentum calculation)

    Returns:
        Tuple of (allocation, signals_dict, impacts_dict)
    """
    # Start with Conservative allocation
    allocation, signals, impacts = calculate_allocation_conservative(
        trade_date, params, enhanced_features
    )

    # Get current signals
    dt = pd.to_datetime(trade_date)
    if dt not in enhanced_features.index:
        idx = enhanced_features.index.get_indexer([dt], method="ffill")[0]
        if idx < 0:
            return allocation, signals, impacts
        dt = enhanced_features.index[idx]

    current = enhanced_features.loc[dt]

    # V2.0 Feature 1: Regime-Adaptive Trend Boost
    before = allocation
    vix = current.get("vix", 20)
    trend_strength = current.get("momentum_consistent", 0)  # Use momentum as trend proxy

    if trend_strength == 1:
        if vix < 15:
            allocation *= params.get("trend_boost_low_vol", 1.30)
        elif vix < 25:
            allocation *= params.get("trend_boost_med_vol", 1.20)
        else:
            allocation *= params.get("trend_boost_high_vol", 1.00)

    impacts["regime_boost_impact"] = allocation - before
    signals["vix"] = float(vix)
    signals["trend_strength"] = int(trend_strength)

    # V2.0 Feature 2: Efficient Momentum
    before = allocation
    efficient_mom = 0.0
    if dt in spy_prices.index:
        # Calculate efficient momentum at this date
        lookback = 126
        if len(spy_prices.loc[:dt]) >= lookback:
            prices = spy_prices.loc[:dt].tail(lookback + 1)
            mom = (prices.iloc[-1] / prices.iloc[0]) - 1
            rolling_max = prices.max()
            dd = abs(prices.iloc[-1] / rolling_max - 1)
            if dd > 0.01:
                efficient_mom = mom / dd
            else:
                efficient_mom = mom / 0.01

    if efficient_mom > params.get("efficient_mom_threshold", 0.5):
        allocation *= params.get("efficient_mom_boost", 1.30)

    impacts["efficient_mom_impact"] = allocation - before
    signals["efficient_momentum"] = float(efficient_mom)

    # V2.0 Feature 3: Alignment Quality Score
    before = allocation
    alignment_quality = 0.0

    # Calculate alignment components
    components = []
    if "slope" in current and not pd.isna(current["slope"]):
        slope_norm = min(1.0, max(0.0, current["slope"] / 3.0))
        components.append(slope_norm)
    if trend_strength:
        components.append(float(trend_strength))
    if current.get("low_vol_regime", 0):
        components.append(float(current["low_vol_regime"]))
    if current.get("momentum_strength", 0) > 0:
        components.append(1.0)
    else:
        components.append(0.0)

    if components:
        alignment_quality = float(np.mean(components))

    # Apply alignment boost (proportional, max 1.10x)
    max_alignment_boost = params.get("max_alignment_boost", 1.10)
    alignment_multiplier = 1.0 + (max_alignment_boost - 1.0) * alignment_quality
    allocation *= alignment_multiplier

    impacts["alignment_impact"] = allocation - before
    signals["alignment_quality"] = alignment_quality

    # Final clipping
    allocation = float(np.clip(allocation, 0.05, params.get("max_allocation", 1.0)))

    return allocation, signals, impacts


# =============================================================================
# Overlay Registry
# =============================================================================

@dataclass
class OverlayConfig:
    """Configuration for a registered overlay model."""
    name: str
    display_name: str
    config_key: str  # Key in S3 for config file
    calculator: Callable
    needs_spy_prices: bool = False  # Whether calculator needs spy_prices arg


# Registry of available overlay models
OVERLAY_REGISTRY: dict[str, OverlayConfig] = {
    "conservative": OverlayConfig(
        name="conservative",
        display_name="Conservative Model (OCT16)",
        config_key="conservative",
        calculator=calculate_allocation_conservative,
        needs_spy_prices=False,
    ),
    "trend_regime_v2": OverlayConfig(
        name="trend_regime_v2",
        display_name="Trend Regime V2.0",
        config_key="trend_regime_v2",
        calculator=calculate_allocation_trend_regime_v2,
        needs_spy_prices=True,
    ),
}

# Default parameters (used when S3 config is unavailable)
DEFAULT_PARAMS = {
    "conservative": {
        "base_allocation": 0.55,
        "max_allocation": 1.0,
        "low_vol_multiplier": 1.0,
        "high_vol_multiplier": 0.55,
        "extreme_stress_factor": 0.35,
        "oversold_boost": 1.2,
        "rsi_oversold": 32,
        "cot_threshold": 1.5,
        "put_call_threshold": 2.25,
        "slope_steep_boost": 1.2,
        "slope_very_steep_boost": 1.55,
        "rebalance_threshold": 0.25,
    },
    "trend_regime_v2": {
        "base_allocation": 0.55,
        "max_allocation": 1.0,
        "low_vol_multiplier": 1.0,
        "high_vol_multiplier": 0.55,
        "extreme_stress_factor": 0.35,
        "oversold_boost": 1.2,
        "rsi_oversold": 32,
        "cot_threshold": 1.5,
        "put_call_threshold": 2.25,
        "slope_steep_boost": 1.2,
        "slope_very_steep_boost": 1.55,
        "rebalance_threshold": 0.25,
        # V2.0 features
        "trend_boost_low_vol": 1.30,
        "trend_boost_med_vol": 1.20,
        "trend_boost_high_vol": 1.00,
        "efficient_mom_threshold": 0.5,
        "efficient_mom_boost": 1.30,
        "max_alignment_boost": 1.10,
    },
}


# =============================================================================
# Overlay Adapter Class
# =============================================================================

class OverlayAdapter:
    """
    Adapter for applying risk overlay models to portfolio NAV.

    Example usage:
        adapter = OverlayAdapter()

        # Apply Conservative overlay
        adjusted_nav, allocation, signals, impacts = adapter.apply_overlay(
            model="conservative",
            raw_nav=100000.0,
            trade_date=date(2025, 11, 30),
        )
    """

    def __init__(self, s3_loader: Optional[S3DataLoader] = None):
        """
        Initialize the overlay adapter.

        Args:
            s3_loader: S3DataLoader instance (uses default if not provided)
        """
        self.s3_loader = s3_loader or get_s3_loader()
        self._config_cache: dict[str, dict] = {}
        self._features_cache: Optional[pd.DataFrame] = None
        self._spy_cache: Optional[pd.DataFrame] = None
        self._enhanced_features_cache: Optional[pd.DataFrame] = None
        self._allocation_history_cache: dict[str, pd.DataFrame] = {}

    def get_available_overlays(self) -> list[str]:
        """Return list of available overlay model names."""
        return list(OVERLAY_REGISTRY.keys())

    def get_overlay_display_name(self, model: str) -> str:
        """Get human-readable name for an overlay model."""
        if model not in OVERLAY_REGISTRY:
            raise ValueError(f"Unknown model: {model}. Available: {self.get_available_overlays()}")
        return OVERLAY_REGISTRY[model].display_name

    def _load_config(self, model: str) -> dict:
        """Load configuration for a model (with caching)."""
        if model in self._config_cache:
            return self._config_cache[model]

        try:
            config = self.s3_loader.load_model_config(model)
            params = config.get("frozen_parameters", config)
            # If config is empty, use defaults
            if not params or "base_allocation" not in params:
                params = DEFAULT_PARAMS.get(model, DEFAULT_PARAMS["conservative"])
            self._config_cache[model] = params
            return params
        except Exception as e:
            logger.warning(f"Could not load config for {model} from S3: {e}. Using defaults.")
            return DEFAULT_PARAMS.get(model, DEFAULT_PARAMS["conservative"])

    def _load_features(self) -> pd.DataFrame:
        """Load and cache features data."""
        if self._features_cache is None:
            self._features_cache = self.s3_loader.load_features_latest()
        return self._features_cache

    def _load_spy(self) -> pd.DataFrame:
        """Load and cache SPY prices."""
        if self._spy_cache is None:
            self._spy_cache = self.s3_loader.load_spy_prices()
        return self._spy_cache

    def _get_enhanced_features(self) -> pd.DataFrame:
        """Get enhanced features (calculated once and cached)."""
        if self._enhanced_features_cache is None:
            features = self._load_features()
            spy_df = self._load_spy()
            spy_prices = spy_df["Close"] if "Close" in spy_df.columns else spy_df["SPY"]
            self._enhanced_features_cache = calculate_enhanced_features(features, spy_prices)
        return self._enhanced_features_cache

    def clear_cache(self):
        """Clear all cached data (call when data might have been updated)."""
        self._config_cache.clear()
        self._features_cache = None
        self._spy_cache = None
        self._enhanced_features_cache = None
        self._allocation_history_cache.clear()

    def _load_allocation_history(self, model: str) -> Optional[pd.DataFrame]:
        """
        Load allocation history from S3 for a model (with caching).

        Returns:
            DataFrame with columns: date, allocation, target_allocation, etc.
            Returns None if not available.
        """
        if model in self._allocation_history_cache:
            return self._allocation_history_cache[model]

        try:
            df = self.s3_loader.load_allocation_history(model)
            if df is not None and not df.empty:
                # Ensure date column is datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                self._allocation_history_cache[model] = df
                logger.info(f"Loaded allocation history for {model}: {len(df)} rows")
                return df
        except Exception as e:
            logger.warning(f"Could not load allocation history for {model}: {e}")

        return None

    def _get_allocation_from_history(
        self, model: str, trade_date: date
    ) -> Optional[tuple[float, dict]]:
        """
        Get allocation for a specific date from S3 allocation history.

        Returns:
            Tuple of (allocation, signals_dict) or None if not found.
        """
        history = self._load_allocation_history(model)
        if history is None:
            return None

        # Find the row for this date
        target_date = pd.to_datetime(trade_date)

        if "date" in history.columns:
            row = history[history["date"] == target_date]
        else:
            # If date is the index
            row = history.loc[history.index == target_date]

        if row.empty:
            logger.debug(f"No allocation found in history for {model} on {trade_date}")
            return None

        row = row.iloc[0]

        # Get allocation (prefer 'allocation' column, fall back to 'target_allocation')
        allocation = row.get("allocation", row.get("target_allocation"))
        if allocation is None or pd.isna(allocation):
            return None

        # Build signals dict from available columns
        signals = {}
        signal_columns = [c for c in row.index if c not in ["date", "allocation", "target_allocation"]]
        for col in signal_columns:
            val = row[col]
            if not pd.isna(val):
                signals[col] = val

        logger.info(f"Using S3 allocation for {model} on {trade_date}: {allocation:.4f}")
        return float(allocation), signals

    def apply_overlay(
        self,
        model: str,
        raw_nav: float,
        trade_date: date,
    ) -> tuple[float, float, dict, dict]:
        """
        Apply a risk overlay model to adjust NAV.

        First tries to get the allocation from S3 allocation_history.csv (source of truth).
        Falls back to local calculation only if S3 data is unavailable.

        Args:
            model: Overlay model name ('conservative' or 'trend_regime_v2')
            raw_nav: Raw portfolio NAV (100% equity)
            trade_date: Trading date

        Returns:
            Tuple of:
            - adjusted_nav: NAV after applying overlay (equity * allocation + cash)
            - allocation: Equity allocation (0.0 to 1.0)
            - signals: Dict of signal values used in calculation
            - impacts: Dict of per-factor allocation impacts
        """
        if model not in OVERLAY_REGISTRY:
            raise ValueError(f"Unknown model: {model}. Available: {self.get_available_overlays()}")

        # FIRST: Try to get allocation from S3 allocation history (source of truth)
        s3_result = self._get_allocation_from_history(model, trade_date)
        if s3_result is not None:
            allocation, signals = s3_result
            # S3 history doesn't have impacts breakdown, so provide empty
            impacts = {"source": "s3_allocation_history"}

            # Calculate adjusted NAV
            adjusted_nav = raw_nav * allocation + raw_nav * (1 - allocation) * 1.0
            return adjusted_nav, allocation, signals, impacts

        # FALLBACK: Calculate allocation locally if S3 data not available
        logger.info(f"S3 allocation not found for {model} on {trade_date}, calculating locally")

        config = OVERLAY_REGISTRY[model]
        params = self._load_config(model)
        enhanced_features = self._get_enhanced_features()

        # Calculate allocation
        if config.needs_spy_prices:
            spy_df = self._load_spy()
            spy_prices = spy_df["Close"] if "Close" in spy_df.columns else spy_df["SPY"]
            allocation, signals, impacts = config.calculator(
                trade_date, params, enhanced_features, spy_prices
            )
        else:
            allocation, signals, impacts = config.calculator(
                trade_date, params, enhanced_features
            )

        impacts["source"] = "local_calculation"

        # Calculate adjusted NAV
        # Assumption: cash portion earns 0% for simplicity (can be enhanced later)
        adjusted_nav = raw_nav * allocation + raw_nav * (1 - allocation) * 1.0

        return adjusted_nav, allocation, signals, impacts


# Convenience function
def get_overlay_adapter() -> OverlayAdapter:
    """Get a default OverlayAdapter instance."""
    return OverlayAdapter()
