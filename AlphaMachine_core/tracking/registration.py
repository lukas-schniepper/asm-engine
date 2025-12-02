"""
Portfolio Registration Helper for Backtester Integration.

Provides functions to register backtested portfolios for tracking.
"""

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def register_portfolio_from_backtest(
    name: str,
    backtest_params: dict,
    holdings_df: pd.DataFrame,
    nav_history: pd.Series,
    source: str,
    description: Optional[str] = None,
) -> dict:
    """
    Register a portfolio from backtester results for tracking.

    Args:
        name: Unique portfolio name
        backtest_params: Dict of backtest parameters (ui_params from backtester)
        holdings_df: DataFrame with current holdings (ticker, weight columns)
        nav_history: Series with historical NAV values (from backtest)
        source: Data source (e.g., "Topweights", "TR20")
        description: Optional description

    Returns:
        Dict with registration result including portfolio_id
    """
    try:
        from .tracker import get_tracker
        from .models import Variants

        tracker = get_tracker()

        # Determine start date from NAV history
        start_date = nav_history.index.min()
        if hasattr(start_date, 'date'):
            start_date = start_date.date()

        # Build config from backtest params
        config = {
            "backtest_params": backtest_params,
            "registered_from": "backtester",
        }

        # Register the portfolio
        portfolio = tracker.register_portfolio(
            name=name,
            config=config,
            source=source,
            start_date=start_date,
            description=description,
        )

        # Record current holdings
        if not holdings_df.empty:
            holdings_list = []
            for _, row in holdings_df.iterrows():
                holding = {
                    "ticker": row.get("ticker") or row.get("Ticker") or row.name,
                    "weight": row.get("weight") or row.get("Weight") or row.get("Target Weight", 0),
                }
                holdings_list.append(holding)

            tracker.record_holdings(
                portfolio_id=portfolio.id,
                effective_date=date.today(),
                holdings=holdings_list,
            )

        # Backfill NAV from historical data (raw variant only)
        backfilled_count = _backfill_nav_from_history(
            tracker, portfolio.id, nav_history
        )

        logger.info(
            f"Registered portfolio '{name}' (id={portfolio.id}) "
            f"with {len(holdings_list) if not holdings_df.empty else 0} holdings, "
            f"backfilled {backfilled_count} NAV records"
        )

        return {
            "success": True,
            "portfolio_id": portfolio.id,
            "portfolio_name": name,
            "holdings_count": len(holdings_list) if not holdings_df.empty else 0,
            "backfilled_nav_count": backfilled_count,
        }

    except Exception as e:
        logger.error(f"Failed to register portfolio '{name}': {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _backfill_nav_from_history(
    tracker,
    portfolio_id: int,
    nav_history: pd.Series,
) -> int:
    """
    Backfill historical NAV data from backtest results.

    Only backfills the 'raw' variant since overlays need to be
    calculated with actual features data.

    Returns:
        Number of records backfilled
    """
    from .models import Variants

    if nav_history.empty:
        return 0

    count = 0
    initial_nav = nav_history.iloc[0]

    for trade_date, nav_value in nav_history.items():
        # Convert to date if needed
        if hasattr(trade_date, 'date'):
            trade_date = trade_date.date()

        # Calculate returns
        if count == 0:
            daily_return = 0.0
        else:
            prev_nav = nav_history.iloc[count - 1]
            daily_return = (nav_value / prev_nav) - 1 if prev_nav > 0 else 0.0

        cumulative_return = (nav_value / initial_nav) - 1 if initial_nav > 0 else 0.0

        tracker.record_nav(
            portfolio_id=portfolio_id,
            trade_date=trade_date,
            variant=Variants.RAW,
            nav=float(nav_value),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            equity_allocation=1.0,
            cash_allocation=0.0,
        )
        count += 1

    return count


def get_suggested_portfolio_name(
    source: str,
    num_stocks: int,
    optimizer: str,
) -> str:
    """
    Generate a suggested portfolio name based on parameters.

    Args:
        source: Data source
        num_stocks: Number of stocks
        optimizer: Optimizer method

    Returns:
        Suggested name like "Topweights_20_mvo"
    """
    # Clean up optimizer name
    opt_short = optimizer.replace("-", "_").replace(" ", "_").lower()

    return f"{source}_{num_stocks}_{opt_short}"


def check_portfolio_name_exists(name: str) -> bool:
    """Check if a portfolio with this name already exists."""
    try:
        from .tracker import get_tracker

        tracker = get_tracker()
        existing = tracker.get_portfolio_by_name(name)
        return existing is not None
    except Exception:
        return False
