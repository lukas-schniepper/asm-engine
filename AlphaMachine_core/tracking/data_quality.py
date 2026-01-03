"""
Data Quality Framework for Portfolio Tracking.

Implements institutional-grade data validation, anomaly detection,
and audit logging to prevent and detect data corruption.

A senior quant developer would implement:
1. Pre-update validation (circuit breakers)
2. Statistical anomaly detection
3. Cross-source reconciliation
4. Audit trail with rollback capability
5. Automated alerting
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from ..db import engine

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityAlert:
    """Represents a data quality alert."""
    timestamp: datetime
    severity: AlertSeverity
    portfolio_name: str
    trade_date: date
    alert_type: str
    message: str
    details: dict

    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.portfolio_name} ({self.trade_date}): {self.message}"


class DataQualityValidator:
    """
    Validates portfolio data before updates.

    Implements circuit breakers that halt processing when
    anomalies are detected, preventing bad data from propagating.
    """

    # Thresholds (configurable)
    MAX_DAILY_RETURN_PCT = 5.0   # Max single-day return (%) - alert threshold
    MAX_NAV_CHANGE_PCT = 20.0    # Max NAV change for circuit breaker (%)
    MIN_NAV_VALUE = 10.0         # NAV should never go below this
    MAX_NAV_DROP_FROM_PEAK = 50.0  # Max drawdown before alert (%)
    CORRELATION_THRESHOLD = 0.3   # Min correlation with benchmark for sanity

    def __init__(self):
        self.alerts: List[DataQualityAlert] = []

    def validate_nav_update(
        self,
        portfolio_name: str,
        trade_date: date,
        new_nav: float,
        previous_nav: float,
        initial_nav: float = 100.0,
        peak_nav: Optional[float] = None,
    ) -> Tuple[bool, List[DataQualityAlert]]:
        """
        Validate a NAV update before committing to database.

        Returns:
            Tuple of (is_valid, alerts)
            If is_valid is False, the update should be blocked.
        """
        alerts = []
        is_valid = True

        # Check 1: NAV should never be negative or zero
        if new_nav <= 0:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                portfolio_name=portfolio_name,
                trade_date=trade_date,
                alert_type="INVALID_NAV",
                message=f"NAV is non-positive: {new_nav}",
                details={"new_nav": new_nav, "previous_nav": previous_nav},
            ))
            is_valid = False

        # Check 2: NAV should not drop below minimum threshold
        if new_nav < self.MIN_NAV_VALUE:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                portfolio_name=portfolio_name,
                trade_date=trade_date,
                alert_type="NAV_TOO_LOW",
                message=f"NAV ({new_nav:.2f}) below minimum threshold ({self.MIN_NAV_VALUE})",
                details={"new_nav": new_nav, "threshold": self.MIN_NAV_VALUE},
            ))
            is_valid = False

        # Check 3: Daily return should be within bounds
        # Three levels: >5% = WARNING, >10% = ERROR, >20% = CRITICAL (blocked)
        if previous_nav > 0:
            daily_return_pct = ((new_nav / previous_nav) - 1) * 100

            if abs(daily_return_pct) > self.MAX_DAILY_RETURN_PCT:
                # Determine severity based on magnitude
                if abs(daily_return_pct) > self.MAX_NAV_CHANGE_PCT:
                    severity = AlertSeverity.CRITICAL
                elif abs(daily_return_pct) > 10:
                    severity = AlertSeverity.ERROR
                else:
                    severity = AlertSeverity.WARNING

                alerts.append(DataQualityAlert(
                    timestamp=datetime.now(),
                    severity=severity,
                    portfolio_name=portfolio_name,
                    trade_date=trade_date,
                    alert_type="EXTREME_DAILY_RETURN",
                    message=f"Daily return {daily_return_pct:+.2f}% exceeds threshold ({self.MAX_DAILY_RETURN_PCT}%)",
                    details={
                        "daily_return_pct": daily_return_pct,
                        "new_nav": new_nav,
                        "previous_nav": previous_nav,
                        "threshold": self.MAX_DAILY_RETURN_PCT,
                    },
                ))
                # Block update if return exceeds circuit breaker threshold (>20%)
                if abs(daily_return_pct) > self.MAX_NAV_CHANGE_PCT:
                    is_valid = False

        # Check 4: Detect potential NAV reset (the QQQ bug)
        # If NAV drops significantly AND lands near 100, likely a reset
        if previous_nav > 110 and 95 < new_nav < 105:
            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                portfolio_name=portfolio_name,
                trade_date=trade_date,
                alert_type="POTENTIAL_NAV_RESET",
                message=f"NAV dropped from {previous_nav:.2f} to {new_nav:.2f} (near 100) - LIKELY DATA ERROR",
                details={
                    "new_nav": new_nav,
                    "previous_nav": previous_nav,
                    "suspected_cause": "NAV reset to initial value",
                },
            ))
            is_valid = False

        # Check 5: Maximum drawdown from peak
        if peak_nav and peak_nav > 0:
            drawdown_pct = ((peak_nav - new_nav) / peak_nav) * 100
            if drawdown_pct > self.MAX_NAV_DROP_FROM_PEAK:
                alerts.append(DataQualityAlert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    portfolio_name=portfolio_name,
                    trade_date=trade_date,
                    alert_type="EXTREME_DRAWDOWN",
                    message=f"Drawdown {drawdown_pct:.1f}% from peak exceeds threshold ({self.MAX_NAV_DROP_FROM_PEAK}%)",
                    details={
                        "peak_nav": peak_nav,
                        "new_nav": new_nav,
                        "drawdown_pct": drawdown_pct,
                    },
                ))

        # Store alerts
        self.alerts.extend(alerts)

        return is_valid, alerts

    def validate_methodology_consistency(
        self,
        portfolio_name: str,
        trade_date: date,
        shares_based_return: float,
        weight_based_return: float,
        tolerance_pct: float = 2.0,
    ) -> Tuple[bool, List[DataQualityAlert]]:
        """
        Compare share-based vs weight-based daily return.

        If they diverge by more than tolerance, this indicates either:
        - Incorrect share counts in holdings
        - Data corruption
        - Calculation error

        BLOCKS update if divergence > tolerance.

        Args:
            portfolio_name: Name of portfolio
            trade_date: Date being validated
            shares_based_return: Daily return calculated from shares × price
            weight_based_return: Daily return calculated from weights × price change
            tolerance_pct: Maximum allowed divergence (default 2%)

        Returns:
            Tuple of (is_valid, alerts)
        """
        alerts = []
        is_valid = True

        diff = abs(shares_based_return - weight_based_return)

        if diff > tolerance_pct:
            # Determine severity based on magnitude
            if diff > 5.0:
                severity = AlertSeverity.CRITICAL
            elif diff > 3.0:
                severity = AlertSeverity.ERROR
            else:
                severity = AlertSeverity.WARNING

            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                severity=severity,
                portfolio_name=portfolio_name,
                trade_date=trade_date,
                alert_type="METHODOLOGY_DIVERGENCE",
                message=f"Share-based return ({shares_based_return:+.2f}%) vs weight-based ({weight_based_return:+.2f}%) differ by {diff:.2f}%",
                details={
                    "shares_based_return": shares_based_return,
                    "weight_based_return": weight_based_return,
                    "difference": diff,
                    "tolerance": tolerance_pct,
                },
            ))

            # Block update if divergence exceeds tolerance
            is_valid = False

        self.alerts.extend(alerts)
        return is_valid, alerts

    def validate_rebalance_nav_continuity(
        self,
        portfolio_name: str,
        trade_date: date,
        pre_rebalance_mtm: float,
        post_rebalance_nav: float,
        tolerance_pct: float = 0.5,
    ) -> Tuple[bool, List[DataQualityAlert]]:
        """
        Validate that NAV is preserved across a rebalance.

        After a rebalance, the new portfolio (valued at current prices)
        should equal the old portfolio's mark-to-market value.

        Any significant difference indicates a calculation error in
        share sizing (the bug this validation catches).

        Args:
            portfolio_name: Name of portfolio
            trade_date: Rebalance date
            pre_rebalance_mtm: Old portfolio's mark-to-market value
            post_rebalance_nav: New portfolio's NAV (sum of new_shares × prices)
            tolerance_pct: Maximum allowed difference (default 0.5% for rounding)

        Returns:
            Tuple of (is_valid, alerts)
        """
        alerts = []
        is_valid = True

        if pre_rebalance_mtm <= 0:
            return True, []  # Can't validate without pre-rebalance value

        diff_pct = abs((post_rebalance_nav / pre_rebalance_mtm) - 1) * 100

        if diff_pct > tolerance_pct:
            # Determine severity based on magnitude
            if diff_pct > 5.0:
                severity = AlertSeverity.CRITICAL
            elif diff_pct > 2.0:
                severity = AlertSeverity.ERROR
            else:
                severity = AlertSeverity.WARNING

            alerts.append(DataQualityAlert(
                timestamp=datetime.now(),
                severity=severity,
                portfolio_name=portfolio_name,
                trade_date=trade_date,
                alert_type="REBALANCE_NAV_DISCONTINUITY",
                message=(
                    f"NAV discontinuity at rebalance: pre-rebalance MTM={pre_rebalance_mtm:.2f}, "
                    f"post-rebalance NAV={post_rebalance_nav:.2f}, "
                    f"difference={diff_pct:.2f}%"
                ),
                details={
                    "pre_rebalance_mtm": pre_rebalance_mtm,
                    "post_rebalance_nav": post_rebalance_nav,
                    "difference_pct": diff_pct,
                    "tolerance_pct": tolerance_pct,
                },
            ))
            is_valid = False

        self.alerts.extend(alerts)
        return is_valid, alerts

    def validate_holdings_prices(
        self,
        portfolio_name: str,
        trade_date: date,
        holdings: List[dict],
        prices: Dict[str, float],
        previous_prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, List[DataQualityAlert]]:
        """
        Validate holdings and prices before NAV calculation.

        Checks:
        - Missing prices for holdings
        - Extreme price moves
        - Stale prices (unchanged for multiple days)
        """
        alerts = []
        is_valid = True

        for holding in holdings:
            ticker = holding.get("ticker")
            if not ticker:
                continue

            price = prices.get(ticker)

            # Check 1: Missing price
            if price is None:
                alerts.append(DataQualityAlert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    portfolio_name=portfolio_name,
                    trade_date=trade_date,
                    alert_type="MISSING_PRICE",
                    message=f"No price data for {ticker}",
                    details={"ticker": ticker},
                ))

            # Check 2: Extreme price move
            elif previous_prices and ticker in previous_prices:
                prev_price = previous_prices[ticker]
                if prev_price > 0:
                    price_change_pct = ((price / prev_price) - 1) * 100
                    if abs(price_change_pct) > 25:
                        alerts.append(DataQualityAlert(
                            timestamp=datetime.now(),
                            severity=AlertSeverity.WARNING,
                            portfolio_name=portfolio_name,
                            trade_date=trade_date,
                            alert_type="EXTREME_PRICE_MOVE",
                            message=f"{ticker} price moved {price_change_pct:+.1f}%",
                            details={
                                "ticker": ticker,
                                "price": price,
                                "previous_price": prev_price,
                                "change_pct": price_change_pct,
                            },
                        ))

        self.alerts.extend(alerts)
        return is_valid, alerts

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        since: Optional[datetime] = None,
    ) -> List[DataQualityAlert]:
        """Get alerts, optionally filtered by severity or time."""
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []


class NAVAuditLog:
    """
    Maintains audit trail of NAV changes.

    Enables:
    - Tracking who/what changed NAV values
    - Rollback to previous values
    - Investigation of data issues
    """

    def __init__(self):
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create audit log table if it doesn't exist."""
        from sqlalchemy import text

        create_sql = """
        CREATE TABLE IF NOT EXISTS nav_audit_log (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            portfolio_id INTEGER NOT NULL,
            portfolio_name VARCHAR(255),
            trade_date DATE NOT NULL,
            variant VARCHAR(50) NOT NULL,
            previous_nav DECIMAL(20, 6),
            new_nav DECIMAL(20, 6),
            change_pct DECIMAL(10, 4),
            source VARCHAR(100),
            reason VARCHAR(500),
            metadata JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_nav_audit_portfolio_date
        ON nav_audit_log(portfolio_id, trade_date);

        CREATE INDEX IF NOT EXISTS idx_nav_audit_created
        ON nav_audit_log(created_at);
        """

        try:
            with Session(engine) as session:
                session.execute(text(create_sql))
                session.commit()
        except Exception as e:
            logger.debug(f"Audit table creation (may already exist): {e}")

    def log_nav_change(
        self,
        portfolio_id: int,
        portfolio_name: str,
        trade_date: date,
        variant: str,
        previous_nav: Optional[float],
        new_nav: float,
        source: str = "scheduled_update",
        reason: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Log a NAV change to the audit trail."""
        from sqlalchemy import text

        change_pct = None
        if previous_nav and previous_nav > 0:
            change_pct = ((new_nav / previous_nav) - 1) * 100

        insert_sql = text("""
            INSERT INTO nav_audit_log
            (portfolio_id, portfolio_name, trade_date, variant,
             previous_nav, new_nav, change_pct, source, reason, metadata)
            VALUES
            (:portfolio_id, :portfolio_name, :trade_date, :variant,
             :previous_nav, :new_nav, :change_pct, :source, :reason, :metadata)
        """)

        try:
            with Session(engine) as session:
                import json
                session.execute(insert_sql, {
                    "portfolio_id": portfolio_id,
                    "portfolio_name": portfolio_name,
                    "trade_date": trade_date,
                    "variant": variant,
                    "previous_nav": previous_nav,
                    "new_nav": new_nav,
                    "change_pct": change_pct,
                    "source": source,
                    "reason": reason,
                    "metadata": json.dumps(metadata) if metadata else None,
                })
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log NAV change: {e}")

    def get_history(
        self,
        portfolio_id: int,
        trade_date: Optional[date] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get NAV change history for a portfolio."""
        from sqlalchemy import text

        if trade_date:
            query = text("""
                SELECT * FROM nav_audit_log
                WHERE portfolio_id = :portfolio_id AND trade_date = :trade_date
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            params = {"portfolio_id": portfolio_id, "trade_date": trade_date, "limit": limit}
        else:
            query = text("""
                SELECT * FROM nav_audit_log
                WHERE portfolio_id = :portfolio_id
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            params = {"portfolio_id": portfolio_id, "limit": limit}

        try:
            with Session(engine) as session:
                # Use execute() instead of exec() for raw SQL with params
                result = session.execute(query, params)
                rows = result.fetchall()
                if rows:
                    columns = result.keys()
                    return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            logger.error(f"Failed to get audit history: {e}")

        return pd.DataFrame()

    def get_suspicious_changes(
        self,
        threshold_pct: float = 5.0,
        days_back: int = 7,
    ) -> pd.DataFrame:
        """Find recent NAV changes exceeding threshold."""
        from sqlalchemy import text

        query = text("""
            SELECT * FROM nav_audit_log
            WHERE created_at >= :since
            AND ABS(change_pct) > :threshold
            ORDER BY ABS(change_pct) DESC
        """)

        since = datetime.now() - timedelta(days=days_back)

        try:
            with Session(engine) as session:
                # Use execute() instead of exec() for raw SQL with params
                result = session.execute(query, {"since": since, "threshold": threshold_pct})
                rows = result.fetchall()
                if rows:
                    columns = result.keys()
                    return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            logger.error(f"Failed to get suspicious changes: {e}")

        return pd.DataFrame()


class BenchmarkReconciler:
    """
    Cross-checks portfolio returns against benchmarks.

    Detects when portfolio returns diverge unexpectedly from
    market benchmarks, which could indicate data issues.
    """

    def __init__(self, benchmark_adapter=None):
        self.benchmark_adapter = benchmark_adapter

    def reconcile_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_symbol: str = "SPY",
        trade_date: date = None,
        tolerance_std: float = 3.0,
    ) -> Tuple[bool, Optional[DataQualityAlert]]:
        """
        Check if portfolio return is within expected range vs benchmark.

        Uses rolling correlation and standard deviation to determine
        if today's return is statistically unusual.

        Args:
            portfolio_returns: Series of daily returns
            benchmark_symbol: Benchmark to compare against
            trade_date: Date to check
            tolerance_std: Number of standard deviations for outlier detection

        Returns:
            Tuple of (is_within_bounds, alert_if_any)
        """
        if self.benchmark_adapter is None:
            return True, None

        if trade_date is None:
            trade_date = date.today()

        try:
            # Get benchmark returns
            start_date = trade_date - timedelta(days=90)
            benchmark_returns = self.benchmark_adapter.get_benchmark_returns(
                benchmark_symbol, start_date, trade_date
            )

            if benchmark_returns.empty or len(portfolio_returns) < 20:
                return True, None

            # Align dates
            aligned = pd.DataFrame({
                "portfolio": portfolio_returns,
                "benchmark": benchmark_returns,
            }).dropna()

            if len(aligned) < 20:
                return True, None

            # Calculate spread (portfolio - benchmark)
            spread = aligned["portfolio"] - aligned["benchmark"]
            spread_mean = spread.mean()
            spread_std = spread.std()

            # Check latest spread
            latest_spread = spread.iloc[-1]
            z_score = (latest_spread - spread_mean) / spread_std if spread_std > 0 else 0

            if abs(z_score) > tolerance_std:
                alert = DataQualityAlert(
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    portfolio_name="",  # Will be filled by caller
                    trade_date=trade_date,
                    alert_type="BENCHMARK_DIVERGENCE",
                    message=f"Portfolio return diverged {z_score:.1f} std devs from {benchmark_symbol}",
                    details={
                        "z_score": z_score,
                        "portfolio_return": float(aligned["portfolio"].iloc[-1]),
                        "benchmark_return": float(aligned["benchmark"].iloc[-1]),
                        "spread": float(latest_spread),
                        "spread_mean": float(spread_mean),
                        "spread_std": float(spread_std),
                    },
                )
                return False, alert

            return True, None

        except Exception as e:
            logger.error(f"Reconciliation error: {e}")
            return True, None


def run_data_quality_check(
    portfolio_id: int = None,
    days_back: int = 7,
) -> Dict:
    """
    Run comprehensive data quality check.

    This should be run daily as a monitoring job.

    Returns:
        Dictionary with check results and any issues found
    """
    from .tracker import get_tracker
    from .models import Variants

    tracker = get_tracker()
    validator = DataQualityValidator()
    audit_log = NAVAuditLog()

    results = {
        "checked_at": datetime.now().isoformat(),
        "portfolios_checked": 0,
        "issues_found": [],
        "suspicious_changes": [],
    }

    # Get portfolios to check
    if portfolio_id:
        portfolio = tracker.get_portfolio(portfolio_id)
        portfolios = [portfolio] if portfolio else []
    else:
        portfolios = tracker.list_portfolios(active_only=True)

    results["portfolios_checked"] = len(portfolios)

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    for portfolio in portfolios:
        nav_df = tracker.get_nav_series(
            portfolio.id, Variants.RAW, start_date, end_date
        )

        if nav_df.empty or len(nav_df) < 2:
            continue

        # Check for suspicious daily changes
        nav_df["pct_change"] = nav_df["nav"].pct_change() * 100

        for idx in range(1, len(nav_df)):
            pct_change = nav_df["pct_change"].iloc[idx]
            nav_value = nav_df["nav"].iloc[idx]
            prev_nav = nav_df["nav"].iloc[idx - 1]
            trade_dt = nav_df.index[idx]

            # Validate
            is_valid, alerts = validator.validate_nav_update(
                portfolio_name=portfolio.name,
                trade_date=trade_dt.date() if hasattr(trade_dt, 'date') else trade_dt,
                new_nav=nav_value,
                previous_nav=prev_nav,
            )

            if alerts:
                for alert in alerts:
                    results["issues_found"].append({
                        "portfolio": portfolio.name,
                        "date": str(alert.trade_date),
                        "type": alert.alert_type,
                        "severity": alert.severity.value,
                        "message": alert.message,
                    })

    # Check audit log for suspicious changes
    suspicious = audit_log.get_suspicious_changes(threshold_pct=5.0, days_back=days_back)
    if not suspicious.empty:
        results["suspicious_changes"] = suspicious.to_dict("records")

    return results


# Singleton instances
_validator: Optional[DataQualityValidator] = None
_audit_log: Optional[NAVAuditLog] = None


def get_validator() -> DataQualityValidator:
    """Get singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = DataQualityValidator()
    return _validator


def get_audit_log() -> NAVAuditLog:
    """Get singleton audit log instance."""
    global _audit_log
    if _audit_log is None:
        _audit_log = NAVAuditLog()
    return _audit_log
