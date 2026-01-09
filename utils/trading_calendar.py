"""
Shared Trading Calendar Utilities for ASM Engine.

This module provides a centralized, authoritative implementation of trading day
logic for the US stock market. It uses pandas USFederalHolidayCalendar for
comprehensive holiday coverage.

US Market Holidays (NYSE):
- New Year's Day (Jan 1)
- Martin Luther King Jr. Day (3rd Monday in January)
- Presidents' Day (3rd Monday in February)
- Good Friday (Friday before Easter - NOTE: not in USFederalHolidayCalendar)
- Memorial Day (Last Monday in May)
- Juneteenth (June 19)
- Independence Day (July 4)
- Labor Day (1st Monday in September)
- Thanksgiving Day (4th Thursday in November)
- Christmas Day (Dec 25)

When holidays fall on weekends, they are typically observed on the nearest weekday:
- Saturday holidays -> observed Friday
- Sunday holidays -> observed Monday

Usage:
    from utils.trading_calendar import is_trading_day, get_trading_days

    # Check if a date is a trading day
    if is_trading_day(date(2026, 1, 1)):
        print("Market is open")

    # Get trading days in a range
    trading_days = get_trading_days(date(2026, 1, 1), date(2026, 1, 10))

    # Get the last 5 trading days
    last_5 = get_last_n_trading_days(5)
"""

from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


# Cache the holiday calendar to avoid repeated instantiation
@lru_cache(maxsize=1)
def _get_holiday_calendar() -> USFederalHolidayCalendar:
    """Get cached US Federal Holiday Calendar instance."""
    return USFederalHolidayCalendar()


@lru_cache(maxsize=10)
def _get_holidays_for_year(year: int) -> set[date]:
    """Get set of holiday dates for a given year (cached)."""
    cal = _get_holiday_calendar()
    holidays = cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
    return {d.date() for d in holidays}


def is_trading_day(check_date: date) -> bool:
    """
    Check if a date is a US stock market trading day.

    A trading day is a weekday (Monday-Friday) that is not a US federal holiday.

    Args:
        check_date: The date to check

    Returns:
        True if the date is a trading day, False otherwise

    Examples:
        >>> is_trading_day(date(2026, 1, 2))  # Friday after New Year
        True
        >>> is_trading_day(date(2026, 1, 1))  # New Year's Day
        False
        >>> is_trading_day(date(2026, 1, 3))  # Saturday
        False
    """
    # Weekend check
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Holiday check using cached holidays for the year
    holidays = _get_holidays_for_year(check_date.year)
    return check_date not in holidays


def get_trading_days(start_date: date, end_date: date) -> list[date]:
    """
    Get list of trading days between start and end dates (inclusive).

    Uses pandas CustomBusinessDay with USFederalHolidayCalendar for accurate
    holiday handling.

    Args:
        start_date: Start of the date range (inclusive)
        end_date: End of the date range (inclusive)

    Returns:
        List of trading days in chronological order

    Examples:
        >>> days = get_trading_days(date(2026, 1, 5), date(2026, 1, 9))
        >>> len(days)  # Mon-Fri = 5 days (no holidays)
        5
    """
    if start_date > end_date:
        return []

    us_bd = CustomBusinessDay(calendar=_get_holiday_calendar())
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    return [d.date() for d in trading_days]


def get_last_trading_day(reference_date: Optional[date] = None) -> date:
    """
    Get the most recent trading day on or before the reference date.

    If reference_date is a trading day, it is returned.
    Otherwise, returns the most recent previous trading day.

    Args:
        reference_date: The date to check from (defaults to today)

    Returns:
        The most recent trading day on or before reference_date

    Examples:
        >>> get_last_trading_day(date(2026, 1, 3))  # Saturday
        date(2026, 1, 2)  # Returns Friday
        >>> get_last_trading_day(date(2026, 1, 2))  # Friday
        date(2026, 1, 2)  # Returns same day
    """
    if reference_date is None:
        reference_date = date.today()

    check_date = reference_date
    max_lookback = 10  # Safety limit (max consecutive non-trading days)

    for _ in range(max_lookback):
        if is_trading_day(check_date):
            return check_date
        check_date -= timedelta(days=1)

    # Fallback (should never reach here in normal circumstances)
    return check_date


def get_previous_trading_day(reference_date: Optional[date] = None) -> date:
    """
    Get the trading day strictly before the reference date.

    Unlike get_last_trading_day, this always returns a day before reference_date,
    even if reference_date itself is a trading day.

    Args:
        reference_date: The date to start from (defaults to today)

    Returns:
        The most recent trading day before reference_date

    Examples:
        >>> get_previous_trading_day(date(2026, 1, 5))  # Monday
        date(2026, 1, 2)  # Returns previous Friday
    """
    if reference_date is None:
        reference_date = date.today()

    check_date = reference_date - timedelta(days=1)
    return get_last_trading_day(check_date)


def get_next_trading_day(reference_date: Optional[date] = None) -> date:
    """
    Get the next trading day on or after the reference date.

    If reference_date is a trading day, it is returned.
    Otherwise, returns the next future trading day.

    Args:
        reference_date: The date to check from (defaults to today)

    Returns:
        The next trading day on or after reference_date

    Examples:
        >>> get_next_trading_day(date(2026, 1, 3))  # Saturday
        date(2026, 1, 5)  # Returns Monday
    """
    if reference_date is None:
        reference_date = date.today()

    check_date = reference_date
    max_lookahead = 10  # Safety limit

    for _ in range(max_lookahead):
        if is_trading_day(check_date):
            return check_date
        check_date += timedelta(days=1)

    # Fallback (should never reach here in normal circumstances)
    return check_date


def get_last_n_trading_days(n: int, reference_date: Optional[date] = None) -> list[date]:
    """
    Get the last N trading days ending on or before reference_date.

    This is useful for data quality checks that need to verify
    NAV entries exist for each of the last N trading days.

    Args:
        n: Number of trading days to return
        reference_date: End date for the range (defaults to last trading day)

    Returns:
        List of N trading days in chronological order (oldest first)

    Examples:
        >>> days = get_last_n_trading_days(5, date(2026, 1, 9))
        >>> len(days)
        5
        >>> days[0] < days[-1]  # Oldest first
        True
    """
    if n <= 0:
        return []

    if reference_date is None:
        reference_date = date.today()

    # Start from the last trading day on or before reference_date
    end_date = get_last_trading_day(reference_date)

    # Collect N trading days going backwards
    trading_days = []
    check_date = end_date

    # Need to look back far enough to find N trading days
    # Worst case: ~2 trading days per 3 calendar days
    max_lookback = n * 2 + 10

    for _ in range(max_lookback):
        if is_trading_day(check_date):
            trading_days.append(check_date)
            if len(trading_days) >= n:
                break
        check_date -= timedelta(days=1)

    # Return in chronological order (oldest first)
    return sorted(trading_days)


def count_trading_days(start_date: date, end_date: date) -> int:
    """
    Count the number of trading days between two dates (inclusive).

    Args:
        start_date: Start of the date range
        end_date: End of the date range

    Returns:
        Number of trading days in the range
    """
    return len(get_trading_days(start_date, end_date))
