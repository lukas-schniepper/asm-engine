"""
Comprehensive test suite for Trading Calendar utilities.

Tests verify:
1. Core trading day logic (weekends, holidays)
2. All US federal holidays are correctly identified
3. Holiday observation rules (weekend shifts)
4. Date range functions
5. Navigation functions (last/next/previous trading day)
6. Regression tests for known dates

Run with:
    pytest tests/test_trading_calendar.py -v

Or directly:
    python tests/test_trading_calendar.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.trading_calendar import (
    is_trading_day,
    get_trading_days,
    get_last_trading_day,
    get_last_n_trading_days,
    get_next_trading_day,
    get_previous_trading_day,
    count_trading_days,
)


# =============================================================================
# Core Function Tests
# =============================================================================

def test_is_trading_day_weekday():
    """Test that regular weekdays (non-holiday) are trading days."""
    print("\n=== Test: is_trading_day - Weekday ===")

    # A regular Monday (not a holiday)
    test_date = date(2026, 1, 12)  # Monday, Jan 12, 2026
    assert is_trading_day(test_date), f"{test_date} should be a trading day"

    # A regular Friday
    test_date = date(2026, 1, 9)  # Friday
    assert is_trading_day(test_date), f"{test_date} should be a trading day"

    print("PASSED: Regular weekdays are trading days")


def test_is_trading_day_weekend():
    """Test that weekends are NOT trading days."""
    print("\n=== Test: is_trading_day - Weekend ===")

    # Saturday
    saturday = date(2026, 1, 3)
    assert not is_trading_day(saturday), f"{saturday} (Saturday) should NOT be a trading day"

    # Sunday
    sunday = date(2026, 1, 4)
    assert not is_trading_day(sunday), f"{sunday} (Sunday) should NOT be a trading day"

    print("PASSED: Weekends are not trading days")


# =============================================================================
# US Holiday Tests (2026)
# =============================================================================

def test_is_trading_day_new_years():
    """Test New Year's Day is not a trading day."""
    print("\n=== Test: is_trading_day - New Year's Day ===")

    # New Year's Day 2026 is Thursday, Jan 1
    new_years = date(2026, 1, 1)
    assert not is_trading_day(new_years), f"New Year's Day {new_years} should NOT be a trading day"

    print("PASSED: New Year's Day is not a trading day")


def test_is_trading_day_mlk_day():
    """Test MLK Day (3rd Monday in January) is not a trading day."""
    print("\n=== Test: is_trading_day - MLK Day ===")

    # MLK Day 2026 is Monday, Jan 19
    mlk_day = date(2026, 1, 19)
    assert not is_trading_day(mlk_day), f"MLK Day {mlk_day} should NOT be a trading day"

    print("PASSED: MLK Day is not a trading day")


def test_is_trading_day_presidents_day():
    """Test Presidents' Day (3rd Monday in February) is not a trading day."""
    print("\n=== Test: is_trading_day - Presidents' Day ===")

    # Presidents' Day 2026 is Monday, Feb 16
    presidents_day = date(2026, 2, 16)
    assert not is_trading_day(presidents_day), f"Presidents' Day {presidents_day} should NOT be a trading day"

    print("PASSED: Presidents' Day is not a trading day")


def test_is_trading_day_memorial_day():
    """Test Memorial Day (last Monday in May) is not a trading day."""
    print("\n=== Test: is_trading_day - Memorial Day ===")

    # Memorial Day 2026 is Monday, May 25
    memorial_day = date(2026, 5, 25)
    assert not is_trading_day(memorial_day), f"Memorial Day {memorial_day} should NOT be a trading day"

    print("PASSED: Memorial Day is not a trading day")


def test_is_trading_day_juneteenth():
    """Test Juneteenth (June 19) is not a trading day."""
    print("\n=== Test: is_trading_day - Juneteenth ===")

    # Juneteenth 2026 is Friday, June 19
    juneteenth = date(2026, 6, 19)
    assert not is_trading_day(juneteenth), f"Juneteenth {juneteenth} should NOT be a trading day"

    print("PASSED: Juneteenth is not a trading day")


def test_is_trading_day_independence_day():
    """Test Independence Day (July 4) is not a trading day."""
    print("\n=== Test: is_trading_day - Independence Day ===")

    # Independence Day 2026 is Saturday -> observed Friday July 3
    july_4 = date(2026, 7, 4)
    july_3_observed = date(2026, 7, 3)

    # The actual July 4 is a weekend anyway
    assert not is_trading_day(july_4), f"July 4 {july_4} (Saturday) should NOT be a trading day"
    # The observed holiday on Friday
    assert not is_trading_day(july_3_observed), f"Observed Independence Day {july_3_observed} should NOT be a trading day"

    print("PASSED: Independence Day (and observed) is not a trading day")


def test_is_trading_day_labor_day():
    """Test Labor Day (1st Monday in September) is not a trading day."""
    print("\n=== Test: is_trading_day - Labor Day ===")

    # Labor Day 2026 is Monday, Sep 7
    labor_day = date(2026, 9, 7)
    assert not is_trading_day(labor_day), f"Labor Day {labor_day} should NOT be a trading day"

    print("PASSED: Labor Day is not a trading day")


def test_is_trading_day_thanksgiving():
    """Test Thanksgiving (4th Thursday in November) is not a trading day."""
    print("\n=== Test: is_trading_day - Thanksgiving ===")

    # Thanksgiving 2026 is Thursday, Nov 26
    thanksgiving = date(2026, 11, 26)
    assert not is_trading_day(thanksgiving), f"Thanksgiving {thanksgiving} should NOT be a trading day"

    print("PASSED: Thanksgiving is not a trading day")


def test_is_trading_day_christmas():
    """Test Christmas Day (Dec 25) is not a trading day."""
    print("\n=== Test: is_trading_day - Christmas ===")

    # Christmas 2026 is Friday, Dec 25
    christmas = date(2026, 12, 25)
    assert not is_trading_day(christmas), f"Christmas {christmas} should NOT be a trading day"

    print("PASSED: Christmas is not a trading day")


# =============================================================================
# Holiday Observation Tests
# =============================================================================

def test_observed_holiday_saturday():
    """Test that Saturday holidays are observed on Friday."""
    print("\n=== Test: Observed Holiday - Saturday -> Friday ===")

    # July 4, 2026 falls on Saturday -> observed Friday July 3
    friday_observed = date(2026, 7, 3)
    assert not is_trading_day(friday_observed), f"Friday {friday_observed} should be closed (observed July 4)"

    print("PASSED: Saturday holidays observed on Friday")


def test_observed_holiday_sunday():
    """Test that Sunday holidays are observed on Monday."""
    print("\n=== Test: Observed Holiday - Sunday -> Monday ===")

    # Christmas 2022 was Sunday -> observed Monday Dec 26
    monday_observed = date(2022, 12, 26)
    assert not is_trading_day(monday_observed), f"Monday {monday_observed} should be closed (observed Christmas)"

    print("PASSED: Sunday holidays observed on Monday")


# =============================================================================
# Range Function Tests
# =============================================================================

def test_get_trading_days_simple_week():
    """Test getting trading days for a simple week (no holidays)."""
    print("\n=== Test: get_trading_days - Simple Week ===")

    # Week of Jan 5-9, 2026 (Mon-Fri, no holidays)
    start = date(2026, 1, 5)
    end = date(2026, 1, 9)

    trading_days = get_trading_days(start, end)

    assert len(trading_days) == 5, f"Expected 5 trading days, got {len(trading_days)}"
    assert trading_days[0] == date(2026, 1, 5), "First day should be Monday"
    assert trading_days[-1] == date(2026, 1, 9), "Last day should be Friday"

    print(f"PASSED: Got {len(trading_days)} trading days for Mon-Fri week")


def test_get_trading_days_with_holiday():
    """Test getting trading days for a week with a holiday."""
    print("\n=== Test: get_trading_days - Week with Holiday ===")

    # Week of Jan 19-23, 2026 (contains MLK Day on Monday)
    start = date(2026, 1, 19)
    end = date(2026, 1, 23)

    trading_days = get_trading_days(start, end)

    assert len(trading_days) == 4, f"Expected 4 trading days (MLK Day off), got {len(trading_days)}"
    assert date(2026, 1, 19) not in trading_days, "MLK Day should not be in trading days"

    print(f"PASSED: Got {len(trading_days)} trading days (excluding MLK Day)")


def test_get_trading_days_empty_range():
    """Test getting trading days when start > end."""
    print("\n=== Test: get_trading_days - Empty Range ===")

    start = date(2026, 1, 10)
    end = date(2026, 1, 5)  # Before start

    trading_days = get_trading_days(start, end)

    assert len(trading_days) == 0, f"Expected 0 trading days for invalid range, got {len(trading_days)}"

    print("PASSED: Empty range returns empty list")


def test_get_trading_days_year_boundary():
    """Test getting trading days crossing year boundary (includes New Year)."""
    print("\n=== Test: get_trading_days - Year Boundary ===")

    # Dec 30, 2025 to Jan 5, 2026 (includes New Year's Day)
    start = date(2025, 12, 29)
    end = date(2026, 1, 2)

    trading_days = get_trading_days(start, end)

    # Dec 29 (Mon), Dec 30 (Tue), Dec 31 (Wed), Jan 2 (Fri) = 4 days
    # Jan 1 is New Year's Day (holiday)
    assert date(2026, 1, 1) not in trading_days, "New Year's Day should not be in trading days"
    assert len(trading_days) == 4, f"Expected 4 trading days, got {len(trading_days)}"

    print(f"PASSED: Year boundary handled correctly, got {len(trading_days)} days")


# =============================================================================
# Navigation Function Tests
# =============================================================================

def test_get_last_trading_day_on_trading_day():
    """Test get_last_trading_day when reference is a trading day."""
    print("\n=== Test: get_last_trading_day - On Trading Day ===")

    # Friday Jan 9, 2026 is a trading day
    friday = date(2026, 1, 9)
    result = get_last_trading_day(friday)

    assert result == friday, f"Expected same day {friday}, got {result}"

    print("PASSED: Returns same day if it's a trading day")


def test_get_last_trading_day_on_weekend():
    """Test get_last_trading_day when reference is a weekend."""
    print("\n=== Test: get_last_trading_day - On Weekend ===")

    # Saturday Jan 10, 2026
    saturday = date(2026, 1, 10)
    result = get_last_trading_day(saturday)

    # Should return Friday Jan 9
    expected = date(2026, 1, 9)
    assert result == expected, f"Expected {expected}, got {result}"

    print("PASSED: Returns previous Friday from Saturday")


def test_get_last_trading_day_on_holiday():
    """Test get_last_trading_day when reference is a holiday."""
    print("\n=== Test: get_last_trading_day - On Holiday ===")

    # New Year's Day 2026 (Thursday)
    new_years = date(2026, 1, 1)
    result = get_last_trading_day(new_years)

    # Should return Dec 31, 2025 (Wednesday)
    expected = date(2025, 12, 31)
    assert result == expected, f"Expected {expected}, got {result}"

    print("PASSED: Returns previous trading day from holiday")


def test_get_last_n_trading_days():
    """Test getting the last N trading days."""
    print("\n=== Test: get_last_n_trading_days ===")

    # Get last 5 trading days from Jan 9, 2026 (Friday)
    reference = date(2026, 1, 9)
    days = get_last_n_trading_days(5, reference)

    assert len(days) == 5, f"Expected 5 days, got {len(days)}"
    assert days[-1] == date(2026, 1, 9), f"Last day should be {reference}"
    assert days[0] == date(2026, 1, 5), f"First day should be Monday Jan 5"
    assert days[0] < days[-1], "Days should be in chronological order"

    print(f"PASSED: Got {len(days)} trading days in correct order")


def test_get_next_trading_day_on_weekend():
    """Test get_next_trading_day from a weekend."""
    print("\n=== Test: get_next_trading_day - From Weekend ===")

    # Saturday Jan 3, 2026
    saturday = date(2026, 1, 3)
    result = get_next_trading_day(saturday)

    # Should return Monday Jan 5
    expected = date(2026, 1, 5)
    assert result == expected, f"Expected {expected}, got {result}"

    print("PASSED: Returns next Monday from Saturday")


def test_get_previous_trading_day():
    """Test get_previous_trading_day (strictly before)."""
    print("\n=== Test: get_previous_trading_day ===")

    # From Monday Jan 5, 2026
    monday = date(2026, 1, 5)
    result = get_previous_trading_day(monday)

    # Should return Friday Jan 2 (not Thursday Jan 1 which is New Year's)
    expected = date(2026, 1, 2)
    assert result == expected, f"Expected {expected}, got {result}"

    print("PASSED: Returns previous trading day (skipping weekend)")


# =============================================================================
# Regression Tests - Known Holidays
# =============================================================================

def test_regression_2024_holidays():
    """Verify all 2024 US market holidays are correctly identified."""
    print("\n=== Test: Regression - 2024 Holidays ===")

    holidays_2024 = [
        (date(2024, 1, 1), "New Year's Day"),
        (date(2024, 1, 15), "MLK Day"),
        (date(2024, 2, 19), "Presidents' Day"),
        (date(2024, 5, 27), "Memorial Day"),
        (date(2024, 6, 19), "Juneteenth"),
        (date(2024, 7, 4), "Independence Day"),
        (date(2024, 9, 2), "Labor Day"),
        (date(2024, 11, 28), "Thanksgiving"),
        (date(2024, 12, 25), "Christmas"),
    ]

    for holiday_date, name in holidays_2024:
        assert not is_trading_day(holiday_date), f"2024 {name} ({holiday_date}) should be a holiday"

    print(f"PASSED: All {len(holidays_2024)} holidays in 2024 correctly identified")


def test_regression_2025_holidays():
    """Verify all 2025 US market holidays are correctly identified."""
    print("\n=== Test: Regression - 2025 Holidays ===")

    holidays_2025 = [
        (date(2025, 1, 1), "New Year's Day"),
        (date(2025, 1, 20), "MLK Day"),
        (date(2025, 2, 17), "Presidents' Day"),
        (date(2025, 5, 26), "Memorial Day"),
        (date(2025, 6, 19), "Juneteenth"),
        (date(2025, 7, 4), "Independence Day"),
        (date(2025, 9, 1), "Labor Day"),
        (date(2025, 11, 27), "Thanksgiving"),
        (date(2025, 12, 25), "Christmas"),
    ]

    for holiday_date, name in holidays_2025:
        assert not is_trading_day(holiday_date), f"2025 {name} ({holiday_date}) should be a holiday"

    print(f"PASSED: All {len(holidays_2025)} holidays in 2025 correctly identified")


def test_regression_2026_holidays():
    """Verify all 2026 US market holidays are correctly identified."""
    print("\n=== Test: Regression - 2026 Holidays ===")

    holidays_2026 = [
        (date(2026, 1, 1), "New Year's Day"),
        (date(2026, 1, 19), "MLK Day"),
        (date(2026, 2, 16), "Presidents' Day"),
        (date(2026, 5, 25), "Memorial Day"),
        (date(2026, 6, 19), "Juneteenth"),
        (date(2026, 7, 3), "Independence Day (observed)"),  # July 4 is Saturday
        (date(2026, 9, 7), "Labor Day"),
        (date(2026, 11, 26), "Thanksgiving"),
        (date(2026, 12, 25), "Christmas"),
    ]

    for holiday_date, name in holidays_2026:
        assert not is_trading_day(holiday_date), f"2026 {name} ({holiday_date}) should be a holiday"

    print(f"PASSED: All {len(holidays_2026)} holidays in 2026 correctly identified")


# =============================================================================
# Integration Tests - Data Quality Scenarios
# =============================================================================

def test_holiday_week_trading_days_count():
    """Test that holiday weeks have correct trading day count for data quality checks."""
    print("\n=== Test: Holiday Week Trading Day Count ===")

    # New Year's week 2026: Dec 29 - Jan 2 (spans year boundary)
    # Mon Dec 29, Tue Dec 30, Wed Dec 31, Thu Jan 1 (holiday), Fri Jan 2
    start = date(2025, 12, 29)
    end = date(2026, 1, 2)

    days = get_trading_days(start, end)
    count = len(days)

    assert count == 4, f"New Year's week should have 4 trading days, got {count}"

    # MLK Day week 2026
    start = date(2026, 1, 19)  # MLK Day Monday
    end = date(2026, 1, 23)    # Friday

    days = get_trading_days(start, end)
    count = len(days)

    assert count == 4, f"MLK Day week should have 4 trading days, got {count}"

    print("PASSED: Holiday weeks have correct trading day counts")


def test_count_trading_days_function():
    """Test the count_trading_days helper function."""
    print("\n=== Test: count_trading_days ===")

    # Normal week
    count = count_trading_days(date(2026, 1, 5), date(2026, 1, 9))
    assert count == 5, f"Expected 5 trading days, got {count}"

    # Week with MLK Day
    count = count_trading_days(date(2026, 1, 19), date(2026, 1, 23))
    assert count == 4, f"Expected 4 trading days (MLK week), got {count}"

    print("PASSED: count_trading_days works correctly")


def test_last_5_trading_days_holiday_week():
    """Test get_last_n_trading_days during holiday weeks (the main use case)."""
    print("\n=== Test: Last 5 Trading Days - Holiday Week ===")

    # Simulating Jan 8, 2026 scenario (the original bug)
    # Last 5 trading days should be: Jan 2, 5, 6, 7, 8
    reference = date(2026, 1, 8)
    days = get_last_n_trading_days(5, reference)

    expected = [
        date(2026, 1, 2),  # Friday
        date(2026, 1, 5),  # Monday
        date(2026, 1, 6),  # Tuesday
        date(2026, 1, 7),  # Wednesday
        date(2026, 1, 8),  # Thursday
    ]

    assert days == expected, f"Expected {expected}, got {days}"
    assert date(2026, 1, 1) not in days, "New Year's Day should NOT be in the list"

    print("PASSED: Correctly identifies 5 trading days during New Year's week")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("TRADING CALENDAR TEST SUITE")
    print("=" * 60)

    tests = [
        # Core tests
        test_is_trading_day_weekday,
        test_is_trading_day_weekend,

        # Holiday tests
        test_is_trading_day_new_years,
        test_is_trading_day_mlk_day,
        test_is_trading_day_presidents_day,
        test_is_trading_day_memorial_day,
        test_is_trading_day_juneteenth,
        test_is_trading_day_independence_day,
        test_is_trading_day_labor_day,
        test_is_trading_day_thanksgiving,
        test_is_trading_day_christmas,

        # Observation tests
        test_observed_holiday_saturday,
        test_observed_holiday_sunday,

        # Range tests
        test_get_trading_days_simple_week,
        test_get_trading_days_with_holiday,
        test_get_trading_days_empty_range,
        test_get_trading_days_year_boundary,

        # Navigation tests
        test_get_last_trading_day_on_trading_day,
        test_get_last_trading_day_on_weekend,
        test_get_last_trading_day_on_holiday,
        test_get_last_n_trading_days,
        test_get_next_trading_day_on_weekend,
        test_get_previous_trading_day,

        # Regression tests
        test_regression_2024_holidays,
        test_regression_2025_holidays,
        test_regression_2026_holidays,

        # Integration tests
        test_holiday_week_trading_days_count,
        test_count_trading_days_function,
        test_last_5_trading_days_holiday_week,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\nFAILED: {test.__name__}")
            print(f"  Error: {e}")
        except Exception as e:
            failed += 1
            print(f"\nERROR: {test.__name__}")
            print(f"  Exception: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
