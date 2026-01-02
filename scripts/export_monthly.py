"""
AlphaWizzard Monthly Post Data Exporter (v2)
============================================
Run this script to export all data needed for generating a monthly marketing post.
The output JSON file can be uploaded to Claude for post generation.

NEW in v2: Also fetches CURRENT holdings to show next month's portfolio changes.

Usage:
    python export_monthly.py --month 2025-12
    python export_monthly.py --month 2025-12 --portfolio-id 8

Requirements:
    pip install supabase boto3 pandas python-dotenv requests sqlalchemy

Environment variables (.env file):
    DATABASE_URL=postgresql://...
    AWS_ACCESS_KEY_ID=xxx
    AWS_SECRET_ACCESS_KEY=xxx
    EODHD_API_KEY=xxx
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path and load secrets from .streamlit/secrets.toml
project_root = Path(__file__).parent.parent

def _load_secrets():
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)

        for key, value in secrets.items():
            if key not in os.environ and isinstance(value, str):
                os.environ[key] = value

_load_secrets()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "alpha-state-machine2030")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")


class DecimalEncoder(json.JSONEncoder):
    """Handle Decimal and date types in JSON encoding"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, 'isoformat'):  # date, datetime
            return obj.isoformat()
        return super().default(obj)


class PostgresClient:
    """PostgreSQL client using SQLAlchemy (replaces Supabase REST API)"""
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(database_url)

    def table(self, table_name: str):
        return PostgresTable(self, table_name)


class PostgresTable:
    """PostgreSQL table query builder (mimics Supabase API)"""
    def __init__(self, client: PostgresClient, table_name: str):
        self.client = client
        self.table_name = table_name
        self._conditions = []
        self._order_by = None
        self._order_desc = False
        self._limit = None
        self._in_column = None
        self._in_values = None

    def select(self, columns: str = '*'):
        return self

    def eq(self, column: str, value):
        self._conditions.append((column, '=', value))
        return self

    def gte(self, column: str, value):
        self._conditions.append((column, '>=', value))
        return self

    def lte(self, column: str, value):
        self._conditions.append((column, '<=', value))
        return self

    def in_(self, column: str, values: list):
        self._in_column = column
        self._in_values = values
        return self

    def order(self, column: str, desc: bool = False):
        self._order_by = column
        self._order_desc = desc
        return self

    def limit(self, count: int):
        self._limit = count
        return self

    def execute(self):
        from sqlalchemy import text

        sql = f"SELECT * FROM {self.table_name}"
        params = {}

        where_clauses = []
        for i, (col, op, val) in enumerate(self._conditions):
            param_name = f"p{i}"
            where_clauses.append(f"{col} {op} :{param_name}")
            params[param_name] = val

        if self._in_column and self._in_values:
            placeholders = ', '.join([f":in_{i}" for i in range(len(self._in_values))])
            where_clauses.append(f"{self._in_column} IN ({placeholders})")
            for i, v in enumerate(self._in_values):
                params[f"in_{i}"] = v

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        if self._order_by:
            sql += f" ORDER BY {self._order_by}"
            if self._order_desc:
                sql += " DESC"

        if self._limit:
            sql += f" LIMIT {self._limit}"

        with self.client.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result]

        class Result:
            def __init__(self, data):
                self.data = data

        return Result(rows)


def get_supabase_client():
    """Initialize database client using PostgreSQL directly"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    return PostgresClient(database_url)


def get_s3_client():
    """Initialize S3 client"""
    import boto3
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-1'
    )


def get_period_dates(month_str: str):
    """Parse month string and return start/end dates"""
    year, month = map(int, month_str.split('-'))
    start_date = datetime(year, month, 1)
    
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_next_month_date(month_str: str):
    """Get the first day of next month (for fetching new holdings)"""
    year, month = map(int, month_str.split('-'))
    if month == 12:
        return f"{year + 1}-01-15"  # Mid-January
    else:
        return f"{year}-{month + 1:02d}-15"  # Mid next month


def get_ytd_start(month_str: str):
    """Get Jan 1 of the year"""
    year = int(month_str.split('-')[0])
    return f"{year}-01-01"


def fetch_nav_data(supabase, start_date: str, end_date: str, portfolio_id: int = None):
    """Fetch NAV data for the period"""
    print(f"  Fetching NAV data from {start_date} to {end_date}...")

    query = supabase.table('portfolio_daily_nav') \
        .select('*') \
        .gte('trade_date', start_date) \
        .lte('trade_date', end_date)
    
    if portfolio_id:
        query = query.eq('portfolio_id', portfolio_id)
    
    response = query.order('trade_date').execute()
    return response.data


def fetch_ytd_nav(supabase, ytd_start: str, end_date: str, portfolio_id: int = None):
    """Fetch YTD NAV data"""
    print(f"  Fetching YTD NAV from {ytd_start}...")

    query = supabase.table('portfolio_daily_nav') \
        .select('*') \
        .gte('trade_date', ytd_start) \
        .lte('trade_date', end_date)
    
    if portfolio_id:
        query = query.eq('portfolio_id', portfolio_id)
    
    response = query.order('trade_date').execute()
    return response.data


def fetch_holdings(supabase, date: str, portfolio_id: int = None):
    """Fetch holdings as of a specific date"""
    print(f"  Fetching holdings as of {date}...")

    query = supabase.table('portfolio_holdings') \
        .select('*') \
        .lte('effective_date', date)
    
    if portfolio_id:
        query = query.eq('portfolio_id', portfolio_id)
    
    response = query.order('effective_date', desc=True).limit(100).execute()

    if not response.data:
        return []

    latest_date = response.data[0]['effective_date']
    
    # Filter by portfolio_id if specified
    if portfolio_id:
        return [h for h in response.data if h['effective_date'] == latest_date and h.get('portfolio_id') == portfolio_id]
    return [h for h in response.data if h['effective_date'] == latest_date]


def fetch_current_holdings(supabase, portfolio_id: int = None):
    """Fetch the most recent holdings (current portfolio)"""
    print(f"  Fetching CURRENT holdings (latest available)...")

    query = supabase.table('portfolio_holdings') \
        .select('*')
    
    if portfolio_id:
        query = query.eq('portfolio_id', portfolio_id)
    
    response = query.order('effective_date', desc=True).limit(100).execute()

    if not response.data:
        return []

    latest_date = response.data[0]['effective_date']
    print(f"    Latest holdings date: {latest_date}")
    
    if portfolio_id:
        return [h for h in response.data if h['effective_date'] == latest_date and h.get('portfolio_id') == portfolio_id]
    return [h for h in response.data if h['effective_date'] == latest_date]


def fetch_overlay_signals(supabase, start_date: str, end_date: str):
    """Fetch overlay signals for the period"""
    print(f"  Fetching overlay signals...")

    response = supabase.table('overlay_signals') \
        .select('*') \
        .gte('trade_date', start_date) \
        .lte('trade_date', end_date) \
        .order('trade_date') \
        .execute()

    return response.data


def fetch_current_overlay(supabase):
    """Fetch most recent overlay signal"""
    print(f"  Fetching current overlay status...")

    response = supabase.table('overlay_signals') \
        .select('*') \
        .order('trade_date', desc=True) \
        .limit(1) \
        .execute()

    return response.data[0] if response.data else None


def fetch_ticker_info(supabase, tickers: list):
    """Fetch ticker metadata"""
    if not tickers:
        return []
    
    print(f"  Fetching ticker info for {len(tickers)} stocks...")
    
    response = supabase.table('ticker_info') \
        .select('*') \
        .in_('ticker', tickers) \
        .execute()
    
    return response.data


def fetch_s3_file(s3_client, key: str):
    """Fetch a file from S3"""
    print(f"  Fetching s3://{S3_BUCKET}/{key}...")
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        content = response['Body'].read()
        
        if key.endswith('.json'):
            return json.loads(content)
        elif key.endswith('.csv'):
            import io
            return pd.read_csv(io.BytesIO(content)).to_dict('records')
        elif key.endswith('.parquet'):
            import io
            return pd.read_parquet(io.BytesIO(content)).to_dict('records')
        else:
            return content.decode('utf-8')
    except Exception as e:
        print(f"    Warning: Could not fetch {key}: {e}")
        return None


def fetch_spy_benchmark(s3_client, start_date: str, end_date: str):
    """Fetch SPY benchmark data from S3"""
    spy_data = fetch_s3_file(s3_client, 'spy.csv')
    
    if spy_data:
        filtered = [
            row for row in spy_data 
            if start_date <= row.get('date', row.get('Date', '')) <= end_date
        ]
        return filtered
    return []


def fetch_eodhd_fundamentals(ticker: str):
    """Fetch fundamental data from EODHD"""
    if not EODHD_API_KEY:
        print(f"    Warning: No EODHD API key, skipping {ticker}")
        return None
    
    print(f"    Fetching EODHD fundamentals for {ticker}...")
    
    url = f"https://eodhd.com/api/fundamentals/{ticker}.US"
    params = {'api_token': EODHD_API_KEY, 'fmt': 'json'}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            
            general = data.get('General', {})
            highlights = data.get('Highlights', {})
            valuation = data.get('Valuation', {})
            analyst = data.get('AnalystRatings', {})
            
            return {
                'ticker': ticker,
                'name': general.get('Name'),
                'description': general.get('Description'),
                'sector': general.get('Sector'),
                'industry': general.get('Industry'),
                'market_cap': highlights.get('MarketCapitalization'),
                'pe_ratio': highlights.get('PERatio'),
                'peg_ratio': highlights.get('PEGRatio'),
                'eps_current': highlights.get('EarningsShare'),
                'eps_estimate_current_year': highlights.get('EPSEstimateCurrentYear'),
                'eps_estimate_next_year': highlights.get('EPSEstimateNextYear'),
                'profit_margin': highlights.get('ProfitMargin'),
                'operating_margin': highlights.get('OperatingMarginTTM'),
                'return_on_equity': highlights.get('ReturnOnEquityTTM'),
                'revenue_ttm': highlights.get('RevenueTTM'),
                'revenue_growth_yoy': highlights.get('QuarterlyRevenueGrowthYOY'),
                'ebitda': highlights.get('EBITDA'),
                'dividend_yield': highlights.get('DividendYield'),
                'beta': highlights.get('Beta'),
                'week_52_high': highlights.get('52WeekHigh'),
                'week_52_low': highlights.get('52WeekLow'),
                'analyst_target_price': valuation.get('TargetPrice'),
                'analyst_rating': analyst.get('Rating'),
                'analyst_buy': analyst.get('Buy'),
                'analyst_hold': analyst.get('Hold'),
                'analyst_sell': analyst.get('Sell'),
                'analyst_strong_buy': analyst.get('StrongBuy'),
                'analyst_strong_sell': analyst.get('StrongSell'),
            }
    except Exception as e:
        print(f"    Error fetching {ticker}: {e}")
    
    return None


def fetch_eodhd_news(ticker: str, limit: int = 5):
    """Fetch recent news from EODHD"""
    if not EODHD_API_KEY:
        return []
    
    url = f"https://eodhd.com/api/news"
    params = {
        'api_token': EODHD_API_KEY,
        's': f'{ticker}.US',
        'limit': limit,
        'fmt': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            news = response.json()
            return [
                {
                    'title': item.get('title'),
                    'date': item.get('date'),
                    'sentiment': item.get('sentiment', {}).get('polarity')
                }
                for item in news[:limit]
            ]
    except Exception as e:
        print(f"    Error fetching news for {ticker}: {e}")
    
    return []


def calculate_performance(nav_data: list, variant: str = 'conservative'):
    """Calculate performance metrics from NAV data for a specific variant"""
    if not nav_data:
        return None
    
    # Filter by variant
    filtered = [n for n in nav_data if n.get('variant') == variant]
    
    if len(filtered) < 2:
        return None
    
    nav_sorted = sorted(filtered, key=lambda x: x['trade_date'])

    start_nav = nav_sorted[0].get('nav') or nav_sorted[0].get('total_value')
    end_nav = nav_sorted[-1].get('nav') or nav_sorted[-1].get('total_value')

    if not start_nav or not end_nav:
        return None

    period_return = ((float(end_nav) / float(start_nav)) - 1) * 100

    peak = float(start_nav)
    max_dd = 0
    for row in nav_sorted:
        nav = float(row.get('nav') or row.get('total_value'))
        if nav > peak:
            peak = nav
        dd = ((nav - peak) / peak) * 100
        if dd < max_dd:
            max_dd = dd

    return {
        'start_date': str(nav_sorted[0]['trade_date']),
        'end_date': str(nav_sorted[-1]['trade_date']),
        'start_nav': float(start_nav),
        'end_nav': float(end_nav),
        'return_pct': round(period_return, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'trading_days': len(nav_sorted),
        'variant': variant
    }


def identify_portfolio_changes(holdings_old: list, holdings_new: list):
    """Identify new, closed, and maintained positions between two holdings snapshots"""
    old_tickers = {h['ticker'] for h in holdings_old}
    new_tickers = {h['ticker'] for h in holdings_new}
    
    added_tickers = new_tickers - old_tickers
    removed_tickers = old_tickers - new_tickers
    maintained_tickers = old_tickers & new_tickers
    
    added_positions = [h for h in holdings_new if h['ticker'] in added_tickers]
    removed_positions = [h for h in holdings_old if h['ticker'] in removed_tickers]
    maintained_positions = [h for h in holdings_new if h['ticker'] in maintained_tickers]
    
    return {
        'added': added_positions,
        'removed': removed_positions,
        'maintained': maintained_positions,
        'added_tickers': sorted(list(added_tickers)),
        'removed_tickers': sorted(list(removed_tickers)),
        'maintained_tickers': sorted(list(maintained_tickers))
    }


def determine_f1_mode(exposure_pct: float):
    """Determine F1 Dashboard mode based on exposure"""
    if exposure_pct >= 80:
        return 'accelerating'
    elif exposure_pct >= 40:
        return 'cruising'
    else:
        return 'braking'


def main():
    parser = argparse.ArgumentParser(description='Export data for AlphaWizzard monthly post')
    parser.add_argument('--month', required=True, help='Month to export (YYYY-MM format, e.g., 2025-12)')
    parser.add_argument('--portfolio-id', type=int, default=8, help='Portfolio ID to export (default: 8)')
    parser.add_argument('--output', default=None, help='Output JSON file path')
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.month, '%Y-%m')
    except ValueError:
        print("Error: Month must be in YYYY-MM format (e.g., 2025-12)")
        return
    
    output_file = args.output or f"monthly_data_{args.month}.json"
    
    print(f"\n{'='*60}")
    print(f"AlphaWizzard Monthly Data Export (v2)")
    print(f"Month: {args.month}")
    print(f"Portfolio ID: {args.portfolio_id}")
    print(f"{'='*60}\n")
    
    start_date, end_date = get_period_dates(args.month)
    ytd_start = get_ytd_start(args.month)
    
    print(f"Report Period: {start_date} to {end_date}")
    print(f"YTD from: {ytd_start}\n")
    
    print("Initializing connections...")
    supabase = get_supabase_client()
    s3_client = get_s3_client()
    
    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'month': args.month,
            'portfolio_id': args.portfolio_id,
            'period_start': start_date,
            'period_end': end_date,
            'ytd_start': ytd_start
        }
    }
    
    # 1. NAV Data
    print("\n1. Fetching NAV data...")
    nav_data = fetch_nav_data(supabase, start_date, end_date, args.portfolio_id)
    ytd_nav_data = fetch_ytd_nav(supabase, ytd_start, end_date, args.portfolio_id)
    export_data['nav_data'] = {
        'period': nav_data,
        'ytd': ytd_nav_data
    }
    
    # 2. Calculate Performance (for conservative variant)
    print("\n2. Calculating performance...")
    export_data['performance'] = {
        'period': calculate_performance(nav_data, 'conservative'),
        'ytd': calculate_performance(ytd_nav_data, 'conservative'),
        'period_raw': calculate_performance(nav_data, 'raw'),
        'ytd_raw': calculate_performance(ytd_nav_data, 'raw')
    }
    
    # 3. Holdings - Report Month
    print("\n3. Fetching holdings for report month...")
    holdings_month_start = fetch_holdings(supabase, start_date, args.portfolio_id)
    holdings_month_end = fetch_holdings(supabase, end_date, args.portfolio_id)
    
    # 4. Holdings - CURRENT (next month / latest)
    print("\n4. Fetching CURRENT holdings (for next month's portfolio)...")
    holdings_current = fetch_current_holdings(supabase, args.portfolio_id)
    
    export_data['holdings'] = {
        'month_start': holdings_month_start,
        'month_end': holdings_month_end,
        'current': holdings_current,
        'current_date': holdings_current[0]['effective_date'] if holdings_current else None
    }
    
    # 5. Portfolio Changes - Within the month
    print("\n5. Identifying portfolio changes...")
    changes_during_month = identify_portfolio_changes(holdings_month_start, holdings_month_end)
    
    # 6. Portfolio Changes - Month End -> Current (NEW FOR NEXT MONTH)
    print("\n6. Identifying changes for NEXT month (month_end -> current)...")
    changes_for_next_month = identify_portfolio_changes(holdings_month_end, holdings_current)
    
    export_data['portfolio_changes'] = {
        'during_month': changes_during_month,
        'for_next_month': changes_for_next_month
    }
    
    # 7. Overlay Signals
    print("\n7. Fetching overlay signals...")
    overlay_history = fetch_overlay_signals(supabase, start_date, end_date)
    current_overlay = fetch_current_overlay(supabase)
    
    current_exposure = current_overlay.get('target_allocation', 1.0) * 100 if current_overlay else 100
    
    export_data['overlay'] = {
        'history': overlay_history,
        'current': current_overlay,
        'current_exposure_pct': current_exposure,
        'f1_mode': determine_f1_mode(current_exposure)
    }
    
    # 8. S3 Data
    print("\n8. Fetching S3 data...")
    export_data['s3_data'] = {
        'allocation_history': fetch_s3_file(s3_client, 'allocation_history.csv'),
        'model_output': fetch_s3_file(s3_client, 'model_output.json'),
        'spy_benchmark': fetch_spy_benchmark(s3_client, start_date, end_date)
    }
    
    # 9. Ticker Info for ALL relevant tickers
    print("\n9. Fetching ticker info...")
    all_tickers = list(set(
        [h['ticker'] for h in holdings_current] + 
        changes_for_next_month['added_tickers'] + 
        changes_for_next_month['removed_tickers'] +
        [h['ticker'] for h in holdings_month_end]
    ))
    ticker_info = fetch_ticker_info(supabase, all_tickers)
    export_data['ticker_info'] = {t['ticker']: t for t in ticker_info}
    
    # 10. EODHD Fundamentals for NEW positions (next month's additions)
    print("\n10. Fetching EODHD fundamentals for NEW positions...")
    fundamentals = {}
    news = {}
    for ticker in changes_for_next_month['added_tickers']:
        fund_data = fetch_eodhd_fundamentals(ticker)
        if fund_data:
            fundamentals[ticker] = fund_data
        news_data = fetch_eodhd_news(ticker)
        if news_data:
            news[ticker] = news_data
    
    export_data['fundamentals'] = fundamentals
    export_data['news'] = news
    
    # 11. Summary stats
    print("\n11. Generating summary...")
    export_data['summary'] = {
        'portfolio_id': args.portfolio_id,
        'report_month': args.month,
        
        # Performance
        'period_return_pct': export_data['performance']['period']['return_pct'] if export_data['performance']['period'] else None,
        'ytd_return_pct': export_data['performance']['ytd']['return_pct'] if export_data['performance']['ytd'] else None,
        'period_max_drawdown_pct': export_data['performance']['period']['max_drawdown_pct'] if export_data['performance']['period'] else None,
        'ytd_max_drawdown_pct': export_data['performance']['ytd']['max_drawdown_pct'] if export_data['performance']['ytd'] else None,
        
        # Holdings counts
        'positions_month_end': len(holdings_month_end),
        'positions_current': len(holdings_current),
        
        # Changes during the month
        'added_during_month': len(changes_during_month['added_tickers']),
        'removed_during_month': len(changes_during_month['removed_tickers']),
        
        # Changes for NEXT month (the key info for the post)
        'new_positions_next_month': changes_for_next_month['added_tickers'],
        'closed_positions_next_month': changes_for_next_month['removed_tickers'],
        'new_count_next_month': len(changes_for_next_month['added_tickers']),
        'closed_count_next_month': len(changes_for_next_month['removed_tickers']),
        
        # Overlay
        'current_exposure_pct': current_exposure,
        'f1_mode': determine_f1_mode(current_exposure),
    }
    
    # Write output
    print(f"\n{'='*60}")
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, cls=DecimalEncoder)
    
    print(f"\n[OK] Export complete!")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"\nReport Month: {args.month}")
    print(f"Portfolio ID: {args.portfolio_id}")
    print(f"\nPERFORMANCE:")
    print(f"   December Return: {export_data['summary']['period_return_pct']}%")
    print(f"   YTD Return: {export_data['summary']['ytd_return_pct']}%")
    print(f"   December Max DD: {export_data['summary']['period_max_drawdown_pct']}%")
    print(f"   YTD Max DD: {export_data['summary']['ytd_max_drawdown_pct']}%")
    print(f"\nF1 DASHBOARD:")
    print(f"   Current Exposure: {current_exposure:.1f}%")
    print(f"   Mode: {determine_f1_mode(current_exposure).upper()}")
    print(f"\nHOLDINGS:")
    print(f"   End of {args.month}: {export_data['summary']['positions_month_end']} positions")
    print(f"   Current (next month): {export_data['summary']['positions_current']} positions")
    print(f"\nNEW FOR NEXT MONTH:")
    if changes_for_next_month['added_tickers']:
        print(f"   Added: {', '.join(changes_for_next_month['added_tickers'])}")
    else:
        print(f"   Added: None")
    print(f"\nCLOSED FOR NEXT MONTH:")
    if changes_for_next_month['removed_tickers']:
        print(f"   Removed: {', '.join(changes_for_next_month['removed_tickers'])}")
    else:
        print(f"   Removed: None")
    print(f"\nOutput file: {output_file}")
    print(f"   Upload this file to Claude to generate the monthly post.\n")


if __name__ == '__main__':
    main()
