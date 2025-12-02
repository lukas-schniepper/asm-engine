# AlphaMachine_core/data_manager.py
import time
import datetime as dt
import pandas as pd
import os
from typing import Dict, Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlmodel import select
from sqlalchemy import func
from decimal import Decimal

from AlphaMachine_core.models import TickerPeriod, TickerInfo, PriceData
from AlphaMachine_core.data_sources.eodhd_http_client import EODHDHttpClient
from AlphaMachine_core.db import get_session
from AlphaMachine_core.tracking.models import PortfolioDefinition, PortfolioHolding

class StockDataManager:
    def __init__(self):
        self.skipped_tickers: List[str] = []

        # Initialize EODHD HTTP client
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

        self.eodhd_client = EODHDHttpClient(api_key)
        print("SUCCESS: EODHD HTTP client initialized successfully")

    def add_tickers_for_period(self, tickers: List[str], period_start_date: str, period_end_date: Optional[str] = None, source_name: str = "manual") -> List[str]:
        start = pd.to_datetime(period_start_date).date()
        end = (pd.to_datetime(period_end_date).date() if period_end_date
            else (pd.to_datetime(start) + pd.offsets.MonthEnd(0)).date())
        created_tickers: List[str] = []
        with get_session() as session:
            for t_str in tickers:
                t = t_str.upper()
                statement = select(TickerPeriod).where(
                    TickerPeriod.ticker    == t,
                    TickerPeriod.start_date == start,
                    TickerPeriod.end_date   == end,
                    TickerPeriod.source     == source_name
                )
                exists = session.exec(statement).first()
                if not exists:
                    obj = TickerPeriod(ticker=t, start_date=start, end_date=end, source=source_name)
                    session.add(obj)
                    created_tickers.append(t)
            if created_tickers:
                session.commit()

        # Also ensure a portfolio exists for this source
        self._ensure_portfolio_exists(source_name, tickers, start)

        return created_tickers

    def _ensure_portfolio_exists(self, source_name: str, tickers: List[str], start_date: dt.date) -> None:
        """Ensure a portfolio exists for this source, creating one if needed."""
        portfolio_name = f"{source_name}_EqualWeight"

        with get_session() as session:
            # Check if portfolio already exists
            existing = session.exec(
                select(PortfolioDefinition).where(PortfolioDefinition.name == portfolio_name)
            ).first()

            if existing:
                # Update holdings with any new tickers
                self._update_portfolio_holdings(session, existing.id, tickers, start_date)
            else:
                # Create new portfolio
                tickers_upper = [t.upper() for t in tickers]
                weights = {t: 1.0 / len(tickers_upper) for t in tickers_upper}

                portfolio = PortfolioDefinition(
                    name=portfolio_name,
                    description=f"Portfolio for {source_name} tickers (equal weight)",
                    config={"tickers": tickers_upper, "weights": weights},
                    source=source_name,
                    start_date=start_date,
                    is_active=True,
                )
                session.add(portfolio)
                session.commit()
                session.refresh(portfolio)

                # Create holdings
                self._update_portfolio_holdings(session, portfolio.id, tickers_upper, start_date)
                print(f"Created portfolio '{portfolio_name}' (id={portfolio.id}) with {len(tickers_upper)} holdings")

    def _update_portfolio_holdings(self, session, portfolio_id: int, tickers: List[str], effective_date: dt.date) -> None:
        """Add or update holdings for a portfolio with equal weights."""
        tickers_upper = [t.upper() for t in tickers]

        # Get existing holdings
        existing_holdings = session.exec(
            select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id)
        ).all()
        existing_tickers = {h.ticker for h in existing_holdings}

        # Calculate new equal weight across all tickers (existing + new)
        all_tickers = existing_tickers.union(set(tickers_upper))
        equal_weight = Decimal(str(round(1.0 / len(all_tickers), 6))) if all_tickers else Decimal("1")

        # Update existing holdings with new weight
        for holding in existing_holdings:
            holding.weight = equal_weight
            session.add(holding)

        # Add new holdings (use weights only, not shares, for normalized NAV calculation)
        new_tickers = set(tickers_upper) - existing_tickers
        for ticker in new_tickers:
            holding = PortfolioHolding(
                portfolio_id=portfolio_id,
                ticker=ticker,
                weight=equal_weight,
                effective_date=effective_date,
                # Note: shares=None forces weight-based NAV calculation (normalized to base 100)
            )
            session.add(holding)

        if new_tickers:
            session.commit()
            print(f"Added {len(new_tickers)} new holdings to portfolio {portfolio_id}")

    def update_ticker_data(self, tickers: Optional[List[str]] = None, history_start: str = '1990-01-01', max_workers: int = 10) -> Dict[str, Any]:
        """
        Update ticker data with optimizations:
        - Batch DB query for last dates (single query instead of per-ticker)
        - Skip up-to-date tickers early
        - Parallel API calls using ThreadPoolExecutor
        - Batch DB inserts
        - Skip ticker info updates (can be run separately)

        Args:
            tickers: List of tickers to update (None = all in DB)
            history_start: Start date for new tickers
            max_workers: Number of parallel API workers (default 10)
        """
        target_tickers_list: List[str]
        if tickers is None:
            with get_session() as session:
                results = session.exec(select(TickerPeriod.ticker).distinct()).all()
                target_tickers_list = sorted([str(ticker_val) for ticker_val in results if ticker_val])
        else:
            target_tickers_list = [t.upper() for t in tickers]

        if not target_tickers_list:
            return {'updated': [], 'skipped': [], 'total': 0, 'details': {}}

        history_dt = pd.to_datetime(history_start).date()
        today = dt.date.today()
        updated_tickers_list: List[str] = []
        ticker_details: Dict[str, Dict[str, Any]] = {}
        self.skipped_tickers = []  # Reset for this run

        # OPTIMIZATION 1: Batch query for all last dates in one DB call
        print(f"📊 Checking last dates for {len(target_tickers_list)} tickers...")
        last_dates: Dict[str, dt.date] = {}
        with get_session() as session:
            # Get max trade_date per ticker in a single query
            stmt = (
                select(PriceData.ticker, func.max(PriceData.trade_date))
                .where(PriceData.ticker.in_(target_tickers_list))
                .group_by(PriceData.ticker)
            )
            results = session.exec(stmt).all()
            for ticker, max_date in results:
                if max_date:
                    last_dates[ticker] = max_date

        # OPTIMIZATION 2: Filter out up-to-date tickers BEFORE making API calls
        tickers_to_update: List[Tuple[str, dt.date]] = []
        for ticker in target_tickers_list:
            last_date = last_dates.get(ticker)
            start_date = (last_date + dt.timedelta(days=1)) if last_date else history_dt

            if start_date > today:
                ticker_details[ticker] = {'status': 'skipped', 'reason': 'Already up-to-date'}
                self.skipped_tickers.append(ticker)
            else:
                tickers_to_update.append((ticker, start_date))
                expected_days = len(pd.bdate_range(start_date, today))
                ticker_details[ticker] = {
                    'status': 'pending',
                    'info': f"{start_date} to {today} (~{expected_days} trading days)"
                }

        print(f"⏭️  Skipping {len(self.skipped_tickers)} up-to-date tickers")
        print(f"🔄 Updating {len(tickers_to_update)} tickers...")

        if not tickers_to_update:
            return {
                'updated': [],
                'skipped': self.skipped_tickers,
                'total': len(target_tickers_list),
                'details': ticker_details
            }

        # OPTIMIZATION 3: Parallel API calls using ThreadPoolExecutor
        def fetch_ticker_data(args: Tuple[str, dt.date]) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
            """Fetch data for a single ticker. Returns (ticker, df, error)."""
            ticker, start_date = args
            try:
                raw = self.eodhd_client.get_eod_data(
                    ticker=ticker,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=(today + dt.timedelta(days=1)).strftime('%Y-%m-%d')
                )
                if raw.empty:
                    return (ticker, None, 'No new data')
                return (ticker, raw, None)
            except Exception as e:
                return (ticker, None, f'API error: {str(e)[:50]}')

        fetched_data: Dict[str, pd.DataFrame] = {}

        print(f"🚀 Fetching data with {max_workers} parallel workers...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_ticker_data, args): args[0] for args in tickers_to_update}
            completed = 0
            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    ticker_result, df, error = future.result()
                    if error:
                        ticker_details[ticker_result] = {'status': 'skipped', 'reason': error}
                        self.skipped_tickers.append(ticker_result)
                    elif df is not None:
                        fetched_data[ticker_result] = df
                        ticker_details[ticker_result]['status'] = 'fetched'
                except Exception as e:
                    ticker_details[ticker] = {'status': 'skipped', 'reason': f'Error: {str(e)[:50]}'}
                    self.skipped_tickers.append(ticker)

                # Progress update every 50 tickers
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Progress: {completed}/{len(tickers_to_update)} ({elapsed:.1f}s)")

        fetch_time = time.time() - start_time
        print(f"✅ Fetched {len(fetched_data)} tickers in {fetch_time:.1f}s")

        # OPTIMIZATION 4: Process and batch insert all data
        if fetched_data:
            print(f"💾 Saving data to database...")
            save_start = time.time()

            all_price_objects: List[PriceData] = []

            for ticker, raw in fetched_data.items():
                try:
                    # Handle MultiIndex columns if present
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker in raw.columns.get_level_values(1):
                            raw = raw.xs(ticker, axis=1, level=1)
                        else:
                            raw.columns = raw.columns.droplevel(0)

                    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in raw.columns for col in expected_cols):
                        ticker_details[ticker] = {'status': 'skipped', 'reason': 'Missing data columns'}
                        self.skipped_tickers.append(ticker)
                        continue

                    df = raw[expected_cols].copy()
                    df.dropna(subset=['Close', 'Volume'], inplace=True)
                    df = df[df['Volume'] > 0]

                    if df.empty:
                        ticker_details[ticker] = {'status': 'skipped', 'reason': 'No valid data after filtering'}
                        self.skipped_tickers.append(ticker)
                        continue

                    df.reset_index(inplace=True)
                    date_col_name = 'Date' if 'Date' in df.columns else 'Datetime' if 'Datetime' in df.columns else None

                    if not date_col_name:
                        ticker_details[ticker] = {'status': 'skipped', 'reason': 'Missing date column'}
                        self.skipped_tickers.append(ticker)
                        continue

                    df['ticker'] = ticker
                    df['trade_date'] = pd.to_datetime(df[date_col_name]).dt.date

                    # Create PriceData objects
                    for _, r in df.iterrows():
                        try:
                            if pd.isna(r['Open']) or pd.isna(r['High']) or pd.isna(r['Low']) or pd.isna(r['Close']) or pd.isna(r['Volume']):
                                continue
                            all_price_objects.append(PriceData(
                                ticker=ticker,
                                trade_date=r['trade_date'],
                                open=float(r['Open']),
                                high=float(r['High']),
                                low=float(r['Low']),
                                close=float(r['Close']),
                                volume=int(r['Volume'])
                            ))
                        except Exception:
                            pass

                    first_date = df['trade_date'].min()
                    last_date = df['trade_date'].max()
                    saved_info = f"{len(df)} days from {first_date} to {last_date}"
                    ticker_details[ticker] = {'status': 'success', 'saved': saved_info}
                    updated_tickers_list.append(ticker)

                except Exception as e:
                    ticker_details[ticker] = {'status': 'skipped', 'reason': f'Processing error: {str(e)[:50]}'}
                    self.skipped_tickers.append(ticker)

            # Batch insert all price data in one transaction
            if all_price_objects:
                with get_session() as session:
                    session.add_all(all_price_objects)
                    session.commit()
                print(f"✅ Saved {len(all_price_objects)} price records for {len(updated_tickers_list)} tickers")

            save_time = time.time() - save_start
            print(f"💾 Database save completed in {save_time:.1f}s")

        total_time = time.time() - start_time
        print(f"\n⏱️  Total update time: {total_time:.1f}s")

        return {
            'updated': updated_tickers_list,
            'skipped': self.skipped_tickers,
            'total': len(target_tickers_list),
            'details': ticker_details
        }

    def _update_ticker_info(self, ticker: str) -> bool:
        print(f"Versuche Ticker-Info für {ticker} zu aktualisieren...")
        try:
            info = self.eodhd_client.get_ticker_info(ticker)

            if not info:
                print(f"WARNING: Keine Info von EODHD fuer {ticker} erhalten.")
                return False
            
            data_to_update: Dict[str, Any] = {
                'ticker': ticker,
                'sector': str(info.get('sector', 'N/A'))[:255] if info.get('sector') else None,
                'industry': str(info.get('industry', 'N/A'))[:255] if info.get('industry') else None,
                'currency': str(info.get('currency', 'N/A'))[:10] if info.get('currency') else None,
                'country': str(info.get('country', 'N/A'))[:255] if info.get('country') else None,
                'exchange': str(info.get('exchange', 'N/A'))[:50] if info.get('exchange') else None,
                'quote_type': str(info.get('quoteType', 'N/A'))[:50] if info.get('quoteType') else None,
                # Sicherer Zugriff auf potenziell fehlende Keys
                'market_cap': float(info.get('marketCap')) if info.get('marketCap') is not None else None,
                'employees': int(info.get('fullTimeEmployees')) if info.get('fullTimeEmployees') is not None else None,
                'website': str(info.get('website', 'N/A'))[:255] if info.get('website') else None,
                'last_update': dt.date.today()
            }

            with get_session() as session:
                date_range_statement = select(func.min(PriceData.trade_date), func.max(PriceData.trade_date)).where(PriceData.ticker == ticker)
                date_range_result = session.exec(date_range_statement).first()
                
                if date_range_result and date_range_result[0] is not None:
                    data_to_update['actual_start_date'] = date_range_result[0]
                    data_to_update['actual_end_date'] = date_range_result[1]
                else: 
                    data_to_update['actual_start_date'] = None
                    data_to_update['actual_end_date'] = None
                
                db_ticker_info = session.exec(select(TickerInfo).where(TickerInfo.ticker == ticker)).first()
                if db_ticker_info:
                    for key, value in data_to_update.items(): setattr(db_ticker_info, key, value)
                    print(f"TickerInfo für {ticker} aktualisiert.")
                else:
                    db_ticker_info = TickerInfo(**data_to_update)
                    session.add(db_ticker_info)
                    print(f"TickerInfo für {ticker} neu erstellt.")
                session.commit()
            return True
        except Exception as e: # Breiterer Exception-Fang für yfinance-Info-Probleme
            print(f"WARNING: Fehler _update_ticker_info fuer {ticker}: {e}")
            return False

    def get_periods(self, month: str, source: str) -> List[Dict[str, Any]]:
        period_dicts = []
        with get_session() as session:
            statement = select(TickerPeriod).where(
                func.to_char(TickerPeriod.start_date, 'YYYY-MM') == month,
                TickerPeriod.source == source
            )
            results = session.exec(statement).all()
            for db_obj in results: period_dicts.append(db_obj.model_dump()) 
        return period_dicts

    def get_ticker_info(self) -> List[Dict[str, Any]]:
        info_dicts = []
        with get_session() as session:
            results = session.exec(select(TickerInfo)).all()
            for db_obj in results: info_dicts.append(db_obj.model_dump())
        return info_dicts
    
    def get_price_data(self, tickers: List[str], start_date: Optional[dt.date], end_date: Optional[dt.date]) -> List[Dict[str, Any]]:
        sd = pd.to_datetime(start_date).date() if start_date else None
        ed = pd.to_datetime(end_date).date() if end_date else None
        price_data_dicts = []
        with get_session() as session:
            if not tickers: return []
            upper_tickers = [t.upper() for t in tickers]
            statement = select(PriceData).where(PriceData.ticker.in_(upper_tickers))
            if sd: statement = statement.where(PriceData.trade_date >= sd)
            if ed: statement = statement.where(PriceData.trade_date <= ed)
            statement = statement.order_by(PriceData.ticker, PriceData.trade_date)
            results = session.exec(statement).all()
            for record in results: price_data_dicts.append(record.model_dump())
        if not price_data_dicts and tickers: print(f"SDM.get_price_data: Keine Daten für {tickers} im Zeitraum {sd} bis {ed}")
        return price_data_dicts

    def delete_period(self, period_id: int) -> bool:
        with get_session() as session:
            obj = session.get(TickerPeriod, period_id) # get() erwartet den Primärschlüssel
            if obj: session.delete(obj); session.commit(); return True
        return False

    def get_periods_distinct_months(self) -> List[str]:
        with get_session() as session:
            results = session.exec(select(func.to_char(TickerPeriod.start_date, 'YYYY-MM')).distinct()).all()
        return [str(row) for row in results if row is not None]

    def get_tickers_for(self, month: str, sources: List[str]) -> List[str]:
        with get_session() as session:
            results = session.exec(
                select(TickerPeriod.ticker).distinct()
                .where(
                    func.to_char(TickerPeriod.start_date, 'YYYY-MM') == month,
                    TickerPeriod.source.in_(sources)
                )
            ).all()
        return [str(row) for row in results if row is not None]