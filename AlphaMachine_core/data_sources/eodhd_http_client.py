"""
EODHD HTTP API Client - Direct REST API calls without external library
Uses only standard requests library (already in requirements.txt)
"""
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time


class EODHDHttpClient:
    """
    HTTP client for EODHD Financial APIs
    Documentation: https://eodhd.com/financial-apis/
    """

    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        """
        Initialize EODHD HTTP client

        Args:
            api_key: Your EODHD API token (All-in-One plan)
        """
        if not api_key:
            raise ValueError("EODHD API key is required")

        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ASM-Engine/1.0',
            'Accept': 'application/json'
        })

    def _transform_ticker(self, ticker: str) -> str:
        """
        Transform yfinance ticker format to EODHD format

        Examples:
            SPY -> SPY.US
            AAPL -> AAPL.US
            ^GSPC -> GSPC.INDX

        Args:
            ticker: Ticker in yfinance format

        Returns:
            Ticker in EODHD format
        """
        # Remove ^ prefix used for indices in yfinance
        if ticker.startswith('^'):
            ticker = ticker[1:]
            # Most indices use .INDX exchange
            return f"{ticker}.INDX"

        # Default to US exchange for stocks
        return f"{ticker}.US"

    def get_eod_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        retry_count: int = 3,
        retry_delay: float = 2.0
    ) -> pd.DataFrame:
        """
        Fetch End-of-Day historical price data via EODHD API

        API Endpoint: GET https://eodhd.com/api/eod/{SYMBOL}

        Args:
            ticker: Stock ticker in yfinance format (e.g., 'SPY', '^GSPC')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            retry_count: Number of retry attempts on failure
            retry_delay: Seconds to wait between retries

        Returns:
            pandas DataFrame with DatetimeIndex and columns:
            - Open (float)
            - High (float)
            - Low (float)
            - Close (float)
            - Volume (int)

            Returns empty DataFrame if request fails or no data available
        """
        eodhd_symbol = self._transform_ticker(ticker)

        # Build API request URL
        endpoint = f"{self.BASE_URL}/eod/{eodhd_symbol}"
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'from': start_date,
            'to': end_date,
            'period': 'd',      # Daily data
            'order': 'a'        # Ascending (oldest first)
        }

        # Retry loop for network resilience
        last_error = None
        for attempt in range(retry_count):
            try:
                response = self.session.get(
                    endpoint,
                    params=params,
                    timeout=30
                )

                # Success case
                if response.status_code == 200:
                    data = response.json()

                    # Handle empty data response
                    if not data or not isinstance(data, list):
                        print(f"⚠️ No data returned for {ticker} ({eodhd_symbol}) from {start_date} to {end_date}")
                        return pd.DataFrame()

                    # Convert JSON array to DataFrame
                    df = pd.DataFrame(data)

                    if df.empty:
                        print(f"⚠️ Empty DataFrame for {ticker} ({eodhd_symbol})")
                        return pd.DataFrame()

                    # Transform to yfinance-compatible format
                    return self._transform_to_yfinance_format(df, ticker)

                # Error cases
                elif response.status_code == 404:
                    print(f"❌ Ticker {ticker} ({eodhd_symbol}) not found in EODHD database")
                    return pd.DataFrame()

                elif response.status_code == 401:
                    print(f"❌ EODHD API authentication failed - check API key")
                    return pd.DataFrame()

                elif response.status_code == 429:
                    # Rate limit (shouldn't happen with All-in-One plan)
                    wait_time = retry_delay * (attempt + 1)
                    print(f"⚠️ Rate limit reached for {ticker}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    error_msg = response.text[:200] if response.text else "Unknown error"
                    print(f"⚠️ HTTP {response.status_code} for {ticker}: {error_msg}")
                    last_error = f"HTTP {response.status_code}"

                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    return pd.DataFrame()

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                print(f"⚠️ Request timeout for {ticker}, attempt {attempt + 1}/{retry_count}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                print(f"⚠️ Connection error for {ticker}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                print(f"❌ Request error for {ticker}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()

            except ValueError as e:
                # JSON parsing error
                last_error = f"JSON parsing error: {str(e)}"
                print(f"❌ Failed to parse JSON response for {ticker}: {e}")
                return pd.DataFrame()

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                print(f"❌ Unexpected error for {ticker}: {e}")
                return pd.DataFrame()

        # All retries exhausted
        if last_error:
            print(f"❌ Failed to fetch {ticker} after {retry_count} attempts. Last error: {last_error}")
        return pd.DataFrame()

    def _transform_to_yfinance_format(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Transform EODHD response to yfinance-compatible DataFrame format

        EODHD format:
            Columns: date, open, high, low, close, adjusted_close, volume

        yfinance format:
            Index: DatetimeIndex
            Columns: Open, High, Low, Close, Volume (Title case)

        Args:
            df: Raw DataFrame from EODHD API
            ticker: Ticker symbol (for logging)

        Returns:
            Transformed DataFrame compatible with existing AlphaMachine code
        """
        try:
            # Convert date column to datetime and set as index
            df['Date'] = pd.to_datetime(df['date'])
            df = df.set_index('Date')

            # Rename columns to match yfinance (lowercase -> Title case)
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }

            df = df.rename(columns=column_mapping)

            # Select only the columns we need (exclude adjusted_close, date, etc.)
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Check if all required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns for {ticker}: {missing_cols}")
                return pd.DataFrame()

            df = df[required_columns]

            # Convert all columns to numeric, replacing errors with NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with all NaN values
            df = df.dropna(how='all')

            # Sort by date (should already be sorted, but ensure it)
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"❌ Error transforming data for {ticker}: {e}")
            return pd.DataFrame()

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch ticker fundamental data from EODHD Fundamentals API

        API Endpoint: GET https://eodhd.com/api/fundamentals/{SYMBOL}

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with ticker information matching yfinance format
        """
        eodhd_symbol = self._transform_ticker(ticker)

        endpoint = f"{self.BASE_URL}/fundamentals/{eodhd_symbol}"
        params = {
            'api_token': self.api_key
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Extract general info
                general = data.get('General', {})

                # Transform to yfinance-compatible format
                return {
                    'sector': general.get('Sector', 'Unknown'),
                    'industry': general.get('Industry', 'Unknown'),
                    'currency': general.get('CurrencyCode', 'USD'),
                    'country': general.get('CountryISO', 'US'),
                    'exchange': general.get('Exchange', 'US'),
                    'quoteType': general.get('Type', 'Common Stock'),
                    'marketCap': general.get('MarketCapitalization', None),
                    'fullTimeEmployees': general.get('FullTimeEmployees', None),
                    'website': general.get('WebURL', None)
                }

            elif response.status_code == 404:
                print(f"⚠️ Fundamentals not found for {ticker} ({eodhd_symbol})")
                return self._get_placeholder_info()

            else:
                print(f"⚠️ Failed to fetch fundamentals for {ticker}: HTTP {response.status_code}")
                return self._get_placeholder_info()

        except Exception as e:
            print(f"⚠️ Error fetching fundamentals for {ticker}: {e}")
            return self._get_placeholder_info()

    def _get_placeholder_info(self) -> Dict[str, Any]:
        """Return placeholder info when fundamentals API fails"""
        return {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'currency': 'USD',
            'country': 'US',
            'exchange': 'US',
            'quoteType': 'EQUITY',
            'marketCap': None,
            'fullTimeEmployees': None,
            'website': None
        }

    def test_connection(self) -> bool:
        """
        Test EODHD API connection and authentication

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a known ticker (AAPL.US) for recent data
            test_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')

            df = self.get_eod_data('AAPL', test_date, today)

            if not df.empty:
                print(f"✅ EODHD API connection successful")
                print(f"   Test data: {len(df)} rows for AAPL")
                return True
            else:
                print(f"⚠️ EODHD API connection test returned empty data")
                return False

        except Exception as e:
            print(f"❌ EODHD API connection test failed: {e}")
            return False
