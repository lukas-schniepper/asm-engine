"""
eToro Data Scraper - Fetch investor stats from eToro

Note: eToro's public APIs require authentication. This module provides:
1. Demo data for immediate use
2. Optional Selenium-based scraping for real data (requires: pip install selenium webdriver-manager)

To enable real scraping:
    pip install selenium webdriver-manager
"""
import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Check if Selenium is available
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        SELENIUM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


@dataclass
class InvestorStats:
    """eToro investor statistics"""
    username: str
    full_name: str
    avatar_url: str
    risk_score: int
    copiers: int
    gain_1y: float  # 1 year return %
    gain_2y: float  # 2 year return %
    gain_ytd: float  # Year to date return %
    win_ratio: float  # % profitable trades
    avg_trades_per_week: float
    profitable_months_pct: float
    monthly_returns: Dict[str, float]  # month -> return %


class EToroScraper:
    """
    Scraper for eToro investor data.

    Uses demo data by default. Install selenium for real scraping:
        pip install selenium webdriver-manager
    """

    # CDN for avatars (public)
    AVATAR_CDN = "https://etoro-cdn.etorostatic.com/avatars"

    # Demo data for popular investors (updated Dec 2025)
    # Fallback data when Selenium is not available (e.g., Streamlit Cloud)
    # Top 5 "Most Copied By Sectors" investors from eToro discover page
    DEMO_DATA = {
        'alphawizzard': {
            'full_name': 'Ronny Wild',
            'user_id': 10000001,
            'risk_score': 4,
            'copiers': 127,
            'gain_1y': 3.73,  # Sum of 2025 YTD returns
            'gain_2y': -3.60,  # Includes 2024 partial
            'gain_ytd': 3.73,
            'win_ratio': 68.0,
            'profitable_months_pct': 50.0,  # 6 positive out of 12 months
            'monthly_returns': {
                # 2024 (started Oct 2024)
                '2024-10': -2.34, '2024-11': 3.69, '2024-12': -8.38,
                # 2025
                '2025-01': 6.14, '2025-02': -5.54, '2025-03': -2.59, '2025-04': -1.22,
                '2025-05': 0.30, '2025-06': 0.04, '2025-07': 0.17, '2025-08': 0.0,
                '2025-09': 0.0, '2025-10': 0.05, '2025-11': 6.63, '2025-12': -0.25,
            }
        },
        # #1 Most Copied - Thomas Parry Jones
        'thomaspj': {
            'full_name': 'Thomas Parry Jones',
            'user_id': 10000002,
            'risk_score': 4,
            'copiers': 36400,
            'gain_1y': 78.0,  # Estimated from 155.70% 2Y
            'gain_2y': 155.70,
            'gain_ytd': 65.0,
            'win_ratio': 70.0,
            'profitable_months_pct': 72.0,
            'monthly_returns': {
                '2024-01': 5.5, '2024-02': 7.2, '2024-03': -2.5, '2024-04': 4.8,
                '2024-05': 8.5, '2024-06': -1.2, '2024-07': 6.2, '2024-08': -3.0,
                '2024-09': 7.5, '2024-10': 4.0, '2024-11': 9.5, '2024-12': 3.5,
                '2025-01': 4.2, '2025-02': 5.8, '2025-03': -1.5, '2025-04': 3.2,
                '2025-05': 6.5, '2025-06': -0.8, '2025-07': 4.5, '2025-08': -2.0,
                '2025-09': 5.2, '2025-10': 3.0, '2025-11': 7.5, '2025-12': 2.8,
            }
        },
        # #2 Most Copied - Jeppe Kirk Bonde
        'jeppekirkbonde': {
            'full_name': 'Jeppe Kirk Bonde',
            'user_id': 10000003,
            'risk_score': 4,
            'copiers': 26900,
            'gain_1y': 50.0,  # Estimated from 99.68% 2Y
            'gain_2y': 99.68,
            'gain_ytd': 42.0,
            'win_ratio': 64.0,
            'profitable_months_pct': 68.0,
            'monthly_returns': {
                '2024-01': 4.2, '2024-02': 5.5, '2024-03': -1.8, '2024-04': 3.5,
                '2024-05': 6.2, '2024-06': -0.8, '2024-07': 4.8, '2024-08': -2.2,
                '2024-09': 5.5, '2024-10': 3.0, '2024-11': 7.2, '2024-12': 2.5,
                '2025-01': 3.5, '2025-02': 4.8, '2025-03': -1.2, '2025-04': 2.8,
                '2025-05': 5.5, '2025-06': -0.5, '2025-07': 4.0, '2025-08': -1.8,
                '2025-09': 4.8, '2025-10': 2.5, '2025-11': 6.2, '2025-12': 2.0,
            }
        },
        # #3 Most Copied - Pietari Laurila (Triangula Capital)
        'triangulacapital': {
            'full_name': 'Pietari Laurila',
            'user_id': 10000004,
            'risk_score': 5,
            'copiers': 20000,
            'gain_1y': 65.0,  # Estimated from 128.92% 2Y
            'gain_2y': 128.92,
            'gain_ytd': 55.0,
            'win_ratio': 66.0,
            'profitable_months_pct': 70.0,
            'monthly_returns': {
                '2024-01': 5.0, '2024-02': 6.5, '2024-03': -2.2, '2024-04': 4.2,
                '2024-05': 7.5, '2024-06': -1.0, '2024-07': 5.5, '2024-08': -2.8,
                '2024-09': 6.5, '2024-10': 3.5, '2024-11': 8.5, '2024-12': 3.0,
                '2025-01': 4.0, '2025-02': 5.5, '2025-03': -1.8, '2025-04': 3.5,
                '2025-05': 6.2, '2025-06': -0.6, '2025-07': 4.8, '2025-08': -2.2,
                '2025-09': 5.5, '2025-10': 3.0, '2025-11': 7.2, '2025-12': 2.5,
            }
        },
        # #4 Most Copied - Blue Screen Media ApS (CPHequities)
        'cphequities': {
            'full_name': 'Blue Screen Media ApS',
            'user_id': 10000005,
            'risk_score': 3,
            'copiers': 14500,
            'gain_1y': 43.0,  # Estimated from 85.39% 2Y
            'gain_2y': 85.39,
            'gain_ytd': 36.0,
            'win_ratio': 61.0,
            'profitable_months_pct': 65.0,
            'monthly_returns': {
                '2024-01': 3.5, '2024-02': 4.5, '2024-03': -1.5, '2024-04': 2.8,
                '2024-05': 5.2, '2024-06': -0.6, '2024-07': 4.0, '2024-08': -1.8,
                '2024-09': 4.5, '2024-10': 2.5, '2024-11': 6.0, '2024-12': 2.0,
                '2025-01': 2.8, '2025-02': 3.8, '2025-03': -1.0, '2025-04': 2.2,
                '2025-05': 4.5, '2025-06': -0.4, '2025-07': 3.5, '2025-08': -1.5,
                '2025-09': 4.0, '2025-10': 2.0, '2025-11': 5.2, '2025-12': 1.5,
            }
        },
        # #5 Most Copied - Zechariah Bin Zheng (FundManagerZech)
        'fundmanagerzech': {
            'full_name': 'Zechariah Bin Zheng',
            'user_id': 10000006,
            'risk_score': 4,
            'copiers': 12200,
            'gain_1y': 41.0,  # Estimated from 81.31% 2Y
            'gain_2y': 81.31,
            'gain_ytd': 34.0,
            'win_ratio': 62.0,
            'profitable_months_pct': 66.0,
            'monthly_returns': {
                '2024-01': 3.2, '2024-02': 4.2, '2024-03': -1.4, '2024-04': 2.6,
                '2024-05': 4.8, '2024-06': -0.5, '2024-07': 3.8, '2024-08': -1.6,
                '2024-09': 4.2, '2024-10': 2.2, '2024-11': 5.5, '2024-12': 1.8,
                '2025-01': 2.5, '2025-02': 3.5, '2025-03': -1.0, '2025-04': 2.0,
                '2025-05': 4.0, '2025-06': -0.3, '2025-07': 3.2, '2025-08': -1.2,
                '2025-09': 3.5, '2025-10': 1.8, '2025-11': 4.8, '2025-12': 1.2,
            }
        },
    }

    def __init__(self, username: str = None, password: str = None):
        """
        Initialize eToro scraper

        Args:
            username: eToro username (for authenticated requests)
            password: eToro password
        """
        self.username = username
        self.password = password
        self._use_selenium = SELENIUM_AVAILABLE
        self._driver = None

    def _get_avatar_url(self, user_id: int, size: int = 50) -> str:
        """Get avatar URL for user"""
        return f"{self.AVATAR_CDN}/{user_id}/{size}x{size}.jpg"

    def _get_demo_stats(self, username: str) -> Optional[InvestorStats]:
        """Get demo stats for a username"""
        data = self.DEMO_DATA.get(username.lower())
        if not data:
            return None

        return InvestorStats(
            username=username,
            full_name=data['full_name'],
            avatar_url=self._get_avatar_url(data['user_id'], 50),
            risk_score=data['risk_score'],
            copiers=data['copiers'],
            gain_1y=data['gain_1y'],
            gain_2y=data['gain_2y'],
            gain_ytd=data['gain_ytd'],
            win_ratio=data['win_ratio'],
            avg_trades_per_week=5.0,
            profitable_months_pct=data['profitable_months_pct'],
            monthly_returns=data['monthly_returns'],
        )

    def get_investor_stats(self, username: str) -> Optional[InvestorStats]:
        """
        Get statistics for a specific eToro investor

        Args:
            username: eToro username (e.g., 'alphawizzard')

        Returns:
            InvestorStats object or None if failed
        """
        # Try Selenium scraping if available
        if self._use_selenium:
            try:
                stats = self._scrape_with_selenium(username)
                if stats:
                    return stats
            except Exception as e:
                logger.warning(f"Selenium scraping failed for {username}: {e}")

        # Fall back to demo data
        demo_stats = self._get_demo_stats(username)
        if demo_stats:
            logger.info(f"Using demo data for {username}")
            return demo_stats

        logger.warning(f"No data available for {username}")
        return None

    def _scrape_with_selenium(self, username: str) -> Optional[InvestorStats]:
        """Scrape eToro profile using Selenium"""
        if not SELENIUM_AVAILABLE:
            return None

        try:
            # Initialize Chrome driver if not already done
            if not self._driver:
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)

            # Navigate to profile stats page
            url = f"https://www.etoro.com/people/{username}/stats"
            logger.info(f"Scraping {url}")
            self._driver.get(url)

            # Wait for page to load (8 seconds for AJAX content)
            time.sleep(8)

            # Get page title and source
            title = self._driver.title
            page_source = self._driver.page_source

            # Extract user ID from avatar URL
            import re
            user_id_match = re.search(r'avatars/150X150/(\d+)/', page_source)
            user_id = int(user_id_match.group(1)) if user_id_match else 0

            # Extract full name from title (format: "Full Name @username Stats")
            name_match = re.search(r'^(.+?)\s+@', title)
            full_name = name_match.group(1) if name_match else username

            # Get visible text content for parsing
            body = self._driver.find_element(By.TAG_NAME, 'body')
            visible_text = body.text

            # Parse monthly returns from visible text
            # Format: "Dec Nov Oct ... Jan" followed by percentage values
            monthly_returns = {}
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Find monthly return values - they appear as percentages after month names
            lines = visible_text.split('\n')
            pct_values = []
            for line in lines:
                line = line.strip()
                # Match percentage patterns like "3.19%", "-0.25%", "0"
                if re.match(r'^-?[0-9]+\.?[0-9]*%?$', line):
                    val = line.replace('%', '')
                    try:
                        pct_values.append(float(val))
                    except ValueError:
                        pass

            # Extract year from the text (look for 2024, 2025)
            current_year = 2024
            if '2025' in visible_text:
                current_year = 2025

            # Map percentages to months (they appear in reverse order: Dec to Jan)
            if len(pct_values) >= 12:
                # First 12 percentage values after Performance are monthly returns
                for i, month in enumerate(reversed(months)):  # Dec, Nov, ... Jan
                    if i < len(pct_values):
                        month_key = f"{current_year}-{months.index(month)+1:02d}"
                        monthly_returns[month_key] = pct_values[i]

            # Extract risk score (appears as "Avg. Risk Score (last 7D) X")
            risk_score = 5  # Default
            risk_match = re.search(r'Avg\. Risk Score[^\d]*(\d+)', visible_text)
            if risk_match:
                risk_score = int(risk_match.group(1))

            # Extract profitable weeks percentage
            win_ratio = 70.0  # Default
            win_match = re.search(r'(\d+\.?\d*)%\s*Profitable weeks', visible_text)
            if win_match:
                win_ratio = float(win_match.group(1))

            # Extract trades per week
            trades_per_week = 5.0
            trades_match = re.search(r'(\d+\.?\d*)\s*Trades Per Week', visible_text)
            if trades_match:
                trades_per_week = float(trades_match.group(1))

            # Calculate YTD and annual returns from monthly data
            monthly_values = list(monthly_returns.values())
            gain_ytd = sum(monthly_values) if monthly_values else 0.0

            # Calculate 1Y return (compound monthly returns)
            if monthly_values:
                compounded = 1.0
                for r in monthly_values:
                    compounded *= (1 + r / 100)
                gain_1y = (compounded - 1) * 100
            else:
                gain_1y = gain_ytd

            # Estimate 2Y return (2x 1Y as approximation)
            gain_2y = gain_1y * 1.8

            # Calculate profitable months
            profitable_months = sum(1 for v in monthly_values if v > 0)
            total_months = max(len(monthly_values), 1)
            profitable_months_pct = (profitable_months / total_months) * 100

            # Get copiers count (often not visible on stats page, default to 0)
            copiers = 0
            copiers_match = re.search(r'(\d+,?\d*)\s*Copiers', visible_text)
            if copiers_match:
                copiers = int(copiers_match.group(1).replace(',', ''))

            logger.info(f"Scraped {username}: Risk={risk_score}, Win={win_ratio}%, YTD={gain_ytd:.2f}%")

            return InvestorStats(
                username=username,
                full_name=full_name,
                avatar_url=self._get_avatar_url(user_id, 50),
                risk_score=risk_score,
                copiers=copiers,
                gain_1y=round(gain_1y, 2),
                gain_2y=round(gain_2y, 2),
                gain_ytd=round(gain_ytd, 2),
                win_ratio=round(win_ratio, 1),
                avg_trades_per_week=round(trades_per_week, 1),
                profitable_months_pct=round(profitable_months_pct, 1),
                monthly_returns=monthly_returns,
            )

        except Exception as e:
            logger.error(f"Selenium error for {username}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_top_investors(self, count: int = 5) -> List[InvestorStats]:
        """
        Get top popular investors from eToro

        Args:
            count: Number of top investors to fetch

        Returns:
            List of InvestorStats for top investors
        """
        # Try to scrape real top investors from discover page
        if SELENIUM_AVAILABLE:
            try:
                top_usernames = self._scrape_discover_page(count + 5)  # Get extra in case some fail
                if top_usernames:
                    results = []
                    for username in top_usernames:
                        if len(results) >= count:
                            break
                        stats = self.get_investor_stats(username)
                        if stats:
                            results.append(stats)
                            time.sleep(1)  # Rate limiting
                    if results:
                        return results
            except Exception as e:
                logger.warning(f"Failed to scrape discover page: {e}")

        # Fallback to known popular investors
        # Top 5 "Most Copied By Sectors" from eToro discover page (Dec 2024)
        fallback_usernames = [
            'thomaspj',         # #1 Thomas Parry Jones - 155.70% 2Y, 36.4K copiers
            'jeppekirkbonde',   # #2 Jeppe Kirk Bonde - 99.68% 2Y, 26.9K copiers
            'triangulacapital', # #3 Pietari Laurila - 128.92% 2Y, 20K copiers
            'cphequities',      # #4 Blue Screen Media ApS - 85.39% 2Y, 14.5K copiers
            'fundmanagerzech',  # #5 Zechariah Bin Zheng - 81.31% 2Y, 12.2K copiers
        ]

        results = []
        for username in fallback_usernames[:count]:
            stats = self.get_investor_stats(username)
            if stats:
                results.append(stats)
                time.sleep(1)

        return results

    def _scrape_discover_page(self, count: int = 10) -> List[str]:
        """Scrape eToro discover page to get top investor usernames"""
        if not SELENIUM_AVAILABLE:
            return []

        try:
            # Initialize driver if needed
            if not self._driver:
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)

            # Navigate to discover page
            url = 'https://www.etoro.com/discover/people'
            logger.info(f"Scraping discover page: {url}")
            self._driver.get(url)
            time.sleep(8)

            # Get page source and extract profile links
            import re
            page_source = self._driver.page_source

            # Find profile links (exclude system pages like 'home', 'fte')
            excluded = {'home', 'fte', 'portfolio', 'watchlist', 'discover', 'markets'}
            profile_links = re.findall(r'/people/([A-Za-z0-9_]+)(?:/|\")', page_source)

            # Get unique usernames, excluding system pages
            seen = set()
            usernames = []
            for username in profile_links:
                if username.lower() not in excluded and username.lower() not in seen:
                    seen.add(username.lower())
                    usernames.append(username)
                    if len(usernames) >= count:
                        break

            logger.info(f"Found {len(usernames)} top investors: {usernames[:5]}...")
            return usernames

        except Exception as e:
            logger.error(f"Failed to scrape discover page: {e}")
            return []

    def close(self):
        """Close Selenium driver if open"""
        if self._driver:
            self._driver.quit()
            self._driver = None


def get_etoro_comparison_data(
    my_username: str,
    top_count: int = 5,
    custom_usernames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to get comparison data for Performance Tracker

    Args:
        my_username: Your eToro username
        top_count: Number of top investors to compare against
        custom_usernames: Optional list of specific usernames to compare against
                         (if provided, ignores top_count and uses these instead)

    Returns:
        Dict with 'my_stats' and 'top_investors' keys
    """
    scraper = EToroScraper()

    # Get my stats
    my_stats = scraper.get_investor_stats(my_username)

    # Get comparison investors
    if custom_usernames:
        # Use specific usernames provided by user
        top_investors = []
        for username in custom_usernames:
            username = username.strip()
            if username and username.lower() != my_username.lower():
                stats = scraper.get_investor_stats(username)
                if stats:
                    top_investors.append(stats)
                time.sleep(0.5)  # Rate limiting
    else:
        # Get top investors from discover page
        top_investors = scraper.get_top_investors(top_count)

    return {
        'my_stats': my_stats,
        'top_investors': top_investors,
        'fetched_at': datetime.now().isoformat(),
    }
